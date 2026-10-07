import unittest


class NeedsTorch(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch is not installed")


class RouteScenariosTest(NeedsTorch):
    def test_original_and_rerouted_topk(self):
        import torch
        from gemq.measure_pop_layer_error import route_scenarios

        logits = torch.tensor([[6., 5., 4., 3., 2., 1.]])
        indices, original, rerouted = route_scenarios(logits, top_k=4, removed=2)
        self.assertEqual(indices.tolist(), [[0, 1, 2, 3, 4, 5]])
        self.assertAlmostEqual(original.sum().item(), 1.0, places=6)
        self.assertAlmostEqual(rerouted.sum().item(), 1.0, places=6)
        self.assertEqual(indices[:, 2:6].tolist(), [[2, 3, 4, 5]])
        self.assertTrue(torch.all(rerouted >= 0))

    def test_bad_topk(self):
        import torch
        from gemq.measure_pop_layer_error import route_scenarios

        with self.assertRaises(ValueError):
            route_scenarios(torch.zeros(1, 4), top_k=4, removed=2)


class ErrorStatisticsTest(NeedsTorch):
    def test_paired_sums_and_aggregate_ratio(self):
        import torch
        from gemq.measure_pop_layer_error import aggregate, batch_error_sums

        fp = torch.tensor([[[1., 0.], [0., 2.]]])
        low = torch.tensor([[[0., 0.], [0., 1.]]])
        pop = torch.tensor([[[0.5, 0.], [0., 1.5]]])
        row = batch_error_sums(fp, low, pop)
        self.assertEqual(row["tokens"], 2)
        self.assertEqual(row["pop_wins"], 2)
        self.assertAlmostEqual(row["reference_sq"], 5.)
        self.assertAlmostEqual(row["low_bit_sq"], 2.)
        self.assertAlmostEqual(row["pop_sq"], 0.5)
        result = aggregate([row])
        self.assertAlmostEqual(result["low_bit_relative_mse"], 0.4)
        self.assertAlmostEqual(result["pop_relative_mse"], 0.1)
        self.assertAlmostEqual(result["pop_to_low_bit_error_ratio"], 0.25)


class BatchedExpertEvaluationTest(NeedsTorch):
    def test_reference_hybrid_and_reroute_match_scalar_construction(self):
        import torch
        from torch import nn
        from gemq.measure_pop_layer_error import evaluate_moe_batch, route_scenarios

        class Expert(nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.register_buffer("weight", torch.tensor(float(weight)))

            def forward(self, x):
                return x * self.weight

        class Moe(nn.Module):
            def __init__(self):
                super().__init__()
                self.top_k = 8
                self.norm_topk_prob = True
                self.gate = nn.Linear(2, 10, bias=False)
                with torch.no_grad():
                    self.gate.weight[:, 0] = torch.arange(10, 0, -1)
                    self.gate.weight[:, 1] = 0
                self.experts = nn.ModuleList([Expert(i + 1) for i in range(10)])

            def forward(self, hidden):
                flat = hidden.reshape(-1, 2)
                ids, weights, _ = route_scenarios(self.gate(flat))
                output = torch.zeros_like(flat)
                for token in range(flat.shape[0]):
                    for rank in range(self.top_k):
                        expert_id = int(ids[token, rank])
                        output[token] += weights[token, rank] * self.experts[expert_id](flat[token])
                return output.reshape_as(hidden)

        moe = Moe()
        one = nn.ModuleList([Expert((i + 1) * 0.5) for i in range(10)])
        two = nn.ModuleList([Expert((i + 1) * 0.8) for i in range(10)])
        hidden = torch.tensor([[[1., 2.], [2., 1.]]])
        fp, hybrid, pop = evaluate_moe_batch(moe, one, two, hidden, expert_batch_size=1)
        ids, original, rerouted = route_scenarios(moe.gate(hidden.reshape(-1, 2)))
        expected = [torch.zeros_like(hidden.reshape(-1, 2)) for _ in range(3)]
        for token in range(2):
            for rank in range(8):
                expert = int(ids[token, rank])
                value = hidden.reshape(-1, 2)[token]
                expected[0][token] += original[token, rank] * moe.experts[expert](value)
                source = one if rank < 2 else two
                expected[1][token] += original[token, rank] * source[expert](value)
                rerouted_expert = int(ids[token, rank + 2])
                expected[2][token] += rerouted[token, rank] * two[rerouted_expert](value)
        for actual, wanted in zip((fp, hybrid, pop), expected):
            self.assertTrue(torch.allclose(actual.reshape(-1, 2), wanted, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
