import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from gemq.router_finetune.losses import compute_sparse_router_kl  # noqa: E402
from gemq.router_finetune.pruned_expert_reroute import (  # noqa: E402
    build_sparse_targets,
    compute_layer_transfer,
    transfer_weights,
    weighted_output_error,
)


def test_weighted_output_error_and_transfer_weights():
    reference = torch.tensor([[1.0], [2.0]])
    weights = torch.tensor([1.0, 0.5])
    assert weighted_output_error(reference, reference, weights) == pytest.approx(0.0)
    assert weighted_output_error(reference, torch.zeros_like(reference), weights) == pytest.approx(1.0)
    probabilities = transfer_weights([0.0, 1.0, 2.0])
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert probabilities[0] > probabilities[1] > probabilities[2]


def test_sparse_target_moves_deleted_expert_mass_and_normalizes_topk():
    # Old experts 0 and 2 survive; old expert 1 moves to new expert 1.
    transfer = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    routes = torch.tensor([[[1, 0], [2, 0]]])
    weights = torch.tensor([[[0.7, 0.3], [0.6, 0.4]]])
    indices, probabilities = build_sparse_targets(routes, weights, transfer, top_k=2)
    dense = torch.zeros(1, 2, 2).scatter_(-1, indices, probabilities)
    torch.testing.assert_close(dense, torch.tensor([[[0.3, 0.7], [0.4, 0.6]]]))


def test_sparse_kl_matches_dense_kl_and_masks_padding():
    student = torch.tensor(
        [[[0.2, -0.1, 0.3], [9.0, -9.0, 0.0]]], requires_grad=True
    )
    indices = torch.tensor([[[0, 2], [1, 2]]])
    weights = torch.tensor([[[0.25, 0.75], [0.5, 0.5]]])
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    loss = compute_sparse_router_kl(student, indices, weights, mask)
    expected = (
        weights[0, 0]
        * (weights[0, 0].log() - student[0, 0].log_softmax(-1)[indices[0, 0]])
    ).sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(student.grad[0, 1], torch.zeros(3))


class _TinyMoe(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(1, 1, bias=False) for _ in range(2)])
        self.gate = nn.Linear(1, 2)
        self.top_k = 1
        with torch.no_grad():
            self.experts[0].weight.zero_()
            self.experts[1].weight.fill_(1.0)


def test_transfer_screens_survivors_and_uses_disjoint_cost_samples():
    moe = _TinyMoe()
    row = {
        "screen_inputs": torch.tensor([[1.0], [2.0]]),
        "screen_reference": torch.tensor([[1.0], [2.0]]),
        "screen_weights": torch.ones(2),
        "cost_inputs": torch.tensor([[3.0], [4.0]]),
        "cost_reference": torch.tensor([[3.0], [4.0]]),
        "cost_weights": torch.ones(2),
        "seen": 4,
    }
    transfer, report = compute_layer_transfer(
        moe, {"samples": {1: row}}, kept_old_ids=(0, 2), layer_idx=0
    )
    torch.testing.assert_close(transfer, torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
    assert report["experts"]["1"]["candidate_old_ids"] == [2]
    assert report["experts"]["1"]["costs"][0] == pytest.approx(0.0)
