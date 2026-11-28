import csv
from unittest.mock import patch

import numpy as np

from gemq import compare_token_routing as comparison


def test_topk_overlap_ignores_expert_order():
    before = np.arange(8).reshape(1, 1, 8)
    same_set = before[..., ::-1]
    one_replaced = before.copy()
    one_replaced[..., 0] = 8

    assert comparison.layer_overlap_counts(before, same_set).tolist() == [8]
    assert comparison.layer_overlap_counts(before, one_replaced).tolist() == [7]


def test_one_baseline_and_three_ordered_rows(tmp_path):
    names = ("gptq", "roco", "router-ft", "norm-ft")
    paths = [tmp_path / name for name in names]
    for path in paths:
        path.mkdir()
    out = tmp_path / "agreement.csv"
    sequences = [(0, [2], 1), (1, [3, 4, 5], 3)]
    layout = comparison.RoutingLayout((0, 17), 128, 8)
    profiled = []
    materialized = []

    def profile(path, inputs, args, consume, expected_layout=None):
        profiled.append(path.name)
        materialized.append(inputs)
        assert expected_layout == (None if path == paths[0] else layout)
        for index, ids in enumerate(inputs):
            routes = np.broadcast_to(np.arange(8), (2, len(ids), 8)).copy()
            if path == paths[1]:
                routes[0, :, 0] = 8
            elif path == paths[2]:
                if len(ids) == 3:
                    routes[1] += 8
            elif path == paths[3]:
                routes += 8
            consume(index, routes[..., ::-1])
        return layout

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.empty_cache"),
        patch("transformers.AutoTokenizer.from_pretrained"),
        patch.object(comparison, "load_sequences", return_value=iter(sequences)),
        patch.object(comparison, "profile_model", side_effect=profile),
    ):
        comparison.main([
            "--before_model_path", str(paths[0]),
            "--after_model_paths", *(str(path) for path in paths[1:]),
            "--labels", "RoCo", "Router-FT", "Norm-FT",
            "--dataset", "c4", "--nsamples", "2", "--output_csv", str(out),
        ])

    assert profiled == list(names)
    assert materialized[0] == [[2], [3, 4, 5]]
    assert all(inputs is materialized[0] for inputs in materialized)
    with out.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert rows == [
        ["model_label", "0", "17"],
        ["RoCo", "87.500000", "100.000000"],
        ["Router-FT", "100.000000", "25.000000"],
        ["Norm-FT", "0.000000", "0.000000"],
    ]
