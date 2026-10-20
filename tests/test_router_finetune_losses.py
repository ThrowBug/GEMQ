import pytest

torch = pytest.importorskip("torch")

from proquant.router_finetune.losses import compute_causal_output_distill_ce


def test_causal_distillation_ignores_last_position():
    torch.manual_seed(1)
    teacher = torch.randn(2, 4, 7)
    student = teacher.clone()
    changed_last = student.clone()
    changed_last[:, -1, 0] += 100.0
    assert torch.allclose(
        compute_causal_output_distill_ce(student, teacher),
        compute_causal_output_distill_ce(changed_last, teacher),
    )


def test_causal_mask_requires_both_adjacent_tokens():
    teacher = torch.zeros(1, 4, 3)
    student = teacher.clone()
    student[0, 0, 0] = 50.0
    student[0, 2, 0] = 50.0
    mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)
    baseline = compute_causal_output_distill_ce(student, teacher, mask)
    student[0, 1, 0] = 50.0
    changed = compute_causal_output_distill_ce(student, teacher, mask)
    assert changed > baseline


def test_soft_ce_matches_hard_ce_for_one_hot_teacher():
    student = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.0, 3.0, -2.0], [4.0, -1.0, 0.0]]]
    )
    teacher = torch.full_like(student, -100.0)
    teacher[0, 0, 1] = 100.0
    teacher[0, 1, 2] = 100.0
    teacher[0, 2, 0] = 100.0
    soft_ce = compute_causal_output_distill_ce(student, teacher)
    hard_ce = torch.nn.functional.cross_entropy(
        student[:, :-1, :].reshape(-1, student.shape[-1]),
        torch.tensor([1, 2]),
    )
    assert torch.allclose(soft_ce, hard_ce)
