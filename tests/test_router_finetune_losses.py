import pytest

torch = pytest.importorskip("torch")

from gemq.router_finetune.losses import (  # noqa: E402
    compute_output_kl,
    compute_router_loss,
)


def test_identical_router_distributions_have_zero_kl():
    teacher = torch.tensor([[[3.0, 2.0, 1.0, -1.0]]])
    for loss_type in ("kd", "kd_tail"):
        loss = compute_router_loss(loss_type, teacher.clone(), teacher, top_m=2)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_kd_uses_teacher_top_m_not_student_top_m():
    teacher = torch.tensor([[[5.0, 4.0, 0.0, -1.0]]])
    student = torch.tensor([[[-5.0, -4.0, 10.0, 9.0]]], requires_grad=True)
    loss = compute_router_loss("kd", student, teacher, top_m=2)

    teacher_selected = torch.softmax(teacher[..., :2], dim=-1)
    student_selected_log = torch.log_softmax(student[..., :2], dim=-1)
    expected = torch.sum(
        teacher_selected * (torch.log(teacher_selected) - student_selected_log), dim=-1
    ).mean()
    assert torch.allclose(loss, expected)


def test_tail_bucket_penalizes_probability_outside_teacher_top_m():
    teacher = torch.tensor([[[5.0, 4.0, -5.0, -6.0]]])
    # The first two logits have the same conditional distribution as the teacher,
    # while almost all full-distribution mass is moved to a tail expert.
    student = torch.tensor([[[5.0, 4.0, 12.0, -6.0]]])
    conditional = compute_router_loss("kd", student, teacher, top_m=2)
    with_tail = compute_router_loss("kd_tail", student, teacher, top_m=2)
    assert conditional.item() == pytest.approx(0.0, abs=1e-7)
    assert with_tail.item() > 1.0


def test_kd_and_kd_tail_match_when_all_experts_are_kept():
    torch.manual_seed(0)
    teacher = torch.randn(2, 3, 8)
    student = torch.randn(2, 3, 8)
    kd = compute_router_loss("kd", student, teacher, top_m=8)
    kd_tail = compute_router_loss("kd_tail", student, teacher, top_m=8)
    assert torch.allclose(kd, kd_tail, atol=1e-7, rtol=1e-6)


def test_centered_l2_ignores_common_logit_shift():
    teacher = torch.tensor([[[4.0, 3.0, 2.0, 1.0]]])
    student = teacher + 7.0
    raw = compute_router_loss("l2", student, teacher, top_m=4)
    centered = compute_router_loss("l2_center", student, teacher, top_m=4)
    assert raw.item() == pytest.approx(49.0)
    assert centered.item() == pytest.approx(0.0, abs=1e-7)


def test_router_loss_only_backpropagates_to_student():
    teacher = torch.randn(2, 4, 8)
    student = torch.randn(2, 4, 8, requires_grad=True)
    loss = compute_router_loss("kd_tail", student, teacher, top_m=3)
    loss.backward()
    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
    assert teacher.grad is None


def test_output_kl_is_zero_for_identical_logits_and_honors_mask():
    teacher = torch.randn(2, 3, 11)
    student = teacher.clone()
    student[1, 2, 0] += 100.0
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    loss = compute_output_kl(student, teacher, token_mask=mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
