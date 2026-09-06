import pytest

torch = pytest.importorskip("torch")

from gemq.router_finetune.losses import (  # noqa: E402
    compute_causal_output_distill_ce,
    compute_causal_output_cakld,
    compute_causal_output_kl,
    compute_output_kl,
    compute_output_cakld,
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


def test_causal_output_losses_ignore_the_last_position():
    torch.manual_seed(1)
    teacher = torch.randn(2, 4, 7)
    baseline = teacher.clone()
    changed_last = baseline.clone()
    changed_last[:, -1, 0] += 100.0

    baseline_ce = compute_causal_output_distill_ce(baseline, teacher)
    changed_ce = compute_causal_output_distill_ce(changed_last, teacher)
    changed_kl = compute_causal_output_kl(changed_last, teacher)

    assert torch.allclose(changed_ce, baseline_ce)
    assert changed_kl.item() == pytest.approx(0.0, abs=1e-6)


def test_causal_output_mask_requires_both_adjacent_tokens():
    teacher = torch.zeros(1, 4, 3)
    student = teacher.clone()
    student[0, 0, 0] = 50.0
    student[0, 2, 0] = 50.0
    attention_mask = torch.tensor([[0, 1, 1, 0]], dtype=torch.bool)

    # Only prediction position 1 has both its current and next token unmasked.
    loss = compute_causal_output_kl(student, teacher, attention_mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)

    student[0, 1, 0] = 50.0
    loss = compute_causal_output_kl(student, teacher, attention_mask)
    assert loss.item() > 1.0


def test_distill_ce_and_kl_have_identical_student_gradients():
    torch.manual_seed(2)
    teacher = torch.randn(2, 4, 9)
    student_ce = torch.randn(2, 4, 9, requires_grad=True)
    student_kl = student_ce.detach().clone().requires_grad_(True)

    ce = compute_causal_output_distill_ce(student_ce, teacher)
    kl = compute_causal_output_kl(student_kl, teacher)
    ce_grad = torch.autograd.grad(ce, student_ce)[0]
    kl_grad = torch.autograd.grad(kl, student_kl)[0]

    assert torch.allclose(ce_grad, kl_grad, atol=1e-7, rtol=1e-6)


def test_distill_ce_reduces_to_hard_ce_for_one_hot_teacher():
    student = torch.tensor(
        [[[2.0, 0.0, -1.0], [0.0, 3.0, -2.0], [4.0, -1.0, 0.0]]]
    )
    teacher = torch.full_like(student, -100.0)
    teacher[0, 0, 1] = 100.0
    teacher[0, 1, 2] = 100.0
    teacher[0, 2, 0] = 100.0

    soft_ce = compute_causal_output_distill_ce(student, teacher)
    hard_targets = torch.tensor([[1, 2]])
    hard_ce = torch.nn.functional.cross_entropy(
        student[:, :-1, :].reshape(-1, student.shape[-1]), hard_targets.reshape(-1)
    )
    assert torch.allclose(soft_ce, hard_ce)


@pytest.mark.parametrize("gamma", [0.0, 0.3, 1.0])
def test_cakld_matches_explicit_kl_values_and_gradients(gamma):
    teacher = torch.tensor([[0.7, 0.2, 0.1]]).log().requires_grad_(True)
    student = torch.tensor([[0.1, 0.3, 0.6]]).log().requires_grad_(True)
    loss, forward, reverse = compute_output_cakld(student, teacher, gamma, return_components=True)
    reference_student = student.detach().clone().requires_grad_(True)
    pt = teacher.detach().softmax(-1)
    ps = reference_student.softmax(-1)
    expected_forward = (pt * (pt.log() - ps.log())).sum(-1).mean()
    expected_reverse = (ps * (ps.log() - pt.log())).sum(-1).mean()
    expected = (1 - gamma) * expected_forward + gamma * expected_reverse
    assert torch.allclose(loss, expected, atol=1e-7)
    assert torch.allclose(forward, expected_forward)
    assert torch.allclose(reverse, expected_reverse)
    expected_grad = torch.autograd.grad(expected, reference_student)[0]
    loss.backward()
    assert torch.allclose(student.grad, expected_grad, atol=1e-7)
    assert teacher.grad is None
    assert not forward.requires_grad and not reverse.requires_grad


def test_cakld_zero_gamma_matches_ce_gradient_with_causal_mask():
    torch.manual_seed(3)
    teacher = torch.randn(2, 5, 7)
    student = torch.randn(2, 5, 7, requires_grad=True)
    mask = torch.tensor([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    ce = compute_causal_output_distill_ce(student, teacher, mask)
    cakld = compute_causal_output_cakld(student, teacher, 0.0, mask)
    ce_grad = torch.autograd.grad(ce, student)[0]
    cakld_grad = torch.autograd.grad(cakld, student)[0]
    assert torch.allclose(ce_grad, cakld_grad, atol=1e-7)
    assert torch.count_nonzero(cakld_grad[:, -1]) == 0


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_cakld_identical_logits_and_padding_have_zero_loss(gamma):
    teacher = torch.randn(1, 4, 5)
    student = teacher.clone()
    student[:, 0, 0] += 80
    student[:, 2:, 0] += 80
    mask = torch.tensor([[0, 1, 1, 0]])
    assert compute_causal_output_cakld(student, teacher, gamma, mask).item() == pytest.approx(0, abs=1e-6)


def test_cakld_large_finite_logits_have_finite_gradients():
    teacher = torch.tensor([[1000.0, -1000.0, 0.0]])
    student = torch.tensor([[-1000.0, 1000.0, 0.0]], requires_grad=True)
    loss = compute_output_cakld(student, teacher, 0.6)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()


def test_cakld_rejects_unresolved_gamma():
    with pytest.raises(ValueError, match="resolved numeric"):
        compute_output_cakld(torch.zeros(1, 3), torch.zeros(1, 3), "auto")
