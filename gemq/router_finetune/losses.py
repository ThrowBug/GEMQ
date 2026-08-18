import torch
import torch.nn.functional as F


def _flatten_logits(student_logits, teacher_logits):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/teacher router shapes differ: {student_logits.shape} vs {teacher_logits.shape}."
        )
    if student_logits.ndim < 2:
        raise ValueError("Router logits must have an expert dimension.")
    num_experts = student_logits.shape[-1]
    return (
        student_logits.reshape(-1, num_experts).float(),
        teacher_logits.reshape(-1, num_experts).float(),
    )


def _masked_mean(per_token_loss, token_mask=None):
    if token_mask is None:
        return per_token_loss.mean()
    mask = token_mask.reshape(-1).to(device=per_token_loss.device, dtype=per_token_loss.dtype)
    if mask.numel() != per_token_loss.numel():
        raise ValueError("Token mask does not match the number of router tokens.")
    return (per_token_loss * mask).sum() / mask.sum().clamp_min(1.0)


def _teacher_top_indices(teacher_logits, top_m):
    if not 0 < top_m <= teacher_logits.shape[-1]:
        raise ValueError(f"Invalid top_m={top_m} for {teacher_logits.shape[-1]} experts.")
    return torch.topk(teacher_logits, k=top_m, dim=-1, sorted=False).indices


def router_kd_loss(student_logits, teacher_logits, top_m, token_mask=None):
    """KL between teacher-selected, conditionally renormalized top-m distributions."""
    student, teacher = _flatten_logits(student_logits, teacher_logits)
    indices = _teacher_top_indices(teacher, top_m)
    student_selected = student.gather(-1, indices)
    teacher_selected = teacher.gather(-1, indices)

    teacher_log_probs = F.log_softmax(teacher_selected, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student_selected, dim=-1)
    per_token = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
    return _masked_mean(per_token, token_mask)


def router_kd_tail_loss(student_logits, teacher_logits, top_m, token_mask=None):
    """KL on teacher top-m experts plus one bucket containing all remaining mass."""
    student, teacher = _flatten_logits(student_logits, teacher_logits)
    num_experts = teacher.shape[-1]
    if top_m == num_experts:
        return router_kd_loss(student, teacher, top_m, token_mask)

    indices = _teacher_top_indices(teacher, top_m)
    teacher_log_probs = F.log_softmax(teacher, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    teacher_top_log = teacher_log_probs.gather(-1, indices)
    student_top_log = student_log_probs.gather(-1, indices)
    teacher_top_probs = teacher_top_log.exp()

    selected = torch.zeros_like(teacher, dtype=torch.bool)
    selected.scatter_(dim=-1, index=indices, value=True)
    teacher_tail_log = torch.logsumexp(teacher_log_probs.masked_fill(selected, -torch.inf), dim=-1)
    student_tail_log = torch.logsumexp(student_log_probs.masked_fill(selected, -torch.inf), dim=-1)
    teacher_tail_prob = teacher_tail_log.exp()

    top_term = torch.sum(teacher_top_probs * (teacher_top_log - student_top_log), dim=-1)
    tail_term = teacher_tail_prob * (teacher_tail_log - student_tail_log)
    return _masked_mean(top_term + tail_term, token_mask)


def router_l2_loss(student_logits, teacher_logits, top_m, token_mask=None, center=False):
    student, teacher = _flatten_logits(student_logits, teacher_logits)
    indices = _teacher_top_indices(teacher, top_m)
    student_selected = student.gather(-1, indices)
    teacher_selected = teacher.gather(-1, indices)
    if center:
        student_selected = student_selected - student_selected.mean(dim=-1, keepdim=True)
        teacher_selected = teacher_selected - teacher_selected.mean(dim=-1, keepdim=True)
    per_token = (student_selected - teacher_selected).square().mean(dim=-1)
    return _masked_mean(per_token, token_mask)


def compute_router_loss(loss_type, student_logits, teacher_logits, top_m, token_mask=None):
    if loss_type == "kd":
        return router_kd_loss(student_logits, teacher_logits, top_m, token_mask)
    if loss_type == "kd_tail":
        return router_kd_tail_loss(student_logits, teacher_logits, top_m, token_mask)
    if loss_type == "l2":
        return router_l2_loss(student_logits, teacher_logits, top_m, token_mask, center=False)
    if loss_type == "l2_center":
        return router_l2_loss(student_logits, teacher_logits, top_m, token_mask, center=True)
    raise ValueError(f"Unsupported router loss: {loss_type}")


def compute_output_kl(student_logits, teacher_logits, token_mask=None):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/teacher output shapes differ: {student_logits.shape} vs {teacher_logits.shape}."
        )
    vocab_size = student_logits.shape[-1]
    student = student_logits.reshape(-1, vocab_size).float()
    teacher = teacher_logits.reshape(-1, vocab_size).float()
    teacher_log_probs = F.log_softmax(teacher, dim=-1)
    teacher_probs = teacher_log_probs.exp()
    student_log_probs = F.log_softmax(student, dim=-1)
    per_token = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
    return _masked_mean(per_token, token_mask)
