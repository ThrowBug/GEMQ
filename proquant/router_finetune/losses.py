"""Teacher soft-label causal cross-entropy for norm fine-tuning."""

import torch
import torch.nn.functional as F


def _masked_mean(per_token_loss, token_mask=None):
    if token_mask is None:
        return per_token_loss.mean()
    mask = token_mask.reshape(-1).to(device=per_token_loss.device, dtype=per_token_loss.dtype)
    if mask.numel() != per_token_loss.numel():
        raise ValueError("Token mask does not match the number of router tokens.")
    return (per_token_loss * mask).sum() / mask.sum().clamp_min(1.0)


def compute_output_distill_ce(student_logits, teacher_logits, token_mask=None):
    """Cross-entropy from a fixed teacher vocabulary distribution to the student."""
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/teacher output shapes differ: {student_logits.shape} vs {teacher_logits.shape}."
        )
    if student_logits.device != teacher_logits.device:
        raise ValueError(
            "Student/teacher output logits must be on the same device, got "
            f"{student_logits.device} and {teacher_logits.device}."
        )
    vocab_size = student_logits.shape[-1]
    student = student_logits.reshape(-1, vocab_size).float()
    teacher = teacher_logits.reshape(-1, vocab_size).float()
    teacher_probs = F.softmax(teacher, dim=-1)
    student_log_probs = F.log_softmax(student, dim=-1)
    per_token = -torch.sum(teacher_probs * student_log_probs, dim=-1)
    return _masked_mean(per_token, token_mask)


def _causal_output_slices(student_logits, teacher_logits, attention_mask=None):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Student/teacher output shapes differ: {student_logits.shape} vs {teacher_logits.shape}."
        )
    if student_logits.ndim < 3:
        raise ValueError("Causal output logits must have batch, sequence, and vocabulary dimensions.")
    if student_logits.shape[-2] < 2:
        raise ValueError("Causal output distillation requires at least two sequence positions.")

    prediction_mask = None
    if attention_mask is not None:
        if tuple(attention_mask.shape) != tuple(student_logits.shape[:-1]):
            raise ValueError(
                f"Attention mask shape {attention_mask.shape} does not match output tokens "
                f"{student_logits.shape[:-1]}."
            )
        mask = attention_mask.to(dtype=torch.bool)
        prediction_mask = mask[..., :-1] & mask[..., 1:]

    return (
        student_logits[..., :-1, :],
        teacher_logits[..., :-1, :],
        prediction_mask,
    )


def compute_causal_output_distill_ce(student_logits, teacher_logits, attention_mask=None):
    student, teacher, prediction_mask = _causal_output_slices(
        student_logits, teacher_logits, attention_mask
    )
    return compute_output_distill_ce(student, teacher, token_mask=prediction_mask)
