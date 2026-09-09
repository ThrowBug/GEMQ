"""Mixed-bit policy for one pruned or unpruned Qwen3-MoE layer."""

from __future__ import annotations

from dataclasses import dataclass

ATTENTION_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ROUTER_NAME = "mlp.gate"


@dataclass(frozen=True)
class Qwen3MoeAWQPolicy:
    """Resolved bit-widths for every weight targeted by AWQ in one layer."""

    expert_bits: tuple[int, ...]
    attention_bits: int = 4
    dense_bits: int = 4
    router_bits: int = 16

    def __post_init__(self):
        if not self.expert_bits:
            raise ValueError("expert_bits must not be empty")
        values = (*self.expert_bits, self.attention_bits, self.dense_bits)
        if any(not isinstance(bit, int) or not 1 <= bit <= 16 for bit in values):
            raise ValueError(
                f"AWQ bit-widths must be integers in [1, 16], got {values}"
            )
        if self.router_bits != 16:
            raise ValueError("GEMQ-AWQ keeps the Qwen3-MoE router at 16 bits")

    @classmethod
    def uniform(
        cls,
        num_experts,
        expert_bits=2,
        attention_bits=4,
        dense_bits=4,
    ):
        return cls(
            expert_bits=tuple([int(expert_bits)] * int(num_experts)),
            attention_bits=int(attention_bits),
            dense_bits=int(dense_bits),
        )

    def bit_for(self, module_name):
        if module_name in ATTENTION_PROJECTIONS:
            return self.attention_bits
        prefix = "mlp.experts."
        if module_name.startswith(prefix):
            pieces = module_name.split(".")
            if len(pieces) == 4 and pieces[-1] in EXPERT_PROJECTIONS:
                expert_idx = int(pieces[2])
                try:
                    return self.expert_bits[expert_idx]
                except IndexError as error:
                    raise ValueError(
                        f"Expert id {expert_idx} is absent from a policy with "
                        f"{len(self.expert_bits)} experts"
                    ) from error
        if module_name == ROUTER_NAME:
            return None
        return self.dense_bits if "mlp" in module_name else None

    def to_dict(self):
        return {
            "expert_bits": list(self.expert_bits),
            "attention_bits": self.attention_bits,
            "dense_bits": self.dense_bits,
            "router_bits": self.router_bits,
        }
