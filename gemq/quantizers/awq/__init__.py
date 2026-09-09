"""Activation-aware fake quantization for Qwen3-MoE.

The scale and clipping search is adapted from MIT-HAN-Lab's llm-awq project
(MIT license).  GEMQ keeps the implementation local so the AWQ workflow does
not depend on a sibling checkout or on llm-awq being installed.
"""

from .policy import Qwen3MoeAWQPolicy
from .qwen3 import AWQSearchOptions
from .quantizer import fake_quantize_weight
from .search import (
    apply_clip,
    apply_linear_pair_scale,
    apply_norm_input_scale,
    search_clip,
    search_scale,
)

__all__ = [
    "Qwen3MoeAWQPolicy",
    "AWQSearchOptions",
    "apply_clip",
    "apply_linear_pair_scale",
    "apply_norm_input_scale",
    "fake_quantize_weight",
    "search_clip",
    "search_scale",
]
