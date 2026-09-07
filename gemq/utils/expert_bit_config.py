"""Dependency-light readers for trusted, locally generated expert allocations."""

import pickle
from pathlib import Path


def load_expert_bit_config(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        config = pickle.load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Bit allocation must be a dict, got {type(config)!r}")
    normalized = {}
    for layer_idx, experts in config.items():
        if not isinstance(experts, dict):
            raise TypeError(f"Allocation for layer {layer_idx!r} must be a dict")
        normalized[int(layer_idx)] = {
            int(expert_idx): int(bit) for expert_idx, bit in experts.items()
        }
    return normalized


def has_zero_bit_experts(bit_config):
    return bool(bit_config) and any(
        bit == 0 for experts in bit_config.values() for bit in experts.values()
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect an actual expert allocation for zero-bit experts.")
    parser.add_argument("path")
    args = parser.parse_args()
    print("true" if has_zero_bit_experts(load_expert_bit_config(args.path)) else "false")
