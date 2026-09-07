import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import pytest

from gemq.utils.expert_bit_config import has_zero_bit_experts, load_expert_bit_config
from gemq.utils.qwen3_distill_path import resolve_distill_path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def allocation(tmp_path, zero):
    path = tmp_path / "GEMQ" / "actual_allocation.pkl"
    path.parent.mkdir(exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({0: {0: 0 if zero else 2, 1: 2, 2: 3}}, handle)
    return path


def test_bit_config_helper_preserves_original_id_mapping(tmp_path):
    path = allocation(tmp_path, True)
    assert load_expert_bit_config(path) == {0: {0: 0, 1: 2, 2: 3}}
    assert has_zero_bit_experts(load_expert_bit_config(path))
    assert not has_zero_bit_experts(None)


def test_serving_path_distinguishes_zero_weight_diagnostics(tmp_path):
    path = allocation(tmp_path, True)
    env = {"BIT_CFG_PATH": str(path), "WBITS": "1,2,3"}
    result = resolve_distill_path(MODEL, "C4-Seed0", "2.0", "4", "4", "0.0", env)
    assert result.endswith("actual_allocation_A4-G16-D4-E2.0_RFT-distill_ce-ReconKL-w0p0-const")
    positive = resolve_distill_path(MODEL, "C4-Seed0", "2.0", "4", "4", "1.0", env)
    assert positive.endswith("-ReconKL-w1p0-const")
    allocation(tmp_path, False)
    env["WBITS"] = "0,2,3"
    no_pruning = resolve_distill_path(MODEL, "C4-Seed0", "2.0", "4", "4", "0.0", env)
    assert no_pruning.endswith("_RFT-distill_ce")
    assert "ReconKL" not in no_pruning


def test_default_allocation_path_matches_quantize_naming(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = Path("configs") / MODEL / "GEMQ"
    directory.mkdir(parents=True)
    name = "C4-Seed0_Metric-expert_cost-Ctxuniform2_E2.0_B0,2,3_Pmax0.25_EqPrune"
    with (directory / (name + ".pkl")).open("wb") as handle:
        pickle.dump({0: {0: 0, 1: 2}}, handle)
    result = resolve_distill_path(MODEL, "C4-Seed0", "2.0", "4", "4", "0", {})
    assert f"/{name}_A4-G16-D4-E2.0_RFT-distill_ce-ReconKL-w0p0-const" in result


def dry_run(tmp_path, path, weight, *, bits="1,2,3"):
    bash = os.environ.get("GEMQ_TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("needs bash for script integration")
    env = dict(os.environ)
    env.update({
        "TRANSFER_TEST_PYTHON": Path(sys.executable).as_posix(),
        "PATH": str(Path(bash).resolve().parent) + os.pathsep + env.get("PATH", ""),
        "CALIB_DATASET": "c4", "BPE": "2.0", "WBITS": bits,
        "BIT_CFG_PATH": path.as_posix(), "MIXED_PREC": "true",
        "FINETUNE_ROUTERS": "true", "RFT_TRAINER": "distill_ce",
        "RFT_TRANSFER_WEIGHT": weight, "EXTRA_CONSTR": "none",
        "ALLOCATION_METRIC": "expert_cost", "ALLOCATION_TAG": "C4-Seed0",
        "SAVE_GPTQ_CHECKPOINT": "true", "LOAD_GPTQ_CHECKPOINT": "false",
        "GPTQ_CHECKPOINT_ROOT": (tmp_path / "checkpoints").as_posix(),
    })
    # Only intercept the real model launch; allocation inspection/validation runs.
    command = '''
python() {
    if [[ "${1:-}" == "-m" && "${2:-}" == "gemq.quantize" ]]; then
        printf 'DRY_RUN_ARG=%s\n' "$@"
    else
        "${TRANSFER_TEST_PYTHON}" "$@"
    fi
}
export -f python
"$BASH" scripts/Qwen3-30B-A3B-Instruct-2507/quantize.sh
'''
    return subprocess.run(
        [bash, "-c", command], cwd=ROOT, env=env, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=30,
    )


def test_shell_zero_weight_uses_same_masked_checkpoint_as_positive(tmp_path):
    path = allocation(tmp_path, True)
    zero, positive = dry_run(tmp_path, path, "0.0"), dry_run(tmp_path, path, "1.0")
    for result in (zero, positive):
        assert result.returncode == 0, result.stdout + result.stderr
        assert "_PruneMask" in result.stdout
        assert "DRY_RUN_ARG=--rft_transfer_weight" in result.stdout
    checkpoint_paths = [next(line for line in result.stdout.splitlines() if "GPTQ ckpt path:" in line)
                        for result in (zero, positive)]
    assert checkpoint_paths[0] == checkpoint_paths[1]
    assert "-ReconKL-w0p0-const" in zero.stdout
    assert "diagnostics only" in zero.stdout
    save_path = next(line.split("Save path:", 1)[1].strip() for line in zero.stdout.splitlines() if "Save path:" in line)
    inferred = resolve_distill_path(MODEL, "C4-Seed0", "2.0", "4", "4", "0.0", {"BIT_CFG_PATH": str(path)})
    assert save_path == inferred


def test_shell_candidate_zero_is_not_enough_to_enable_diagnostics(tmp_path):
    result = dry_run(tmp_path, allocation(tmp_path, False), "0.0", bits="0,2,3")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "_PruneMask" not in result.stdout
    assert "ReconKL" not in result.stdout


@pytest.mark.parametrize("weight", ["nan", "inf", "-1", "bad"])
def test_shell_invalid_weight_fails_before_model_launch(tmp_path, weight):
    result = dry_run(tmp_path, allocation(tmp_path, True), weight)
    assert result.returncode != 0
    assert "DRY_RUN_ARG" not in result.stdout
