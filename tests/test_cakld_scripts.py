"""Dry-run the shell entry points; never load or quantize a real model."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/Qwen3-30B-A3B-Instruct-2507/quantize.sh",
    "scripts/OLMoE-1B-7B-0125-Instruct/quantize.sh",
]


def dry_run(script, tmp_path, **overrides):
    bash = os.environ.get("GEMQ_TEST_BASH") or shutil.which("bash")
    if not bash:
        pytest.skip("needs bash for shell entry-point tests")
    env = dict(os.environ)
    env.update({
        "CAKLD_TEST_PYTHON": Path(sys.executable).as_posix(),
        "CALIB_DATASET": "c4", "MIXED_PREC": "false", "FINETUNE_ROUTERS": "true",
        "RFT_TRAINER": "cakld", "RFT_CAKLD_GAMMA": "auto",
        "RFT_REBUILD_TEACHER_CACHE": "false",
        "SAVE_GPTQ_CHECKPOINT": "true", "LOAD_GPTQ_CHECKPOINT": "false",
        "GPTQ_CHECKPOINT_ROOT": tmp_path.as_posix(),
        "ALLOCATION_METRIC": "expert_cost", "EXTRA_CONSTR": "none",
    })
    env.update(overrides)
    # Run real argument validation, intercept only the final model launch.
    command = '''
python() {
    if [[ "${1:-}" == "-m" ]]; then
        printf 'DRY_RUN_ARG=%s\n' "$@"
    else
        "${CAKLD_TEST_PYTHON}" "$@"
    fi
}
export -f python
bash "$1"
'''
    return subprocess.run(
        [bash, "-c", command, "cakld-test", script], cwd=ROOT, env=env,
        text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=30,
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_cakld_script_names_do_not_change_gptq_path(script, tmp_path):
    cakld = dry_run(script, tmp_path)
    manual = dry_run(script, tmp_path, RFT_CAKLD_GAMMA=".50")
    ce = dry_run(script, tmp_path, RFT_TRAINER="distill_ce")
    for result in (cakld, manual, ce):
        assert result.returncode == 0, result.stdout + result.stderr
    assert "_RFT-cakld-maxprob-gauto" in cakld.stdout
    assert "_RFT-cakld-maxprob-g0p5" in manual.stdout
    assert "_RFT-distill_ce" in ce.stdout
    assert "DRY_RUN_ARG=--rft_cakld_gamma" in cakld.stdout
    assert "DRY_RUN_ARG=--rft_cakld_gamma" not in ce.stdout
    paths = [next(line for line in run.stdout.splitlines() if "GPTQ ckpt path:" in line)
             for run in (cakld, manual, ce)]
    assert paths[0] == paths[1] == paths[2]


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("gamma", ["nan", "inf", "-0.1", "1.1", "bad"])
def test_cakld_script_rejects_bad_gamma_before_model_launch(script, gamma, tmp_path):
    result = dry_run(script, tmp_path, RFT_CAKLD_GAMMA=gamma)
    assert result.returncode != 0
    assert "RFT_CAKLD_GAMMA must be" in result.stderr
    assert "DRY_RUN_ARG" not in result.stdout
