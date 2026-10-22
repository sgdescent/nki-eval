"""Explicitly opt in; never schedule paid hardware or silently fake an SDK test."""

import os
from pathlib import Path

import pytest

from nki_eval import evaluate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "example", ["add", "fused_relu", "row_sum", "softmax", "layernorm", "matmul"]
)
@pytest.mark.parametrize("backend", ["simulator", "neuron"])
def test_real_sdk(example, backend, tmp_path):
    if os.environ.get(f"NKI_EVAL_RUN_{backend.upper()}") != "1":
        pytest.skip(f"Set NKI_EVAL_RUN_{backend.upper()}=1 to run real SDK integration")
    folder = ROOT / "examples" / example
    result = evaluate(
        str(folder / "reference.py") + ":reference",
        str(folder / "kernel.py") + ":kernel",
        folder / "workload.json",
        backend=backend,
        output_dir=tmp_path,
        iterations=10,
        timeout=1200,
        compiler_args=["--auto-cast=none"],
    )
    assert result["status"] == "passed", result
    assert (result["performance"] is None) == (backend == "simulator")


@pytest.mark.parametrize("candidate", ["nki_wrong_add", "nki_wrong_tail"])
@pytest.mark.parametrize("backend", ["simulator", "neuron"])
def test_real_sdk_rejects_wrong_native_kernel(candidate, backend, tmp_path):
    if os.environ.get(f"NKI_EVAL_RUN_{backend.upper()}") != "1":
        pytest.skip(f"Set NKI_EVAL_RUN_{backend.upper()}=1 to run real SDK integration")
    result = evaluate(
        str(ROOT / "examples/add/reference.py") + ":reference",
        str(ROOT / "tests/fixtures" / f"{candidate}.py") + ":kernel",
        {"cases": [{"name": "tails", "inputs": [{"shape": [129, 513]}, {"shape": [129, 513]}]}]},
        backend=backend,
        output_dir=tmp_path,
        trials=5,
        timeout=1200,
        compiler_args=["--auto-cast=none"],
    )
    assert result["status"] == "incorrect", result
    assert result["error"]["stage"] == "comparison", result
    assert result["performance"] is None
    assert not result["cases"][0]["passed"]
    if backend == "neuron":
        assert all(trial["baseline"]["passed"] for trial in result["cases"][0]["trials"])
