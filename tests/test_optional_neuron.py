"""Explicitly opt in; never schedule paid hardware or silently fake an SDK test."""

import os
from pathlib import Path

import pytest

from nki_eval import evaluate

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("example", ["add", "fused_relu"])
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
    )
    assert result["status"] == "passed", result
    assert (result["performance"] is None) == (backend == "simulator")
