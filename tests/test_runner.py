import json
import sys
from pathlib import Path

import pytest

from nki_eval import evaluate

ROOT = Path(__file__).resolve().parents[1]
SPEC = {"cases": [{"name": "small", "inputs": [{"shape": [2, 3]}, {"shape": [2, 3]}]}]}


def run(tmp_path, code, **kwargs):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(code)
    return evaluate(
        str(ROOT / "examples/add/reference.py") + ":reference",
        str(candidate) + ":kernel",
        SPEC,
        backend="cpu",
        output_dir=tmp_path / "runs",
        trials=2,
        warmup=1,
        iterations=3,
        **kwargs,
    )


def test_success_artifacts_and_performance(tmp_path):
    result = run(tmp_path, "def kernel(a, b): return a + b")
    assert result["status"] == "passed"
    assert result["correct"]
    assert result["performance"]["timing_scope"] == "cpu_debug_host_call"
    assert result["cases"][0]["performance"]["candidate"]["count"] == 3
    report = json.loads(Path(result["artifacts"]["report"]).read_text())
    assert report == result
    assert len(result["config"]["source"]["candidate"]["sha256"]) == 64


def test_wrong_answer_never_gets_performance(tmp_path):
    result = run(tmp_path, "def kernel(a, b): return a - b")
    assert result["status"] == "incorrect"
    assert result["performance"] is None
    assert all(x["performance"] is None for x in result["cases"])


def test_input_mutation_is_a_failure(tmp_path):
    result = run(tmp_path, "def kernel(a, b):\n a.add_(b)\n return a")
    assert result["status"] == "incorrect"
    assert "mutated" in str(result["cases"][0]["trials"])


@pytest.mark.parametrize(
    ("code", "category"),
    [
        ("def kernel(:", "compile_error"),
        ("def kernel(a, b): raise RuntimeError('broken')", "runtime_error"),
        ("import os\ndef kernel(a, b): os._exit(9)", "worker_crash"),
    ],
)
def test_failures_are_structured(tmp_path, code, category):
    result = run(tmp_path, code)
    assert result["status"] == "error"
    assert result["error"]["category"] == category
    assert result["performance"] is None


def test_correctness_only(tmp_path):
    result = run(tmp_path, "def kernel(a, b): return a + b", correctness_only=True)
    assert result["status"] == "passed"
    assert result["performance"] is None


def test_worker_timeout(tmp_path):
    result = run(tmp_path, "import time\ndef kernel(a, b): time.sleep(60)", timeout=5)
    assert result["status"] == "timeout"
    assert result["error"]["category"] == "timeout"


def test_hardware_rejects_pytorch_fallback_before_import(tmp_path):
    candidate = tmp_path / "fallback.py"
    candidate.write_text("import torch\ndef kernel(a, b): return torch.add(a, b)")
    result = evaluate(
        str(ROOT / "examples/add/reference.py") + ":reference",
        str(candidate) + ":kernel",
        SPEC,
        output_dir=tmp_path,
    )
    assert result["error"]["category"] == "policy_error"


def test_cli_exit_codes(tmp_path):
    import subprocess

    candidate = tmp_path / "bad.py"
    candidate.write_text("def kernel(a, b): return a - b")
    report = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nki_eval.cli",
            "--backend",
            "cpu",
            "--reference",
            str(ROOT / "examples/add/reference.py") + ":reference",
            "--candidate",
            str(candidate) + ":kernel",
            "--spec",
            str(ROOT / "examples/add/workload.json"),
            "--output-dir",
            str(tmp_path / "runs"),
            "--json",
            str(report),
        ],
        capture_output=True,
    )
    assert completed.returncode == 1, completed.stderr.decode()
    assert json.loads(report.read_text())["status"] == "incorrect"
