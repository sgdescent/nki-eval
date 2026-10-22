"""Failure-injection regressions for the harness, not a hostile-code security audit.

CPU submissions exercise the real subprocess and shared validation/timing pipeline.
These tests do not establish native NKI enforcement or process sandboxing.
"""

from pathlib import Path

import pytest

from nki_eval import evaluate

ROOT = Path(__file__).resolve().parents[1]
SPEC = {"cases": [{"name": "random", "inputs": [{"shape": [4, 7]}, {"shape": [4, 7]}]}]}


def evaluate_code(tmp_path, body, **kwargs):
    candidate = tmp_path / "candidate.py"
    candidate.write_text(body)
    return evaluate(
        f"{ROOT / 'examples/add/reference.py'}:reference",
        f"{candidate}:kernel",
        SPEC,
        backend="cpu",
        trials=2,
        warmup=2,
        iterations=4,
        output_dir=tmp_path / "runs",
        **kwargs,
    )


def phase_switch(action, after=4):
    return (
        "calls = 0\n"
        "def kernel(a, b):\n"
        " global calls\n"
        " calls += 1\n"
        f" if calls > {after}:\n"
        + "".join(f"  {line}\n" for line in action.splitlines())
        + " return a + b\n"
    )


@pytest.mark.parametrize(
    ("action", "after", "phase", "problem"),
    [
        ("return a - b", 2, "warmup", None),
        ("return a - b", 4, "timed", None),
        ("return a - b", 7, "timed", None),
        ("return (a + b).double()", 4, "timed", "torch.float64"),
        ("a.add_(b)\nreturn a", 4, "timed", "mutated"),
        ("return (a + b)[:, :-1]", 4, "timed", "expected"),
        ("return (a + b) * float('nan')", 4, "timed", "nan"),
    ],
)
def test_phase_switch_rejected(tmp_path, action, after, phase, problem):
    result = evaluate_code(tmp_path, phase_switch(action, after))
    assert result["status"] == "incorrect"
    assert result["correct"] is False
    assert result["performance"] is None
    row = result["cases"][0]
    assert all(trial["candidate"]["passed"] for trial in row["trials"])
    assert row["performance"] is None
    assert row["passed"] is False
    error = result["error"]
    assert error["category"] == "correctness_error"
    assert error["stage"] == "benchmark_validation"
    assert error["actor"] == "candidate"
    assert error["phase"] == phase
    assert error["iteration"] == after - (2 if phase == "warmup" else 4)
    assert error["seed"] == after
    if problem:
        assert problem in str(error["check"])


def test_stale_answer_cache_rejected(tmp_path):
    result = evaluate_code(
        tmp_path,
        "calls = 0\n"
        "cached = None\n"
        "def kernel(a, b):\n"
        " global calls, cached\n"
        " calls += 1\n"
        " if calls <= 4: cached = a + b\n"
        " return cached\n",
    )
    assert result["status"] == "incorrect"
    assert result["error"]["phase"] == "timed"
    assert result["error"]["seed"] == 4
    assert result["performance"] is None


def test_valid_kernel_receives_fresh_inputs_and_passes(tmp_path):
    result = evaluate_code(
        tmp_path,
        "seen = set()\n"
        "def kernel(a, b):\n"
        " fingerprint = tuple(a.flatten().tolist())\n"
        " if fingerprint in seen: raise RuntimeError('Repeated input')\n"
        " seen.add(fingerprint)\n"
        " return a + b\n",
    )
    assert result["status"] == "passed"
    performance = result["cases"][0]["performance"]
    assert result["performance"]["benchmark_protocol"] == "fresh_inputs_checked_v1"
    assert performance["candidate"]["count"] == 4
    assert performance["reference"]["count"] == 4
    assert performance["validation"]["checked_warmup_pairs"] == 2
    assert performance["validation"]["checked_timed_pairs"] == 4
    assert performance["validation"]["seed_start"] == 2
    assert performance["validation"]["seed_end"] == 7


def test_benchmark_exception_retains_context_without_reward(tmp_path):
    result = evaluate_code(tmp_path, phase_switch("raise RuntimeError('phase failure')"))
    assert result["status"] == "error"
    assert result["performance"] is None
    assert result["cases"][0]["performance"] is None
    assert result["error"]["actor"] == "candidate"
    assert result["error"]["phase"] == "timed"
    assert result["error"]["seed"] == 4
    assert result["error"]["exception_type"] == "RuntimeError"
    assert "phase failure" in result["error"]["message"]


def test_timeout_after_correctness_withholds_reward(tmp_path):
    result = evaluate_code(
        tmp_path,
        "import time\n" + phase_switch("time.sleep(60)"),
        timeout=8,
    )
    assert result["status"] == "timeout"
    assert result["performance"] is None
    assert result["error"]["stage"] == "benchmark"
