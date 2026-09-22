import torch

from nki_eval import worker


def request():
    return {
        "backend": "cpu",
        "threads": 1,
        "seed": 0,
        "reference": "ref",
        "candidate": "candidate",
        "spec": {
            "cases": [
                {"name": "good", "inputs": [{"shape": [2]}]},
                {"name": "bad", "inputs": [{"shape": [3]}]},
            ]
        },
        "trials": 2,
        "compiler_args": [],
        "correctness_only": False,
        "warmup": 0,
        "iterations": 2,
    }


def test_later_failure_gates_all_timing(monkeypatch, tmp_path):
    def reference(x):
        return x + 1

    def candidate(x):
        return x + (1 if x.numel() == 2 else 0)

    monkeypatch.setattr(worker, "load", lambda name: reference if name == "ref" else candidate)

    def forbidden(*args):
        raise AssertionError("Benchmark must not run after any failure")

    monkeypatch.setattr(worker, "benchmark", forbidden)
    result = worker.run(request(), tmp_path)
    assert result["status"] == "incorrect"
    assert result["cases"][0]["passed"]
    assert not result["cases"][1]["passed"]
    assert all(case["performance"] is None for case in result["cases"])


def test_bad_compiled_baseline_invalidates_speedup(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "load", lambda _: torch.neg)

    class BadBaseline:
        timing_scope = "test"

        def prepare(self, candidate, reference, inputs, workdir, compiler_args):
            return candidate, torch.abs, {}

    monkeypatch.setitem(worker.BACKENDS, "cpu", BadBaseline)
    result = worker.run(request(), tmp_path)
    assert result["status"] == "incorrect"
    assert result["performance"] is None


def test_compile_failure_never_marks_case_passed(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "load", lambda _: torch.neg)

    class Broken:
        def prepare(self, *args):
            raise RuntimeError("compile failed")

    monkeypatch.setitem(worker.BACKENDS, "cpu", Broken)
    result = worker.run(request(), tmp_path)
    assert result["status"] == "error"
    assert not result["cases"][0]["passed"]


def test_benchmark_failure_preserves_correctness_but_not_success(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "load", lambda _: torch.neg)

    def broken(*args):
        raise RuntimeError("benchmark failed")

    monkeypatch.setattr(worker, "benchmark", broken)
    result = worker.run(request(), tmp_path)
    assert result["correct"]
    assert result["status"] == "error"
    assert result["performance"] is None
