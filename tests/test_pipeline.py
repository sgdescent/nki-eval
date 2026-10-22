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


def test_later_benchmark_failure_erases_earlier_performance(monkeypatch, tmp_path):
    calls = {}

    def candidate(x):
        size = x.numel()
        calls[size] = calls.get(size, 0) + 1
        return -x if size == 2 or calls[size] <= 2 else x

    monkeypatch.setattr(worker, "load", lambda name: torch.neg if name == "ref" else candidate)
    result = worker.run(request(), tmp_path)
    assert result["status"] == "incorrect"
    assert result["correct"] is False
    assert calls[2] == 4  # Earlier case finished both correctness and timing.
    assert result["cases"][0]["passed"]
    assert result["error"]["case"] == "bad"
    assert result["performance"] is None
    assert all(case["performance"] is None for case in result["cases"])


def test_benchmark_baseline_failure_invalidates_speedup(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "load", lambda _: torch.neg)

    class BadTimedBaseline:
        timing_scope = "test"

        def prepare(self, candidate, reference, inputs, workdir, compiler_args):
            calls = 0

            def baseline(x):
                nonlocal calls
                calls += 1
                return -x if calls <= 2 else x

            return candidate, baseline, {}

    monkeypatch.setitem(worker.BACKENDS, "cpu", BadTimedBaseline)
    result = worker.run(request(), tmp_path)
    assert result["status"] == "incorrect"
    assert result["error"]["actor"] == "baseline"
    assert result["error"]["phase"] == "timed"
    assert result["performance"] is None


def test_baseline_mutation_invalidates_correctness(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "load", lambda _: torch.neg)

    class MutatingBaseline:
        timing_scope = "test"

        def prepare(self, candidate, reference, inputs, workdir, compiler_args):
            return candidate, lambda x: x.neg_(), {}

    monkeypatch.setitem(worker.BACKENDS, "cpu", MutatingBaseline)
    result = worker.run(request(), tmp_path)
    assert result["status"] == "incorrect"
    assert "mutated" in str(result["cases"][0]["trials"][0]["baseline"])
    assert result["performance"] is None


def test_benchmark_seed_ranges_follow_all_correctness_trials(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "load", lambda _: torch.neg)
    config = request()
    config["warmup"] = 1
    config["seed"] = 101
    result = worker.run(config, tmp_path)
    assert result["status"] == "passed"
    first, second = result["cases"]
    assert [trial["seed"] for trial in first["trials"]] == [101, 102]
    assert [trial["seed"] for trial in second["trials"]] == [103, 104]
    assert first["performance"]["validation"]["seed_start"] == 105
    assert first["performance"]["validation"]["seed_end"] == 107
    assert second["performance"]["validation"]["seed_start"] == 108
    assert second["performance"]["validation"]["seed_end"] == 110


def test_benchmark_seed_overflow_is_configuration_error(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "load", lambda _: torch.neg)
    config = request()
    config["seed"] = 2**63 - 4  # Correctness alone fits; timing would overflow.
    result = worker.run(config, tmp_path)
    assert result["status"] == "error"
    assert result["error"]["category"] == "configuration_error"
    assert result["performance"] is None
