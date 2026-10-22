import copy
import json
from pathlib import Path

import pytest

from nki_eval.suite import load_manifest, run_suite, summarize_runs


def valid_run(name="add", repeat=1):
    return {
        "workload": name,
        "repeat": repeat,
        "status": "passed",
        "correct": True,
        "cases": [
            {
                "name": "small",
                "passed": True,
                "performance": {
                    "timing_scope": "cpu_debug_host_call",
                    "candidate": {"median": 2},
                    "reference": {"median": 3},
                    "speedup": 999,
                },
            }
        ],
    }


def test_summary_uses_raw_medians_and_repeat_spread():
    a, b = valid_run(), valid_run(repeat=2)
    b["cases"][0]["performance"]["candidate"]["median"] = 1
    summary = summarize_runs([a, b], 2)
    row = summary["performance"][0]
    assert row["repeats"] == 2
    assert row["median_speedup"] == 2.25
    assert (row["min_speedup"], row["max_speedup"]) == (1.5, 3)
    assert summary["geometric_mean_speedup"] == pytest.approx(2.25)


@pytest.mark.parametrize(
    "failure",
    ["incorrect", "error", "timeout", "incomplete", "mixed_scope", "missing_case", "no_timing"],
)
def test_partial_or_incompatible_runs_never_get_aggregate_performance(failure):
    a, b = valid_run(), valid_run(repeat=2)
    if failure in {"incorrect", "error", "timeout"}:
        b["status"] = failure
    elif failure == "mixed_scope":
        b["cases"][0]["performance"]["timing_scope"] = "neuron_host_request_including_transfers"
    elif failure == "missing_case":
        b["cases"] = []
    elif failure == "no_timing":
        b["cases"][0]["performance"] = None
    summary = summarize_runs([a, b], 3 if failure == "incomplete" else 2)
    assert summary["performance"] is None
    assert summary["geometric_mean_speedup"] is None


def manifest(tmp_path):
    (tmp_path / "reference.py").write_text("def reference(a): return a + 1\n")
    (tmp_path / "candidate.py").write_text("def kernel(a): return a + 1\n")
    (tmp_path / "spec.json").write_text(
        json.dumps({"cases": [{"name": "vector", "inputs": [{"shape": [7]}]}]})
    )
    path = tmp_path / "manifest.json"
    item = {
        "name": "add_one",
        "reference": "reference.py:reference",
        "candidate": "candidate.py:kernel",
        "cpu_candidate": "candidate.py:kernel",
        "spec": "spec.json",
    }
    path.write_text(json.dumps({"schema_version": 1, "workloads": [item]}))
    return path


def test_repeated_real_workers_save_reproducible_suite(tmp_path):
    path = manifest(tmp_path)
    output = tmp_path / "output"
    result = run_suite(
        path,
        backend="cpu",
        output_dir=output,
        repeats=2,
        trials=2,
        warmup=1,
        iterations=2,
        timeout=30,
    )
    assert result["summary"]["all_passed"]
    assert result["summary"]["performance"][0]["repeats"] == 2
    reports = [json.loads(Path(run["report"]).read_text()) for run in result["runs"]]
    assert reports[0]["config"]["seed"] != reports[1]["config"]["seed"]
    assert all(report["backend"] == "cpu" for report in reports)
    assert json.loads((output / "suite.json").read_text()) == result
    assert "cpu_debug_host_call" in (output / "suite.md").read_text()
    with pytest.raises(ValueError, match="new output directory"):
        run_suite(path, backend="cpu", output_dir=output)


def test_manifest_rejects_duplicates_and_unknown_selection(tmp_path):
    path = manifest(tmp_path)
    with pytest.raises(ValueError, match="Unknown workloads"):
        load_manifest(path, ["not_a_workload"])
    contents = json.loads(path.read_text())
    contents["workloads"].append(copy.deepcopy(contents["workloads"][0]))
    path.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="unique"):
        load_manifest(path)


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"schema_version": 1, "workloads": [3]},
        {"schema_version": 1, "workloads": [{"name": "bad", "reference": None}]},
    ],
)
def test_malformed_manifest_is_a_configuration_error(tmp_path, value):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_duplicate_repeats_and_source_drift_cannot_be_aggregated():
    a, b = valid_run(), valid_run()
    assert summarize_runs([a, b], 2)["performance"] is None
    b["repeat"] = 2
    a["identity"] = {"candidate_sha256": "one"}
    b["identity"] = {"candidate_sha256": "two"}
    assert summarize_runs([a, b], 2)["performance"] is None


def test_infrastructure_exception_is_saved_as_failure(monkeypatch, tmp_path):
    from nki_eval import suite

    def broken(*args, **kwargs):
        raise OSError("simulated inaccessible artifact directory")

    monkeypatch.setattr(suite, "evaluate", broken)
    path = manifest(tmp_path)
    result = run_suite(path, backend="cpu", output_dir=tmp_path / "output", repeats=1)
    assert not result["summary"]["all_passed"]
    assert result["summary"]["performance"] is None
    assert result["runs"][0]["error"]["category"] == "suite_infrastructure_error"
    assert Path(result["runs"][0]["report"]).is_file()
