"""Serial, repeatable workload suites; never provision devices or hide failed runs."""

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from .loading import source_info
from .runner import evaluate


def load_manifest(path, selected=None):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or not isinstance(manifest.get("workloads"), list)
    ):
        raise ValueError("Expected schema_version=1 and a workloads list")
    names = set()
    workloads = []
    for item in manifest["workloads"]:
        if not isinstance(item, dict):
            raise ValueError("Each workload must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Workload names must be unique nonempty strings")
        names.add(name)
        row = {"name": name}
        for key in ("reference", "candidate", "cpu_candidate"):
            if key == "cpu_candidate" and key not in item:
                continue
            entry = item.get(key, "")
            if not isinstance(entry, str):
                raise ValueError(f"{name}: {key} must be a string")
            source, separator, symbol = entry.rpartition(":")
            if not separator or not symbol or not source:
                raise ValueError(f"{name}: invalid {key} entry")
            source = (path.parent / source).resolve()
            if not source.is_file():
                raise ValueError(f"{name}: missing {key} file {source}")
            row[key] = f"{source}:{symbol}"
        if not isinstance(item.get("spec"), str) or not item["spec"]:
            raise ValueError(f"{name}: spec must be a nonempty path string")
        spec = (path.parent / item["spec"]).resolve()
        if not spec.is_file():
            raise ValueError(f"{name}: missing workload specification")
        row["spec"] = str(spec)
        if selected is None or name in selected:
            workloads.append(row)
    if selected and set(selected) - names:
        raise ValueError(f"Unknown workloads: {sorted(set(selected) - names)}")
    if not workloads:
        raise ValueError("No workloads selected")
    return workloads


def summarize_runs(runs, expected_runs):
    """Withhold cross-suite performance unless every planned run passes in one scope."""
    complete = (
        len(runs) == expected_runs
        and len({(run["workload"], run["repeat"]) for run in runs}) == expected_runs
    )
    all_passed = complete and all(run["status"] == "passed" and run["correct"] for run in runs)
    groups = {}
    valid_performance = all_passed
    scopes = set()
    workloads = {}
    identities = {}
    protocols = set()
    for run in runs:
        names = set()
        identity = run.get("identity")
        if run["workload"] in identities and identities[run["workload"]] != identity:
            valid_performance = False
        identities[run["workload"]] = identity
        if not run["cases"]:
            valid_performance = False
        for case in run["cases"]:
            names.add(case["name"])
            perf = case.get("performance")
            if not case["passed"] or not perf:
                valid_performance = False
                continue
            scopes.add(perf["timing_scope"])
            protocols.add(perf.get("validation", {}).get("protocol", "legacy_unchecked"))
            numbers = (perf["candidate"]["median"], perf["reference"]["median"])
            if any(not math.isfinite(x) or x <= 0 for x in numbers):
                valid_performance = False
                continue
            groups.setdefault((run["workload"], case["name"]), []).append(numbers)
        prior = workloads.setdefault(run["workload"], names)
        if prior != names:
            valid_performance = False
    rows = []
    if valid_performance and len(scopes) == 1 and len(protocols) == 1:
        for (workload, case), pairs in sorted(groups.items()):
            ratios = [reference / candidate for candidate, reference in pairs]
            rows.append(
                {
                    "workload": workload,
                    "case": case,
                    "repeats": len(pairs),
                    "candidate_median_ms": statistics.median(pair[0] for pair in pairs),
                    "reference_median_ms": statistics.median(pair[1] for pair in pairs),
                    "median_speedup": statistics.median(ratios),
                    "min_speedup": min(ratios),
                    "max_speedup": max(ratios),
                }
            )
    return {
        "complete": complete,
        "all_passed": all_passed,
        "runs_completed": len(runs),
        "runs_planned": expected_runs,
        "runs_passed": sum(run["status"] == "passed" for run in runs),
        "timing_scope": next(iter(scopes)) if rows else None,
        "benchmark_protocol": next(iter(protocols)) if rows else None,
        "performance": rows or None,
        "geometric_mean_speedup": (
            math.exp(sum(math.log(row["median_speedup"]) for row in rows) / len(rows))
            if rows
            else None
        ),
        "note": "Each speedup is the median of paired run ratios, not necessarily the ratio "
        "of the displayed medians. The geometric mean weights cases equally. "
        "Repeat ranges are descriptive, not confidence intervals. Failed, incomplete, "
        "changed-source, mixed-protocol and correctness-only suites have no aggregate "
        "performance. Source hashes cover entry files, not imported dependencies.",
    }


def write_summary(folder, report):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "suite.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    summary = report["summary"]
    lines = [
        "# Evaluation suite results",
        "",
        f"Backend: `{report['config']['backend']}`. "
        f"Completed {summary['runs_completed']}/{summary['runs_planned']} runs; "
        f"{summary['runs_passed']} passed.",
        "",
        "| Workload | Repeat | Status | Report |",
        "| --- | ---: | --- | --- |",
    ]
    for run in report["runs"]:
        relative = Path(run["report"]).relative_to(folder)
        lines.append(
            f"| {run['workload']} | {run['repeat']} | {run['status']} | "
            f"[JSON]({relative.as_posix()}) |"
        )
    if summary["performance"]:
        lines.extend(
            [
                "",
                f"Timing scope: `{summary['timing_scope']}`.",
                "",
                "| Workload / case | Candidate median ms | Baseline median ms | "
                "Median speedup | Repeat range |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in summary["performance"]:
            lines.append(
                f"| {row['workload']} / {row['case']} | "
                f"{row['candidate_median_ms']:.6f} | "
                f"{row['reference_median_ms']:.6f} | "
                f"{row['median_speedup']:.3f}× | "
                f"{row['min_speedup']:.3f}–{row['max_speedup']:.3f}× |"
            )
    lines.extend(["", summary["note"], ""])
    (folder / "suite.md").write_text("\n".join(lines))


def run_suite(
    manifest,
    *,
    backend,
    output_dir,
    selected=None,
    repeats=3,
    seed=0,
    trials=5,
    warmup=20,
    iterations=200,
    timeout=1200,
    max_seconds=7200,
    compiler_args=None,
    correctness_only=False,
):
    for name, value, minimum in (
        ("repeats", repeats, 1),
        ("seed", seed, 0),
        ("trials", trials, 1),
        ("warmup", warmup, 0),
        ("iterations", iterations, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if backend not in {"cpu", "simulator", "neuron"}:
        raise ValueError("Unsupported backend")
    workloads = load_manifest(manifest, selected)
    if backend == "cpu" and any("cpu_candidate" not in row for row in workloads):
        raise ValueError("CPU suites require an explicit cpu_candidate for each workload")
    folder = Path(output_dir).resolve()
    if (folder / "suite.json").exists():
        raise ValueError("Use a new output directory to preserve the previous suite")
    report = {
        "schema_version": 1,
        "config": {
            "backend": backend,
            "repeats": repeats,
            "seed": seed,
            "trials": trials,
            "warmup": warmup,
            "iterations": iterations,
            "timeout": timeout,
            "max_seconds": max_seconds,
            "compiler_args": compiler_args or [],
            "correctness_only": correctness_only,
            "workloads": [x["name"] for x in workloads],
            "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        },
        "runs": [],
    }
    expected = len(workloads) * repeats
    # Allocate disjoint deterministic seed ranges for runs, including benchmark input streams.
    stride = max(len(json.loads(Path(x["spec"]).read_text())["cases"]) for x in workloads)
    stride *= trials + warmup + iterations
    if seed + expected * stride >= 2**63:
        raise ValueError("Suite seed range exceeds supported values")
    start = time.monotonic()
    report["summary"] = summarize_runs([], expected)
    write_summary(folder, report)
    for repeat in range(repeats):
        # Rotate workload order between repeats to expose/reduce fixed-order drift.
        offset = repeat % len(workloads)
        for row in workloads[offset:] + workloads[:offset]:
            remaining = max_seconds - (time.monotonic() - start)
            if remaining <= 0:
                break
            run_seed = seed + len(report["runs"]) * stride
            candidate = row["cpu_candidate"] if backend == "cpu" else row["candidate"]
            identity = None
            try:
                identity = {
                    "reference_sha256": source_info(row["reference"])["sha256"],
                    "candidate_sha256": source_info(candidate)["sha256"],
                    "spec_sha256": hashlib.sha256(Path(row["spec"]).read_bytes()).hexdigest(),
                }
                result = evaluate(
                    row["reference"],
                    candidate,
                    row["spec"],
                    backend=backend,
                    output_dir=folder / "runs",
                    seed=run_seed,
                    trials=trials,
                    warmup=warmup,
                    iterations=iterations,
                    timeout=min(timeout, remaining),
                    compiler_args=compiler_args,
                    correctness_only=correctness_only,
                )
                # Use the source actually read by evaluate, not only the preflight snapshot.
                source = result.get("config", {}).get("source", {})
                if source:
                    identity["reference_sha256"] = source["reference"]["sha256"]
                    identity["candidate_sha256"] = source["candidate"]["sha256"]
                    identity["spec_sha256"] = hashlib.sha256(
                        json.dumps(result["config"]["spec"], sort_keys=True).encode()
                    ).hexdigest()
            except Exception as exc:
                error_file = folder / f"suite-error-{len(report['runs'])}.json"
                result = {
                    "status": "error",
                    "correct": False,
                    "cases": [],
                    "error": {
                        "category": "suite_infrastructure_error",
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                    "artifacts": {"report": str(error_file)},
                }
                error_file.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            report["runs"].append(
                {
                    "workload": row["name"],
                    "repeat": repeat + 1,
                    "status": result["status"],
                    "correct": result["correct"],
                    "seed": run_seed,
                    "identity": identity,
                    "cases": result["cases"],
                    "error": result.get("error"),
                    "report": result["artifacts"]["report"],
                }
            )
            report["summary"] = summarize_runs(report["runs"], expected)
            write_summary(folder, report)
    report["elapsed_s"] = time.monotonic() - start
    report["summary"] = summarize_runs(report["runs"], expected)
    write_summary(folder, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmarks/suite.json")
    parser.add_argument("--backend", choices=["cpu", "simulator", "neuron"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples", nargs="+", dest="selected")
    for name, default in (
        ("repeats", 3),
        ("seed", 0),
        ("trials", 5),
        ("warmup", 20),
        ("iterations", 200),
    ):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--compiler-arg", action="append", dest="compiler_args")
    parser.add_argument("--correctness-only", action="store_true")
    try:
        result = run_suite(**vars(parser.parse_args(argv)))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result["summary"], indent=2))
    return 0 if result["summary"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
