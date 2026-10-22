#!/usr/bin/env python3
"""Compare reviewed failure injections against baseline and current evaluator sources.

Uses real evaluator subprocesses on CPU. It does not execute OS escapes, checker
replacement, or credential-access attacks. Source snapshots and complete reports
are retained locally; the compact output excludes local absolute paths.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {"backend": "cpu", "trials": 2, "warmup": 2, "iterations": 4, "seed": 0, "threads": 1}
SPEC = {"cases": [{"name": "random", "inputs": [{"shape": [2, 3]}, {"shape": [2, 3]}]}]}


def runtime_environment():
    """Collect metadata even when every worker dies before writing environment."""
    versions = {}
    for name in ("torch", "numpy", "nki", "nki-stdlib", "neuronx-cc", "torch-neuronx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "captured_by": "matrix_driver",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
    }


def phase_switch(action, after=4):
    return (
        "calls = 0\ndef kernel(a, b):\n global calls\n calls += 1\n"
        f" if calls > {after}:\n"
        + "".join(f"  {line}\n" for line in action.splitlines())
        + " return a + b\n"
    )


def stale_cache(after):
    return (
        "calls = 0\ncached = None\ndef kernel(a, b):\n global calls, cached\n"
        " calls += 1\n"
        f" if calls <= {after}: cached = a + b\n"
        " return cached\n"
    )


def corpus():
    # Hand-chosen diagnostic examples, not a sampled population of attacks.
    return [
        ("correct_add", "passed", "control", "def kernel(a, b): return a + b", None),
        ("correct_commuted_add", "passed", "control", "def kernel(a, b): return b + a", None),
        (
            "always_subtract",
            "incorrect",
            "numerical",
            "def kernel(a, b): return a - b",
            "At a=1, b=2, expected 3 but returns -1.",
        ),
        (
            "always_wrong_dtype",
            "incorrect",
            "dtype",
            "def kernel(a, b): return (a + b).double()",
            "Float32 inputs require float32 output; returns float64.",
        ),
        (
            "tail_corruption",
            "incorrect",
            "numerical",
            "def kernel(a, b):\n out = a + b\n out[:, -1] = 0\n return out",
            "At a=1, b=2, expected last column 3 but returns 0.",
        ),
        (
            "warmup_wrong_output",
            "incorrect",
            "numerical",
            phase_switch("return a - b", 2),
            "From invocation 3: at a=1, b=2, expected 3 but returns -1.",
        ),
        (
            "timed_wrong_output",
            "incorrect",
            "numerical",
            phase_switch("return a - b"),
            "From invocation 5: at a=1, b=2, expected 3 but returns -1.",
        ),
        (
            "last_timed_wrong_output",
            "incorrect",
            "numerical",
            phase_switch("return a - b", 7),
            "Invocation 8: at a=1, b=2, expected 3 but returns -1.",
        ),
        (
            "warmup_stale_cache",
            "incorrect",
            "stale_output",
            stale_cache(2),
            "After caching a=1,b=2 output 3, new a=4,b=5 needs 9 but returns 3.",
        ),
        (
            "timed_stale_cache",
            "incorrect",
            "stale_output",
            stale_cache(4),
            "After caching a=1,b=2 output 3, new a=4,b=5 needs 9 but returns 3.",
        ),
        (
            "timed_mutation",
            "incorrect",
            "input_mutation",
            phase_switch("a.add_(b)\nreturn a"),
            "From invocation 5: a=1,b=2 yields correct output 3 but changes input a to 3.",
        ),
        (
            "timed_wrong_dtype",
            "incorrect",
            "dtype",
            phase_switch("return (a + b).double()"),
            "From invocation 5: float32 output contract is violated by float64 output.",
        ),
        (
            "timed_nan",
            "incorrect",
            "numerical",
            phase_switch("return (a + b) * float('nan')"),
            "From invocation 5: at a=1,b=2, expected finite 3 but returns NaN.",
        ),
        (
            "warmup_exception",
            "error",
            "execution_error",
            phase_switch("raise RuntimeError('injected warmup failure')", 2),
            "Invocation 3 raises instead of returning an output.",
        ),
        (
            "timed_exception",
            "error",
            "execution_error",
            phase_switch("raise RuntimeError('injected timed failure')"),
            "Invocation 5 raises instead of returning an output.",
        ),
        (
            "timed_timeout",
            "timeout",
            "execution_timeout",
            "import time\n" + phase_switch("time.sleep(60)"),
            "Invocation 5 sleeps for 60 seconds, beyond the configured evaluation deadline.",
        ),
    ]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(destination, ref=None):
    if ref:
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", ref, "src/nki_eval"], cwd=ROOT, text=True
        ).splitlines()
        sources = {
            name: subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=ROOT)
            for name in names
            if name.endswith(".py")
        }
    else:
        sources = {
            path.relative_to(ROOT).as_posix(): path.read_bytes()
            for path in sorted((ROOT / "src/nki_eval").rglob("*.py"))
        }
    manifest = {}
    for name, content in sorted(sources.items()):
        target = destination / Path(name).relative_to("src")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest[name] = digest(content)
    return {
        "sha256": digest(json.dumps(manifest, sort_keys=True).encode()),
        "files": manifest,
    }


def run_version(config_path):
    from nki_eval import evaluate

    config = json.loads(config_path.read_text())
    observations = {}
    for item in config["cases"]:
        report = evaluate(
            config["reference"],
            item["candidate"],
            SPEC,
            output_dir=Path(config["directory"]) / item["id"],
            timeout=config["timeout"],
            **SETTINGS,
        )
        error = report.get("error", {})
        check = error.get("check")
        if check is None:
            for case in report.get("cases", []):
                for trial in case["trials"]:
                    if not trial["candidate"]["passed"]:
                        check = trial["candidate"]
                        break
        observations[item["id"]] = {
            "status": report["status"],
            "correct": report["correct"],
            "performance_published": report["performance"] is not None,
            "error_category": error.get("category"),
            "stage": error.get("stage"),
            "phase": error.get("phase"),
            "actor": error.get("actor"),
            "iteration": error.get("iteration"),
            "seed": error.get("seed"),
            "check": check,
            "report": Path(report["artifacts"]["report"]).relative_to(config["bundle"]).as_posix(),
        }
    write_json(Path(config["observations"]), observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", default="0ebffec")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/reliability-latest.json")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs/reliability")
    parser.add_argument("--timeout", type=float, default=8)
    parser.add_argument("--run-version", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_version:
        run_version(args.run_version)
        return
    baseline_commit = subprocess.check_output(
        ["git", "rev-parse", f"{args.baseline_ref}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    bundle = args.runs_dir.resolve() / uuid.uuid4().hex
    bundle.mkdir(parents=True)
    started_at = datetime.now(timezone.utc).isoformat()
    script_bytes = Path(__file__).read_bytes()
    (bundle / "regression_matrix.py").write_bytes(script_bytes)
    reference = bundle / "reference.py"
    reference.write_text("def reference(a, b): return a + b\n")
    cases = []
    for name, verdict, kind, code, counterexample in corpus():
        path = bundle / "submissions" / f"{name}.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(code + "\n")
        cases.append(
            {
                "id": name,
                "expected_status": verdict,
                "kind": kind,
                "candidate": f"{path}:kernel",
                "source_sha256": digest(path.read_bytes()),
                "counterexample": counterexample,
            }
        )
    versions = {}
    for label, ref in (("baseline", baseline_commit), ("hardened", None)):
        sources = bundle / "sources" / label
        manifest = snapshot(sources, ref)
        observations_path = bundle / f"{label}-observations.json"
        config = bundle / f"{label}-request.json"
        write_json(
            config,
            {
                "reference": f"{reference}:reference",
                "cases": cases,
                "directory": str(bundle / label),
                "bundle": str(bundle),
                "timeout": args.timeout,
                "observations": str(observations_path),
            },
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(sources)
        subprocess.run(
            [sys.executable, str(bundle / "regression_matrix.py"), "--run-version", str(config)],
            env=env,
            cwd=bundle,
            check=True,
        )
        observations = json.loads(observations_path.read_text())
        summary = {
            "controls_passed": 0,
            "controls_total": 0,
            "functional_mutants_rejected": 0,
            "functional_mutants_total": 0,
            "execution_failures_reported": 0,
            "execution_failures_total": 0,
            "false_acceptances": 0,
        }
        for case in cases:
            observed = observations[case["id"]]
            observed["expected_status_matched"] = observed["status"] == case["expected_status"]
            if case["kind"] == "control":
                summary["controls_total"] += 1
                summary["controls_passed"] += observed["status"] == "passed"
            elif case["expected_status"] == "incorrect":
                summary["functional_mutants_total"] += 1
                summary["functional_mutants_rejected"] += observed["status"] == "incorrect"
            else:
                summary["execution_failures_total"] += 1
                summary["execution_failures_reported"] += observed["expected_status_matched"]
            if case["kind"] != "control":
                summary["false_acceptances"] += observed["status"] == "passed"
        versions[label] = {
            "git_commit": ref,
            "source": manifest,
            "summary": summary,
            "observations": observations,
        }
    compact = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "reviewed_cpu_failure_injection_matrix",
        "settings": {**SETTINGS, "timeout_seconds": args.timeout},
        "workload": SPEC,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "environment": runtime_environment(),
        "script_sha256": digest(script_bytes),
        "reference_sha256": digest(reference.read_bytes()),
        "cases": [{k: v for k, v in case.items() if k != "candidate"} for case in cases],
        "versions": versions,
        "evidence_directory": bundle.relative_to(ROOT).as_posix()
        if bundle.is_relative_to(ROOT)
        else bundle.name,
        "limitations": [
            "Hand-chosen corpus: 2 controls, 11 functional-contract mutants, 3 execution failures.",
            "CPU harness experiment; does not establish NKI device correctness or performance.",
            "No checker replacement, OS escape, credential access, or hostile isolation attacks.",
            "Public deterministic inputs and a shared checker process are not a security boundary.",
        ],
    }
    write_json(args.output.resolve(), compact)
    print(json.dumps({label: item["summary"] for label, item in versions.items()}, indent=2))


if __name__ == "__main__":
    main()
