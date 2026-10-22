"""One process per evaluation, with progress and stage-specific failure reporting."""

import importlib.metadata
import json
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .backends import BACKENDS, DependencyError
from .loading import lint_candidate, load
from .metrics import (
    BENCHMARK_PROTOCOL,
    BenchmarkExecutionError,
    BenchmarkValidationError,
    aggregate_speedup,
    benchmark,
    check_output,
)
from .spec import clone_inputs, inputs_for, validate


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def environment():
    versions = {}
    for name in ("torch", "numpy", "nki", "nki-stdlib", "neuronx-cc", "torch-neuronx"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "settings": {
            k: os.environ.get(k)
            for k in (
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
                "NKI_PRECISE_FP",
                "NEURON_CC_FLAGS",
                "NKI_SIMULATOR",
            )
        },
    }


def run(request, folder):
    result = {
        "schema_version": 1,
        "status": "error",
        "correct": False,
        "backend": request["backend"],
        "config": request,
        "cases": [],
        "performance": None,
        "environment": environment(),
    }
    stage = "validation"
    case_name = None

    def progress(value):
        nonlocal stage
        stage = value
        write_json(folder / "progress.json", {"stage": stage, "case": case_name})

    start = time.perf_counter()
    try:
        spec = validate(request["spec"])
        seed_count = request["trials"] * len(spec["cases"])
        if request["backend"] != "simulator" and not request["correctness_only"]:
            seed_count += (request["warmup"] + request["iterations"]) * len(spec["cases"])
        if request["seed"] + seed_count > 2**63:
            raise ValueError("Evaluation seeds exceed the supported range")
        torch.set_num_threads(request["threads"])
        torch.manual_seed(request["seed"])
        random.seed(request["seed"])
        np.random.seed(request["seed"] % (2**32))
        if request["backend"] == "simulator":
            os.environ["NKI_PRECISE_FP"] = "1"
        result["native_check"] = {
            "kind": "static_import_and_jit_lint",
            "security_boundary": False,
            "enabled": request["backend"] != "cpu",
        }
        if request["backend"] != "cpu":
            progress("candidate_policy")
            violations = lint_candidate(request["candidate"])
            result["native_check"]["violations"] = violations
            if violations:
                raise ValueError("; ".join(violations))
        progress("reference_load")
        reference = load(request["reference"])
        if isinstance(reference, type) and issubclass(reference, torch.nn.Module):
            reference = reference()
        if isinstance(reference, torch.nn.Module):
            reference = reference.cpu().eval()
        progress("candidate_load")
        candidate = load(request["candidate"])
        backend = BACKENDS[request["backend"]]()
        tolerance = {
            "rtol": spec.get("rtol", 1e-4),
            "atol": spec.get("atol", 1e-5),
            "equal_nan": spec.get("equal_nan", False),
        }
        prepared = []
        with torch.no_grad():
            for case_index, case in enumerate(spec["cases"]):
                case_name = case.get("name", f"case_{case_index}")
                row = {"name": case_name, "passed": False, "trials": [], "performance": None}
                case_passed = True
                result["cases"].append(row)
                seed = request["seed"] + case_index * request["trials"]
                example = inputs_for(case, seed)
                progress("compile" if request["backend"] == "neuron" else "prepare")
                fn, baseline, metadata = backend.prepare(
                    candidate,
                    reference,
                    clone_inputs(example),
                    folder / f"case_{case_index}",
                    request["compiler_args"],
                )
                row["preparation"] = metadata
                for trial in range(request["trials"]):
                    trial_seed = seed + trial
                    inputs = inputs_for(case, trial_seed)
                    progress("reference_execution")
                    expected = reference(*clone_inputs(inputs))
                    progress("candidate_execution")
                    args = clone_inputs(inputs)
                    actual = fn(*args)
                    check = check_output(expected, actual, inputs, args, **tolerance)
                    trial_result = {"seed": trial_seed, "candidate": check}
                    if baseline is not None:
                        progress("baseline_execution")
                        baseline_args = clone_inputs(inputs)
                        baseline_check = check_output(
                            expected, baseline(*baseline_args), inputs, baseline_args, **tolerance
                        )
                        trial_result["baseline"] = baseline_check
                        case_passed &= baseline_check["passed"]
                    case_passed &= check["passed"]
                    row["trials"].append(trial_result)
                row["passed"] = case_passed
                prepared.append((fn, baseline, case))
            result["correct"] = all(x["passed"] for x in result["cases"])
            if not result["correct"]:
                result["status"] = "incorrect"
                result["error"] = {
                    "category": "correctness_error",
                    "stage": "comparison",
                    "message": "At least one candidate or baseline trial failed",
                }
            else:
                if backend.timing_scope and not request["correctness_only"]:
                    benchmark_seed = request["seed"] + request["trials"] * len(spec["cases"])
                    for row, (fn, baseline, case) in zip(result["cases"], prepared):
                        case_name = row["name"]
                        progress("benchmark")
                        row["performance"] = benchmark(
                            fn,
                            baseline,
                            reference,
                            request["warmup"],
                            request["iterations"],
                            backend.timing_scope,
                            case,
                            benchmark_seed,
                            tolerance,
                        )
                        benchmark_seed += request["warmup"] + request["iterations"]
                    result["performance"] = {
                        "geometric_mean_speedup": aggregate_speedup(result["cases"]),
                        "timing_scope": backend.timing_scope,
                        "benchmark_protocol": BENCHMARK_PROTOCOL,
                        "baseline": "compiled_pytorch_neuron"
                        if request["backend"] == "neuron"
                        else "pytorch_cpu_debug",
                    }
                result["status"] = "passed"
    except BenchmarkValidationError as exc:
        result["status"] = "incorrect"
        result["correct"] = False
        row["passed"] = False
        row["benchmark_validation"] = {"passed": False, **exc.details}
        result["error"] = {
            "category": "correctness_error",
            "stage": "benchmark_validation",
            "case": case_name,
            "message": str(exc),
            **exc.details,
        }
    except Exception as exc:
        category = (
            "dependency_error"
            if isinstance(exc, (DependencyError, ModuleNotFoundError))
            else "compile_error"
            if isinstance(exc, SyntaxError)
            else "policy_error"
            if stage == "candidate_policy"
            else "compile_error"
            if stage == "compile"
            else "configuration_error"
            if stage == "validation"
            else "runtime_error"
        )
        result["status"] = "error"
        result["error"] = {
            "category": category,
            "stage": stage,
            "case": case_name,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        if isinstance(exc, BenchmarkExecutionError):
            result["error"].update(exc.details)
    # A later failure invalidates all partial timing, including earlier cases.
    if result["status"] != "passed":
        result["performance"] = None
        for case in result["cases"]:
            case["performance"] = None
    result["environment"] = environment()
    result["elapsed_s"] = time.perf_counter() - start
    return result


def main():
    folder = Path(sys.argv[1]).resolve()
    request = json.loads((folder / "request.json").read_text())
    result = run(request, folder)
    write_json(folder / "result.json", result)


if __name__ == "__main__":
    main()
