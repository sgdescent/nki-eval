import math
import time

import numpy as np
import torch

from .compare import compare
from .spec import clone_inputs, inputs_for

BENCHMARK_PROTOCOL = "fresh_inputs_checked_v1"


class BenchmarkValidationError(RuntimeError):
    """A warmup or timed invocation violated the functional contract."""

    def __init__(self, details):
        self.details = details
        super().__init__(
            f"{details['actor']} failed {details['phase']} validation at "
            f"iteration {details['iteration']} (seed {details['seed']})"
        )


class BenchmarkExecutionError(RuntimeError):
    """Retain which invocation failed without publishing partial performance."""

    def __init__(self, details, cause):
        self.details = {**details, "exception_type": type(cause).__name__}
        super().__init__(
            f"{details['actor']} raised during {details['phase']} iteration "
            f"{details['iteration']} (seed {details['seed']}): {cause}"
        )


def check_output(expected, actual, inputs, args, **tolerance):
    """Compare an output and enforce the evaluator's immutable-input contract."""
    check = compare(expected, actual, **tolerance)
    for index, (before, after) in enumerate(zip(inputs, args)):
        if (
            isinstance(before, torch.Tensor)
            and not compare(before, after, rtol=0, atol=0, equal_nan=True)["passed"]
        ):
            check["passed"] = False
            check["problems"].append(f"Input {index} was mutated")
    return check


def summarize(samples):
    values = np.asarray(samples, dtype=np.float64)
    if not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Latency samples must be finite and positive")
    return {
        "unit": "ms",
        "count": len(samples),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "samples": samples,
    }


def benchmark(candidate, reference, oracle, warmup, iterations, scope, case, seed, tolerance):
    """Time serial calls, verifying every output on fresh deterministic input pairs.

    Input generation, cloning, eager oracle execution, and output/mutation checks
    are outside the timer. These checks affect the surrounding cache/workload
    state, so this is a validated request-latency protocol, not an isolated device
    microbenchmark. Seeds are reproducible and public; this is not a sandbox.
    """
    samples = {"candidate": [], "reference": []}
    functions = {"candidate": candidate, "reference": reference}
    for index in range(warmup + iterations):
        phase = "warmup" if index < warmup else "timed"
        iteration = index if phase == "warmup" else index - warmup
        trial_seed = seed + index
        inputs = inputs_for(case, trial_seed)
        context = {"phase": phase, "iteration": iteration, "seed": trial_seed}
        try:
            expected = oracle(*clone_inputs(inputs))
        except Exception as exc:
            raise BenchmarkExecutionError({**context, "actor": "oracle"}, exc) from exc
        # Both functions receive the same values, each in independent allocations.
        # Alternate ordering within each phase to reduce one-sided drift.
        order = ("candidate", "reference") if iteration % 2 == 0 else ("reference", "candidate")
        for name in order:
            args = clone_inputs(inputs)
            details = {**context, "actor": "baseline" if name == "reference" else name}
            try:
                start = time.perf_counter_ns()
                actual = functions[name](*args)
                elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            except Exception as exc:
                raise BenchmarkExecutionError(details, exc) from exc
            check = check_output(expected, actual, inputs, args, **tolerance)
            if not check["passed"]:
                raise BenchmarkValidationError({**details, "check": check})
            if phase == "timed":
                samples[name].append(elapsed_ms)
            del actual
    stats = {name: summarize(values) for name, values in samples.items()}
    latency = stats["candidate"]["median"]
    output = {
        "timing_scope": scope,
        **stats,
        "speedup": stats["reference"]["median"] / latency,
        "serial_calls_per_second": 1000 / stats["candidate"]["mean"],
        "validation": {
            "protocol": BENCHMARK_PROTOCOL,
            "input_policy": "fresh_seed_per_pair",
            "oracle": "eager_pytorch",
            "placement": "outside_timed_interval",
            "checked_warmup_pairs": warmup,
            "checked_timed_pairs": iterations,
            "seed_start": seed,
            "seed_end": seed + warmup + iterations - 1,
            "note": "Constant distributions may produce identical values across seeds",
        },
    }
    if "flops" in case:
        output["estimated_tflops_per_second"] = case["flops"] / (latency * 1e9)
    if "bytes_moved" in case:
        output["estimated_gb_per_second"] = case["bytes_moved"] / (latency * 1e6)
    output["estimates_note"] = "Workload-supplied operation/byte counts, not hardware counters"
    return output


def aggregate_speedup(cases):
    values = [x["performance"]["speedup"] for x in cases]
    return math.exp(sum(math.log(x) for x in values) / len(values))
