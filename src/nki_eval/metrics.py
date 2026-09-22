import math
import time

import numpy as np

from .spec import clone_inputs


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


def benchmark(candidate, reference, inputs, warmup, iterations, scope, case):
    for _ in range(warmup):
        candidate(*clone_inputs(inputs))
        reference(*clone_inputs(inputs))
    samples = {"candidate": [], "reference": []}
    functions = {"candidate": candidate, "reference": reference}
    for i in range(iterations):
        # Alternate ordering to reduce one-sided drift and clone outside the timer.
        order = ("candidate", "reference") if i % 2 == 0 else ("reference", "candidate")
        for name in order:
            args = clone_inputs(inputs)
            start = time.perf_counter_ns()
            result = functions[name](*args)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            samples[name].append(elapsed_ms)
            del result
    stats = {name: summarize(values) for name, values in samples.items()}
    latency = stats["candidate"]["median"]
    output = {
        "timing_scope": scope,
        **stats,
        "speedup": stats["reference"]["median"] / latency,
        "serial_calls_per_second": 1000 / stats["candidate"]["mean"],
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
