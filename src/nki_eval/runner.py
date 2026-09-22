"""Process lifecycle management. This isolates failures, not hostile code."""

import json
import math
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from .loading import source_info, split_entry


def evaluate(
    reference,
    candidate,
    spec,
    *,
    backend="neuron",
    output_dir="runs",
    trials=5,
    seed=0,
    warmup=10,
    iterations=100,
    timeout=600,
    threads=1,
    compiler_args=None,
    correctness_only=False,
):
    """Evaluate file.py:symbol entries using a JSON workload path or dictionary.

    Returns a JSON-serializable report. Each call creates a unique artifact directory.
    A passed simulator/CPU report does not establish hardware correctness or performance.
    """
    if backend not in {"neuron", "simulator", "cpu"}:
        raise ValueError("backend must be neuron, simulator, or cpu")
    for name, value, minimum in (
        ("trials", trials, 1),
        ("warmup", warmup, 0),
        ("iterations", iterations, 1),
        ("threads", threads, 1),
        ("seed", seed, 0),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if seed > 2**63 - 1:
        raise ValueError("seed is too large")
    if compiler_args is not None and (
        not isinstance(compiler_args, list) or not all(isinstance(x, str) for x in compiler_args)
    ):
        raise ValueError("compiler_args must be a list of strings")
    if not isinstance(spec, dict):
        spec = json.loads(Path(spec).read_text())
    if isinstance(spec, dict) and isinstance(spec.get("cases"), list):
        if seed + trials * len(spec["cases"]) > 2**63:
            raise ValueError("Trial seeds exceed the supported range")
    entries = {}
    for name, entry in (("reference", reference), ("candidate", candidate)):
        path, symbol = split_entry(entry)
        entries[name] = f"{path}:{symbol}"
    folder = Path(output_dir).resolve() / uuid.uuid4().hex
    folder.mkdir(parents=True)
    request = {
        **entries,
        "spec": spec,
        "backend": backend,
        "trials": trials,
        "seed": seed,
        "warmup": warmup,
        "iterations": iterations,
        "threads": threads,
        "compiler_args": compiler_args or [],
        "correctness_only": correctness_only,
        "source": {name: source_info(entry) for name, entry in entries.items()},
    }
    (folder / "request.json").write_text(json.dumps(request, indent=2, allow_nan=False) + "\n")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    # Preserve an editable/source checkout even when the worker changes directory.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parent.parent), env.get("PYTHONPATH", "")]
    )
    timed_out = False
    with (folder / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "nki_eval.worker", str(folder)],
            cwd=folder,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Kill the whole process group, including compiler descendants on timeout/crash.
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.kill()
            process.wait()
    report_path = folder / "result.json"
    if not timed_out and process.returncode == 0 and report_path.is_file():
        result = json.loads(report_path.read_text())
    else:
        progress = folder / "progress.json"
        context = json.loads(progress.read_text()) if progress.exists() else {"stage": "startup"}
        result = {
            "schema_version": 1,
            "status": "timeout" if timed_out else "error",
            "correct": False,
            "backend": backend,
            "cases": [],
            "performance": None,
            "config": request,
            "error": {
                "category": "timeout" if timed_out else "worker_crash",
                **context,
                "message": f"Exceeded {timeout}s"
                if timed_out
                else f"Worker exited with code {process.returncode}; inspect worker.log",
            },
        }
    result["artifacts"] = {
        "directory": str(folder),
        "report": str(report_path),
        "log": str(folder / "worker.log"),
    }
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
