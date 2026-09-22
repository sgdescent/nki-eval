import argparse
import json
import sys
from pathlib import Path

from .runner import evaluate


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate NKI kernels against PyTorch references")
    parser.add_argument("--reference", required=True, help="reference.py:function or :Model")
    parser.add_argument("--candidate", required=True, help="kernel.py:kernel (@nki.jit)")
    parser.add_argument("--spec", required=True, help="JSON workload definition")
    parser.add_argument("--backend", choices=["neuron", "simulator", "cpu"], default="neuron")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--json", dest="json_path", help="Also write report to this path")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--compiler-arg", action="append", default=[], help="Repeat as --compiler-arg=--target=trn2"
    )
    parser.add_argument("--correctness-only", action="store_true")
    args = vars(parser.parse_args(argv))
    json_path = args.pop("json_path")
    args["compiler_args"] = args.pop("compiler_arg")
    try:
        result = evaluate(**args)
        if json_path:
            path = Path(json_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    except (ValueError, OSError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print(f"{result['status'].upper()} | backend={result['backend']}")
    for case in result["cases"]:
        print(f"  {case['name']}: {'PASS' if case['passed'] else 'FAIL'}")
        if perf := case.get("performance"):
            print(
                f"    median={perf['candidate']['median']:.6f} ms "
                f"p99={perf['candidate']['p99']:.6f} ms speedup={perf['speedup']:.3f}x"
            )
            print(f"    scope={perf['timing_scope']}")
    if error := result.get("error"):
        print(f"  {error['category']}: {error['message']}")
    print(f"Report: {result['artifacts']['report']}")
    if result["backend"] == "simulator":
        print("Simulator correctness only; no hardware performance measured.")
    elif result["backend"] == "cpu":
        print("CPU debug backend; this does not execute or validate NKI.")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
