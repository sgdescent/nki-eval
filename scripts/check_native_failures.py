"""Verify two intentionally wrong NKI kernels are rejected by the real backend."""

import argparse
import json
from pathlib import Path

from nki_eval import evaluate

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["simulator", "neuron"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="trn1")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if (output / "checks.json").exists():
        parser.error("Use a new output directory to preserve prior results")
    observations = []
    for name in ("nki_wrong_add", "nki_wrong_tail"):
        result = evaluate(
            str(ROOT / "examples/add/reference.py") + ":reference",
            str(ROOT / "tests/fixtures" / f"{name}.py") + ":kernel",
            {
                "cases": [
                    {"name": "tails", "inputs": [{"shape": [129, 513]}, {"shape": [129, 513]}]}
                ]
            },
            backend=args.backend,
            output_dir=output / "runs",
            trials=5,
            seed=911,
            timeout=1200,
            compiler_args=[f"--target={args.target}", "--auto-cast=none"],
        )
        baseline_passed = all(
            trial.get("baseline", {"passed": True})["passed"]
            for case in result["cases"]
            for trial in case["trials"]
        )
        rejected = (
            result["status"] == "incorrect"
            and result["error"]["stage"] == "comparison"
            and result["performance"] is None
            and baseline_passed
        )
        observations.append(
            {
                "candidate": name,
                "expected_status": "incorrect",
                "rejected_as_expected": rejected,
                "report": result,
            }
        )
    report = {
        "backend": args.backend,
        "all_rejected_as_expected": all(row["rejected_as_expected"] for row in observations),
        "observations": observations,
    }
    (output / "checks.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "backend": args.backend,
                "all_rejected_as_expected": report["all_rejected_as_expected"],
                "cases": [
                    {
                        "candidate": row["candidate"],
                        "status": row["report"]["status"],
                        "rejected_as_expected": row["rejected_as_expected"],
                    }
                    for row in observations
                ],
            },
            indent=2,
        )
    )
    return 0 if report["all_rejected_as_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
