import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_startup_timeouts_preserve_failed_controls_and_matrix(tmp_path):
    output = tmp_path / "matrix.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/regression_matrix.py"),
            "--baseline-ref",
            "HEAD",
            "--timeout",
            "0.001",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    matrix = json.loads(output.read_text())
    assert matrix["environment"]["captured_by"] == "matrix_driver"
    assert matrix["environment"]["packages"]["torch"]
    for version in matrix["versions"].values():
        assert version["summary"]["controls_total"] == 2
        assert version["summary"]["controls_passed"] == 0
        assert len(version["observations"]) == 16
        for name in ("correct_add", "correct_commuted_add"):
            observation = version["observations"][name]
            assert observation["status"] == "timeout"
            assert observation["expected_status_matched"] is False
            receipt = tmp_path / "runs" / matrix["evidence_directory"] / observation["report"]
            report = json.loads(receipt.read_text())
            assert "environment" not in report  # Exercise the missing metadata failure path.
