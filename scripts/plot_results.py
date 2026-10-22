#!/usr/bin/env python3
"""Render evidence figures from saved JSON, without running any evaluations.

Requires matplotlib. For example:
  MPLCONFIGDIR=/tmp/nki-eval-mpl python scripts/plot_results.py \
    --reliability benchmarks/reliability-20261022.json \
    --suite validation/repeated-suite/suite.json

All counts and plotted values come from the supplied reports. Incomplete,
failed, changed-source, or mixed-protocol suites are rejected, not plotted.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

INK = "#172b3a"
MUTED = "#536675"
BLUE = "#426b9a"
TEAL = "#087e8b"
CORAL = "#b83f38"
GREEN_PALE = "#d9eee8"
RED_PALE = "#fae0dc"
GRID = "#dfe5e9"


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure, output_dir, stem):
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
        if suffix == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
        print(path)
    plt.close(figure)


def label(identifier):
    return identifier.replace("_", " ").replace("fp32", "FP32").replace("bf16", "BF16")


def reliability_figure(path, output_dir):
    report = json.loads(path.read_text())
    if report["settings"]["backend"] != "cpu":
        raise ValueError("The reliability figure is specifically a CPU harness experiment")
    versions = report["versions"]
    keys = ("baseline", "hardened")
    if not all(key in versions for key in keys):
        raise ValueError("Expected baseline and hardened observations in reliability JSON")
    cases = report["cases"]
    defects = [case for case in cases if case["expected_status"] == "incorrect"]
    controls = [case for case in cases if case["kind"] == "control"]
    failures = [case for case in cases if case["expected_status"] in ("error", "timeout")]
    if not defects or not controls:
        raise ValueError("Reliability figure requires defects and positive controls")

    counts = []
    for key in keys:
        observations = versions[key]["observations"]
        counts.append(
            {
                "defects": sum(observations[c["id"]]["status"] == "incorrect" for c in defects),
                "controls": sum(observations[c["id"]]["status"] == "passed" for c in controls),
                "failures": sum(
                    observations[c["id"]]["status"] == c["expected_status"] for c in failures
                ),
            }
        )

    fig = plt.figure(figsize=(12.5, 8.3))
    bars = fig.add_axes((0.07, 0.39, 0.29, 0.37))
    matrix = fig.add_axes((0.73, 0.21, 0.22, 0.58))
    fig.text(0.07, 0.94, "Reliability of benchmark-time checking", fontsize=21, weight="bold")
    fig.text(
        0.07,
        0.897,
        f"Hand-chosen CPU corpus: {len(defects)} functional defects, "
        f"{len(controls)} correct controls, {len(failures)} execution failures",
        fontsize=11,
        color=MUTED,
    )

    names = ("Original", "Revised")
    for i, (name, count, color) in enumerate(zip(names, counts, (BLUE, TEAL))):
        y = 1 - i
        bars.barh(y, len(defects), color="#edf1f4", height=0.46)
        bars.barh(y, count["defects"], color=color, height=0.46)
        bars.text(
            0.02,
            y + 0.34,
            name,
            fontsize=12,
            weight="bold",
            va="bottom",
        )
        text_x = count["defects"] + 0.22
        bars.text(
            text_x,
            y,
            f"{count['defects']}/{len(defects)}",
            fontsize=17,
            weight="bold",
            va="center",
            color=color,
        )
    bars.set_xlim(0, len(defects) * 1.30)
    bars.set_ylim(-0.48, 1.65)
    bars.set_yticks([])
    bars.set_xticks(sorted(set((0, len(defects) // 2, len(defects)))))
    bars.set_xlabel("Functional defects rejected", labelpad=10)
    bars.set_axisbelow(True)
    bars.grid(axis="x", alpha=0.5)
    fig.text(0.07, 0.31, "Positive controls", fontsize=11, weight="bold")
    fig.text(
        0.07,
        0.275,
        "   ·   ".join(
            f"{name}: {count['controls']}/{len(controls)} passed"
            for name, count in zip(names, counts)
        ),
        fontsize=10,
        color=MUTED,
    )
    fig.text(0.07, 0.225, "Execution failures", fontsize=11, weight="bold")
    fig.text(
        0.07,
        0.19,
        "   ·   ".join(
            f"{name}: {count['failures']}/{len(failures)} reported"
            for name, count in zip(names, counts)
        ),
        fontsize=10,
        color=MUTED,
    )

    rows = controls + defects + failures
    matrix.set_xlim(-0.5, 1.5)
    matrix.set_ylim(len(rows) - 0.5, -0.5)
    matrix.set_xticks((0, 1), names)
    matrix.xaxis.tick_top()
    matrix.tick_params(axis="both", length=0, pad=9)
    matrix.set_yticks(range(len(rows)), [label(c["id"]).capitalize() for c in rows], fontsize=9)
    matrix.spines["bottom"].set_visible(False)
    for row, case in enumerate(rows):
        for column, key in enumerate(keys):
            observed = versions[key]["observations"][case["id"]]["status"]
            matched = observed == case["expected_status"]
            false_acceptance = observed == "passed" and case["expected_status"] != "passed"
            color = GREEN_PALE if matched else RED_PALE
            matrix.add_patch(
                Rectangle(
                    (column - 0.47, row - 0.43), 0.94, 0.86, facecolor=color, edgecolor="none"
                )
            )
            outcome = (
                "MISSED"
                if false_acceptance
                else {
                    "incorrect": "REJECTED",
                    "passed": "PASSED",
                    "error": "ERROR",
                    "timeout": "TIMEOUT",
                }.get(observed, observed.upper())
            )
            matrix.text(
                column,
                row,
                outcome,
                ha="center",
                va="center",
                fontsize=8.1,
                weight="bold",
                color=TEAL if matched else CORAL,
            )
    for boundary in (len(controls), len(controls) + len(defects)):
        matrix.axhline(boundary - 0.5, color="white", lw=3)
    fig.legend(
        handles=[
            Patch(facecolor=GREEN_PALE, label="Expected outcome"),
            Patch(facecolor=RED_PALE, label="Unexpected outcome"),
        ],
        loc="lower right",
        bbox_to_anchor=(0.96, 0.105),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.07,
        0.075,
        "Descriptive results for reviewed injections; no population inference. "
        "This does not establish native NKI or hostile-code isolation coverage.",
        fontsize=9.2,
        color=MUTED,
    )
    fig.text(0.07, 0.045, f"Source: {path.name}", fontsize=8.5, color=MUTED)
    save_figure(fig, output_dir, "cpu-reliability-matrix")


def suite_values(report):
    summary = report["summary"]
    if (
        not summary.get("complete")
        or not summary.get("all_passed")
        or not summary.get("performance")
    ):
        raise ValueError("Refusing to plot an incomplete, failed, or noncomparable suite")
    if report["config"]["backend"] != "neuron":
        raise ValueError("The performance figure requires a real Neuron hardware suite")
    expected_repeats = set(range(1, report["config"]["repeats"] + 1))
    groups = defaultdict(dict)
    identities = {}
    scopes, protocols = set(), set()
    for run in report["runs"]:
        if run["status"] != "passed" or not run["correct"]:
            raise ValueError("Failed run found inside supposedly passing suite")
        identity = run.get("identity")
        if not identity or (
            run["workload"] in identities and identities[run["workload"]] != identity
        ):
            raise ValueError("Missing or inconsistent source identity")
        identities[run["workload"]] = identity
        for case in run["cases"]:
            perf = case.get("performance")
            if not case["passed"] or not perf:
                raise ValueError("Case has no passing performance result")
            scopes.add(perf["timing_scope"])
            protocols.add(perf.get("validation", {}).get("protocol"))
            candidate, reference = perf["candidate"]["median"], perf["reference"]["median"]
            if not all(math.isfinite(x) and x > 0 for x in (candidate, reference)):
                raise ValueError("Invalid median timing")
            key = (run["workload"], case["name"])
            if run["repeat"] in groups[key]:
                raise ValueError("Duplicate workload/case/repeat")
            groups[key][run["repeat"]] = reference / candidate
    if scopes != {"neuron_host_request_including_transfers"}:
        raise ValueError("Mixed or unsupported timing scopes; do not mix host and device timing")
    if protocols != {"fresh_inputs_checked_v1"}:
        raise ValueError("Mixed or unchecked protocols; do not mix old and revised timings")
    if not groups or any(set(values) != expected_repeats for values in groups.values()):
        raise ValueError("Missing repeat observations")
    return groups, sorted(expected_repeats)


def performance_figure(path, output_dir):
    report = json.loads(path.read_text())
    groups, repeats = suite_values(report)
    keys = sorted(groups)
    rows = len(keys)
    fig, ax = plt.subplots(figsize=(12.5, max(8.2, rows * 0.35 + 2.5)))
    fig.subplots_adjust(left=0.38, right=0.95, top=0.82, bottom=0.18)
    fig.text(0.07, 0.95, "Trainium host request performance", fontsize=21, weight="bold")
    fig.text(
        0.07,
        0.907,
        f"{rows} cases · {len(repeats)} repeats · identical benchmark protocol · "
        "compiled PyTorch reference",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.07,
        0.871,
        "Timing includes transfers and host overhead; this is not device-only latency.",
        fontsize=10.5,
        color=MUTED,
    )
    palette = [BLUE, TEAL, "#c47c26", "#9160a7", "#bb5968"]
    colors = {repeat: palette[i % len(palette)] for i, repeat in enumerate(repeats)}
    all_ratios = [ratio for values in groups.values() for ratio in values.values()]
    minimum, maximum = min(all_ratios + [1.0]), max(all_ratios + [1.0])
    padding = max(0.025, (maximum - minimum) * 0.15)
    ax.set_xlim(max(0, minimum - padding), maximum + padding)
    ax.set_ylim(rows - 0.55, -0.65)
    last_family = None
    family_number = -1
    for y, key in enumerate(keys):
        if key[0] != last_family:
            family_number += 1
            last_family = key[0]
        if family_number % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color="#f4f6f8", zorder=0)
        ratios = list(groups[key].values())
        ax.plot((min(ratios), max(ratios)), (y, y), color="#a3adb5", lw=3, zorder=2)
        for i, repeat in enumerate(repeats):
            offset = (i - (len(repeats) - 1) / 2) * 0.12
            ax.scatter(
                groups[key][repeat],
                y + offset,
                color=colors[repeat],
                s=31,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
    ax.axvline(1, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.set_yticks(
        range(rows), [f"{label(family)}  /  {label(case)}" for family, case in keys], fontsize=9.3
    )
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlabel(
        "Speedup = reference median / candidate median    (higher is faster)", labelpad=13
    )
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[r],
            markeredgecolor="white",
            markersize=7,
            label=f"Repeat {r}",
        )
        for r in repeats
    ]
    handles += [Line2D([0], [0], color="#a3adb5", lw=3, label="Observed min–max")]
    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.96, 0.087),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.07,
        0.057,
        "Points are per-run ratios; ranges are descriptive, not confidence intervals. "
        "The dashed line denotes equal host latency.",
        fontsize=9,
        color=MUTED,
    )
    fig.text(0.07, 0.031, f"Source: {path.parent.name}/{path.name}", fontsize=8.5, color=MUTED)
    save_figure(fig, output_dir, "trainium-repeated-speedups")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reliability", type=Path, default=Path("benchmarks/reliability-20261022.json")
    )
    parser.add_argument("--suite", type=Path, help="Optional completed real-hardware suite JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("validation/figures"))
    args = parser.parse_args(argv)
    configure_style()
    reliability_figure(args.reliability, args.output_dir)
    if args.suite:
        performance_figure(args.suite, args.output_dir)


if __name__ == "__main__":
    main()
