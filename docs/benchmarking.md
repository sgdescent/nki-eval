# Repeated suites and reproducible reporting

`nki-eval-suite` runs a versioned workload manifest serially, rotates workload order between
repeats, assigns disjoint deterministic seed ranges, and writes `suite.json` plus a readable
`suite.md` table. It does not create instances or launch work in the background. Use an activated
SDK environment with the intended device reserved, and retain the source checkout.

## Check compatibility before timing

```bash
export NEURON_PLATFORM_TARGET_OVERRIDE=trn1
export NEURON_RT_VISIBLE_CORES=0
python -m nki_eval.suite --manifest benchmarks/suite.json \
  --backend simulator --repeats 1 --correctness-only \
  --output-dir runs/simulator-suite --max-seconds 1200
```

Simulator success is a separate result from hardware success. Use `--examples row_sum softmax`
to select workloads explicitly. A fresh output directory is required so an earlier result is
not silently overwritten. Unsupported examples must be recorded as such before selecting a
smaller performance suite; do not quietly drop failures from a claimed full-suite denominator.

## Collect a hardware suite

```bash
python -m nki_eval.suite --manifest benchmarks/suite.json \
  --backend neuron --repeats 3 --trials 5 --warmup 20 --iterations 200 \
  --compiler-arg=--target=trn1 --compiler-arg=--auto-cast=none \
  --timeout 1200 --max-seconds 5400 \
  --output-dir runs/trainium-repeats
```

Both implementations get the same compiler options. The time limit covers this suite, not the
instance's billing lifecycle; arrange an independent instance stop. It checks remaining time
between evaluations and limits each worker to the remaining allowance. Reports are saved after
each completed or failed evaluation. Compiler work and validation can make the total wall time
much larger than summed request latencies.

The benchmark protocol `fresh_inputs_checked_v1` checks candidate and compiled-baseline outputs
against eager PyTorch for every warmup and timed pair. Input generation, eager reference calls,
copying, and comparison occur outside the timed interval. The resulting cache/workload state
differs from the original unchecked repeated-input protocol. Do not pool the protocols.

## Interpret the summary

The full manifest currently specifies six families and 18 cases. Three repeats schedule 18
evaluations and 54 case runs. A complete timing suite requires all scheduled evaluations and
cases to pass. Raw reports remain the authority for the configuration, actual comparisons,
samples, and package versions. The suite saves entry-file/spec identity and per-run seeds;
imported dependencies are not captured by these entry-file hashes.

For each case, the summary reports the median candidate and baseline latency across runs,
the median of per-run speedup ratios, and the smallest/largest observed ratio. The ratio of the
displayed medians can differ from the median paired ratio. Ranges describe these repeats;
they are not confidence intervals. The cross-case geometric mean gives each case equal weight,
so families with more cases receive more weight. No aggregate performance is published for
failed or incomplete suites, correctness-only runs, mixed scopes/protocols, or source drift.

CPU debug suites require a separately named `cpu_candidate` in the manifest. The catalog
provides one for addition; use `--backend cpu --examples add` to test the suite locally. This
does not execute NKI. A correct CPU reference is not silently substituted as an NKI candidate.

## Before and after evaluator comparison

```bash
python scripts/regression_matrix.py --help
```

The regression driver snapshots the old and new evaluator sources and tests the same reviewed
controls and defects using real worker subprocesses. Read [the experiment](reliability-results.md)
for its exact command, measured counts, and limitations. This is a CPU harness study, separate
from native operator correctness and hardware timing. Keep the frozen script, module hashes,
full reports, and environment alongside the compact public matrix.
