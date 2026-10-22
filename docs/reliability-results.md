# Reliability regression results

On October 22, 2026, a local CPU comparison found **eight false acceptances in the initial evaluator and zero in the revised evaluator** on a hand-chosen corpus. The initial evaluator rejected 3 of 11 functional-contract mutants; the revised evaluator rejected all 11. Both versions passed two correct controls and reported three injected execution failures.

This experiment tests the evaluator's shared correctness and timing pipeline. It does not establish native NKI correctness, accelerator performance, or protection against hostile code.

## What was measured

The experiment ran 16 reviewed Python submissions against two evaluator source snapshots, for **32 real subprocess evaluations**. The baseline was commit `0ebffec7e76fe1e0df52dc94c7053a294b8f285d`. The revised snapshot is identified by per-file SHA-256 hashes in [the machine-readable matrix](../benchmarks/reliability-20261022.json), rather than an unrecorded working-tree label.

| Outcome | Initial evaluator | Revised evaluator |
| --- | ---: | ---: |
| Correct controls passed | 2 / 2 | 2 / 2 |
| Functional-contract mutants rejected as incorrect | 3 / 11 | 11 / 11 |
| Invalid submissions accepted with performance results | 8 | 0 |
| Execution failures reported with the expected status | 3 / 3 | 3 / 3 |

The three execution failures are two exceptions and one timeout. They are reported separately and are **not counted as numerical correctness detections**. The functional corpus includes numerical errors, stale outputs, dtype violations and input mutation; it is not exclusively numerical.

All eight previously accepted mutants received performance results despite violating the functional contract. The revised evaluator returns `incorrect`, identifies the failing phase, and withholds all case and aggregate performance results.

## Per-submission observations

| Submission | Expected | Initial evaluator | Revised evaluator | Revised failure location |
| --- | --- | --- | --- | --- |
| Correct addition | Passed | Passed | Passed | — |
| Correct commuted addition | Passed | Passed | Passed | — |
| Subtraction throughout | Incorrect | Incorrect | Incorrect | Initial comparison |
| Wrong dtype throughout | Incorrect | Incorrect | Incorrect | Initial comparison |
| Corrupted last column | Incorrect | Incorrect | Incorrect | Initial comparison |
| Wrong output during warmup | Incorrect | **Passed** | Incorrect | Warmup, seed 2 |
| Wrong output during timing | Incorrect | **Passed** | Incorrect | Timed call 1, seed 4 |
| Wrong output on last timed call | Incorrect | **Passed** | Incorrect | Timed call 4, seed 7 |
| Stale answer during warmup | Incorrect | **Passed** | Incorrect | Warmup, seed 2 |
| Stale answer during timing | Incorrect | **Passed** | Incorrect | Timed call 1, seed 4 |
| Input mutation during timing | Incorrect | **Passed** | Incorrect | Timed call 1, seed 4 |
| Wrong dtype during timing | Incorrect | **Passed** | Incorrect | Timed call 1, seed 4 |
| NaN output during timing | Incorrect | **Passed** | Incorrect | Timed call 1, seed 4 |
| Exception during warmup | Error | Error | Error | Warmup, seed 2 |
| Exception during timing | Error | Error | Error | Timed call 1, seed 4 |
| Timeout during timing | Timeout | Timeout | Timeout | Benchmark stage |

Call numbers in this table are one-based. JSON diagnostic iteration indices are zero-based. Each mutant has an explicit counterexample in the matrix and its exact source in [the reproduction script](../scripts/regression_matrix.py). For example, a cached answer of 3 for inputs 1 and 2 is invalid when fresh inputs 4 and 5 require 9; a function that returns the correct sum but changes its first input violates the immutable-input contract.

## Protocol and implementation

Both versions use the same two float32 tensors of shape `[2, 3]`, generated from normal distributions. Settings are seed 0, two initial correctness trials, two warmup pairs, four timed pairs, one CPU thread and an eight-second evaluation deadline. Default comparison tolerances are `rtol=1e-4`, `atol=1e-5`, with NaNs unequal.

The initial implementation repeats its first example input during warmup and timing and discards returned outputs. The revised implementation uses protocol `fresh_inputs_checked_v1`:

1. Initial correctness trials retain their existing seed sequence and gate all timing.
2. Every warmup and timed pair receives newly generated inputs. Candidate and baseline receive equal values in separate allocations. In this experiment, initial trials use seeds 0–1, warmups 2–3 and timed calls 4–7.
3. An eager PyTorch reference produces the expected output for every pair. Both candidate and baseline outputs are checked, including dtype, structure, values and input mutation.
4. Input generation, input cloning, eager reference execution and comparisons occur outside the timed interval. Call ordering alternates between candidate-first and baseline-first.
5. Any validation failure invalidates all partial timings, including measurements from previously completed cases. The report includes the actor, phase, iteration, seed and comparison diagnostics.

Constant distributions can produce identical values across different seeds. Seeds are public and reproducible. This change detects the reviewed stale-output and phase-switch failures; it does not make input values secret.

The report schema remains version 1 with additive protocol and validation metadata. The public `evaluate` arguments are unchanged. The internal `benchmark` helper now receives the eager oracle, seed range and tolerances. Runtime errors can retain `correct: true` when initial correctness trials passed; consumers must require `status: passed` before accepting any result.

## Interpretation and limitations

The result supports a specific claim: the added validation closes eight demonstrated false-acceptance cases in this corpus without rejecting its two correct controls. It is not a statistical estimate of failure rates for arbitrary kernels, an adversarial security evaluation, or evidence of broad operator coverage.

The submissions run through the CPU debug backend and the shared harness. These runs alone do not establish native NKI correctness or device synchronization. Separately, the revised protocol passed 12 new cases on Trainium1 across row reduction, softmax, LayerNorm and matrix multiplication; [the native reports](../validation/native-20261022/native-new-families-summary.json) record that evidence. The original 21 October Trainium results use the earlier protocol and remain identified separately.

The evaluator still runs submitted code, the oracle, checker and result writer within the same worker process. It inherits host privileges and does not hide test seeds. This experiment deliberately excludes checker replacement, OS escapes and credential-access attacks. Stronger isolation and an external authority for validation/results remain necessary before treating untrusted agent submissions as secure evaluations.

Outside-timer validation adds total evaluation work and changes surrounding cache and scheduling state. Timings from the two protocols should not be pooled. The matrix reports verdicts, not kernel speedups.

## Reproduction and evidence

From the repository root with the project's Python dependencies installed:

```bash
python scripts/regression_matrix.py \
  --baseline-ref 0ebffec \
  --runs-dir runs/reliability-20261022 \
  --output benchmarks/reliability-20261022.json
```

The script preserves both source snapshots, a copy of the executed experiment script, all submitted programs, full evaluator reports and worker logs under a unique ignored run directory. The compact JSON records source and script hashes, exact settings, expected and observed verdicts, failure stages, counterexamples, environment details and paths relative to that evidence directory. No raw credentials or local absolute paths are included in the compact matrix. The driver captures its environment independently, so even startup timeouts that omit worker environment metadata still produce a complete receipt with failed control verdicts preserved. Failed controls must be investigated before interpreting rejection counts.

The regression tests additionally exercise a later case failing after an earlier case completed timing, a bad or mutating compiled-baseline stand-in, seed-range separation/overflow, and a controlled clock proving that eager-reference work and comparison costs lie outside the measured interval. The focused reliability suite passed 21 tests. A separate driver regression passed with a one-millisecond deadline, confirming that failed controls and missing worker metadata are preserved. These tests complement the recorded subprocess experiment; they are not substituted for its measurements.
