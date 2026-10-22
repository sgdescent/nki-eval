# Result contract (schema version 1)

`result.json` is the source of truth. The CLI prints a short summary and exits nonzero on failure.

| Field | Meaning |
| --- | --- |
| `status` | `passed`, `incorrect`, `error`, or `timeout` |
| `correct` | Initial trials and any completed warmup/timed correctness checks passed; becomes false for a benchmark mismatch, but can remain true after a later execution error |
| `backend` | `neuron`, `simulator`, or `cpu`; always inspect before comparing performance |
| `config` | Workload, entries, source hashes, trial/timing settings, compiler flags |
| `cases[].trials` | Per-seed candidate and, where applicable, baseline comparisons |
| `cases[].preparation` | Compile durations/artifact paths or simulator details |
| `cases[].performance` | Raw latency samples and statistics; null until all cases pass correctness |
| `cases[].performance.validation` | Benchmark protocol, fresh input policy, validation placement, checked pair counts, and seed range |
| `cases[].benchmark_validation` | Failure details when a warmup or measured output violates the contract |
| `performance` | Aggregate performance, null on incorrect results or when timing is disabled |
| `native_check` | Conservative direct-import/JIT lint result, disabled for CPU debug |
| `error` | Failure category, stage, message, and usually traceback/case |
| `environment` | Python/platform/package versions and selected Neuron settings |
| `artifacts` | Report, log, and evaluation directory paths |

Failure categories include `configuration_error`, `dependency_error`, `policy_error`,
`compile_error`, `runtime_error`, `correctness_error`, `worker_crash`, and `timeout`.
A dependency or compiler failure is not evidence of mathematical incorrectness. A baseline
comparison failure also invalidates performance even if the candidate itself passes.

Successful tensor comparisons contain shape, dtype, element count, mismatch count, and finite
error metrics. Nonfinite mismatches appear in the count and first-mismatch diagnostic. Undefined
or overflowed error metrics are JSON null; diagnostics stringify NaN/Inf and large integer values.
Reports use standard JSON, without nonstandard `NaN` or `Infinity` literals.

Performance is correctness-gated across the entire workload. Every warmup and measured output
is checked against eager PyTorch outside the timer, and input mutation is checked for both
candidate and baseline. Any mismatch produces `status=incorrect`; execution exceptions produce
`status=error`. Both remove **all** earlier case and aggregate timings. Failures report the actor,
phase, iteration, and seed where available. Consumers should only score `status=passed`
hardware reports with the desired timing scope. CPU/simulator success is not hardware validation.

Timing reports include `min`, `median`, `p90`, `p99`, `mean`, population `std`, `count`, and raw
`samples`, all in milliseconds. Percentiles use NumPy's linear interpolation. Serial calls per
second uses inverse mean latency and does not predict concurrent serving throughput. Effective
TFLOP/s and GB/s use median candidate latency and explicit workload-supplied counts.

The seed for case index `i`, trial `j` is `base_seed + i*trials + j`. Input generation uses a local
CPU PyTorch generator. Compile examples use trial zero's inputs. After initial trials, fresh
benchmark seeds start at `base_seed + trials*number_of_cases`, with each case consuming
`warmup + iterations` seeds. Candidate and baseline use equal values in separate allocations.
Constant distributions may yield identical values across seeds. Global random
seeds are set once per worker for reproducible initialization, not to support stochastic kernels.
The timeout covers process startup, imports, all compilations, correctness, and timing together.

The new benchmark protocol is `fresh_inputs_checked_v1`. Older reports lacking this identifier
used repeated fixed inputs without checks during timing. The timing scope string is unchanged,
but the surrounding protocol is different; do not combine the two in one repeated experiment.

This release does not implement scalar RL rewards, distributed queues, automatic retries, or a
strong anti-cheating verifier. Its JSON can be consumed by an external agent loop; define reward
policy separately and run generated programs within an actual security boundary.
