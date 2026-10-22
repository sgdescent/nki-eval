# Device profiling smoke test

On October 22, 2026, the installed Neuron profiler successfully captured and
exported a device trace for one validated NKI matmul case and its compiled
PyTorch baseline on `trn1.2xlarge`. This establishes a working profiling path
for this compiler/runtime combination. It is a compatibility smoke test, not
a repeated device performance benchmark or an additional correctness test.

## Workload and capture

The case was `matmul / multiple_tiles_bf16`: contiguous BF16 inputs
`lhs_t[256, 256]` and `rhs[256, 1024]`, computing `lhs_t.T @ rhs` to a BF16
`[256, 1024]` result with FP32 accumulation. Both compiled models had already
passed the evaluator's seeded correctness and checked benchmark runs. The
profiler used its own default allocated inputs, with no evaluator seed replay
or output comparison during capture.

The candidate and baseline were captured serially with only NeuronCore 0
visible. Each command executed its model three times and profiled the third
execution. Capture and export commands each had an external 120-second
timeout, with a ten-second kill grace period. No drivers, OS settings, or
security settings were changed.

## Observations

| Model | Profiled executions | Reported device duration | Matmul instructions | Trace events |
| --- | ---: | ---: | ---: | ---: |
| NKI candidate | 1 | 22.360086 µs | 8 | 1,133 |
| Compiled PyTorch reference | 1 | 23.202204 µs | 8 | 802 |

Durations come from the profiler's `total_time` field. The installed
`neuron-profile view --terminology` output explicitly defines that field in
seconds and excludes host-to-device transfers and host runtime/framework
overhead. The table converts seconds to microseconds by multiplying by
1,000,000. AWS's [profiler user guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.25.0/tools/neuron-sys-tools/neuron-profile-user-guide.html)
also documents the `summary-json` duration unit. A single trace per model is
insufficient to estimate run-to-run variation or establish a speedup.

`trace_count` includes instructions, events/semaphores, and DMA notifications;
it is not the number of executions or a count of arithmetic instructions.
These device durations must remain separate from the evaluator's repeated
host-request timings, which include transfer and host overhead.

## Tools and compatibility

- Profiler: `neuron-profile 2.29.18.0%kaena-tools/2.29@d5fe7ba`.
- Compiler: `2.24.5133.0+58f8de22`; compilation targeted `trn1` with
  `--auto-cast=none`.
- Runtime reported by the profile: `2.31.24 (0b044)`.
- Driver reported by the profile: `2.27.0`.
- NKI: `0.3.0+23928721754.g18aa1271`.

Both captures exited successfully. The first export attempts failed because
the profiler appended `_exec_3` to the requested trace filenames. The corrected
exports used `candidate_exec_3.ntff` and `reference_exec_3.ntff` and both exited
successfully. The failed attempts and their diagnostic logs are retained.

Both successful exports warned that `summary-json` disables SBUF/PSUM buffer
usage simulation over time. Consequently, these exports do not establish
buffer occupancy over time. No other export warning or error was reported.

## Evidence and reproduction

The [compact summary](../validation/profiling-20261022/summary.json) records the
scope, units, limitations, and observed values. The
[command metadata](../validation/profiling-20261022/metadata.json) preserves
the exact executed argument lists, timestamps, exit codes, source NEFF paths,
and binary hashes. Raw successful exports are available for the
[candidate](../validation/profiling-20261022/candidate-summary-attempt2.json)
and [reference](../validation/profiling-20261022/reference-summary-attempt2.json).
The installed metric definitions are saved in
[terminology output](../validation/profiling-20261022/terminology.stdout.txt).
An [artifact manifest](../validation/profiling-20261022/artifact-manifest.json)
records SHA256 hashes and byte sizes for the downloaded evidence.

The captured binary traces are 63,505 and 52,320 bytes. The
[complete profiling archive](../validation/profiling-evidence-20261022.tar.gz)
preserves both traces, both NEFF binaries, and the text evidence with its
artifact manifest. Their remote originals are retained under
`/home/ubuntu/nki-eval-runs/20261022/profiling-smoke/` on the test instance.

The command pattern below reproduces a capture using a copied NEFF on a
compatible Trainium host; substitute `reference` for `candidate` to profile
the baseline. Activation of the Neuron environment and the same core/target
settings are required. Use a new output directory to preserve previous runs.

```bash
. /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
export NEURON_PLATFORM_TARGET_OVERRIDE=trn1
export NEURON_RT_VISIBLE_CORES=0

timeout --signal=TERM --kill-after=10s 120s \
  /opt/aws/neuron/bin/neuron-profile capture \
  --neff candidate.neff --session-file candidate.ntff \
  --num-exec 3 --profile-nth-exec 3

timeout --signal=TERM --kill-after=10s 120s \
  /opt/aws/neuron/bin/neuron-profile view \
  --neff-path candidate.neff --session-file candidate_exec_3.ntff \
  --output-format summary-json > candidate-summary.json
```

The next profiling milestone is repeated device captures with explicitly
saved inputs, output validation outside the timed region, and consistent
capture settings across candidates and baselines. That would support a
device performance experiment distinct from this compatibility check.
