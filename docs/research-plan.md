# Research plan and this week’s demonstration

Prepared Thursday, 22 October 2026. The working target for this week's PI update is Friday, 23 October. This plan prioritizes a credible demonstration and a reproducible evidence package within that window. It does not assume a new performance record or completion of a large research program in two days.

## Objective and current position

Demonstrate that nki-eval can compare a PyTorch reference with an NKI implementation, reject representative incorrect candidates, and report trustworthy performance measurements on the available Trainium instance. The research direction is evaluator reliability and measurement validity. Existing NKI generation and optimization work makes a “first NKI benchmark” claim inappropriate; the [literature survey](literature-survey.md) explains the relevant comparisons.

| Area | Evidence status as of 22 October | Release gate |
| --- | --- | --- |
| Original evaluator | Verified on 21 October: two example kernels, six workload cases, simulator and Trainium paths | Preserve the original reports and exact source identity |
| Original performance | Verified host-request speedups of 0.878–1.004× against compiled PyTorch | Describe these as integration measurements; no meaningful speedup claim |
| Validation during benchmarking | Measured: false acceptance fell from 8 to 0 on 11 reviewed CPU contract mutants; both correct controls passed | Every warmup and measured output is checked; any failure removes performance results |
| Broader operators | Row sum, softmax, layer normalization, and matrix multiplication passed all 12 declared cases in simulator and on Trainium1 | Record simulator and hardware outcomes separately for every declared case |
| Repeated suites | Verified: all 18 evaluations and 54 case runs passed across three repeats; geometric-mean host ratio 0.984× | Repeated runs preserve source identity and never conceal failed or incomplete suites |
| Device profiling | Compatibility verified: one profiled BF16 matmul execution per implementation, 22.360086 µs NKI and 23.202204 µs baseline; profiler-default inputs without output validation | Repeat captures with saved inputs and output validation before drawing performance conclusions; [scope and evidence](profiling-results.md) |

Implementation, a passing local test, a simulator pass, and a hardware pass are distinct states. The [PI update](pi-update-20261022.md) links the saved reports for completed claims. The milestones below record release gates and the follow-up work, rather than infer native correctness from authored code or test counts.

## Parallel work and shared dependencies

Three tracks proceed together: evaluator checks and defective candidates; operator implementations and numerical references; literature, experimental design, and evidence packaging. Local tests and documentation can run while native compilation proceeds. One coordinator owns the frozen revision, hardware queue, result integration, and final review.

Accelerator measurements run serially on one visible NeuronCore of the existing `trn1.2xlarge`. No autoscaling or additional instances are required. The completed session used a two-hour automatic stop, and AWS confirmed the instance stopped after evidence collection. Future sessions should retain that bound and reserve the final 15 minutes for downloading artifacts and checking the stop configuration. A suite’s time limit must fit inside the remaining session. Model training and large paid generation sweeps are outside this week’s execution plan.

## Thursday session milestones

### 1. Freeze and validate the evaluator

Run the complete local suite after integrating the parallel changes. Verify late incorrect outputs, cached answers, warmup-only correctness, mutated inputs, failed baselines, timeouts, and report aggregation. Retain correct implementations as controls so stricter checks are not rewarded merely for rejecting everything.

Create a mutation matrix with one row per reviewed defective candidate: its intended defect, a demonstrated counterexample, the old and revised verdicts where reproducible, detection stage, backend, and report path. Exclude equivalent mutations from the denominator. Keep process tampering separate from numerical defects: the current worker is not a security sandbox.

**Release gate:** no known incorrect candidate in the included corpus receives a valid performance result; any remaining miss is a named limitation. Freeze the tested revision and entry-file fingerprints before hardware evaluation. The matrix measures this corpus, not all possible attacks.

### 2. Establish broader native coverage

Run simulator checks before hardware checks, beginning with one case per new operator family. Classify failures as unsupported SDK behavior, compilation, execution, or numerical disagreement. Repair the smallest reproducible issue, rerun the affected gate, and then expand to the family’s remaining cases. Preserve failed attempts alongside the final result.

**Release gate:** every scheduled case has an explicit outcome. The demo uses only independently checked native successes. If one family remains blocked, present its failure and the validated subset; do not relabel a CPU result as NKI execution. A reduction or normalization example adds more meaningful coverage than additional addition shapes alone.

### 3. Collect repeated host-request results

Use the frozen workload manifest and validated candidates. Run three complete suite repeats, with five initial correctness trials per case, 20 warmups, and 200 alternating candidate/baseline measurements. Use disjoint reproducible seed ranges and fresh input pairs. Keep oracle execution and validation outside the measured interval while documenting that they affect surrounding workload state.

Retain raw samples, source and spec identity, compiler options, environment, timing scope, and protocol version. Report each repeat, median speedup across repeats, and the descriptive repeat range. Compilation time remains separate. The aggregate weights cases equally and is withheld if the planned suite fails, remains incomplete, or mixes source identities or timing protocols.

**Release gate:** all advertised aggregate results are traceable to a complete consistent suite. A shortened smoke run can verify execution but does not satisfy the repeated-performance gate. Compare the new protocol with the original results only as different measurement protocols, not as an optimization gain.

### 4. Device profiling compatibility established

The installed `neuron-profile` tool successfully captured both saved executables for the BF16 matmul `multiple_tiles_bf16` case. Each capture executed its model three times and profiled the third execution, yielding one profile per model. The exported `total_time` field, verified as seconds using the installed tool's terminology, gave 22.360086 µs for NKI and 23.202204 µs for compiled PyTorch. The [profiling report](profiling-results.md) records versions, commands, source artifacts, units, limitations, and the public evidence archive.

This is a completed compatibility smoke test. The profiled models had passed the seeded evaluator, but these captures used profiler-default inputs without output checking. Device durations exclude transfers and host overhead, and must remain separate from host-request ratios. One profile per implementation does not establish a repeatable speedup.

**Next release gate:** use saved specified inputs, verify outputs outside the measured interval, and collect repeated captures under equivalent candidate and baseline procedures. Preserve every scheduled capture and report variation before interpreting device performance or utilization.

## Friday deliverables and demonstration

Reproduce the final frozen run, assemble the evidence package, and use a short walkthrough:

1. Supply the reference, candidate, and workload specification; show a native passing case and its numerical diagnostics.
2. Substitute a deliberately broken candidate; show its concrete failure and withheld timing.
3. Show a candidate that becomes wrong after initial validation; demonstrate detection during the benchmark lifecycle.
4. Open the broader coverage matrix and repeated timing table, including failures and timing scope.
5. Explain the remaining security and coverage limits and the next research experiment.

The PI package contains the updated design document, literature survey, tested source revision, reproduction instructions, mutation matrix, coverage matrix, and compact performance figure or table. Raw JSON and logs remain available behind every summary. Acceptance requires that another researcher can identify exactly what ran, recreate an included result, and distinguish completed evidence from planned extensions. A documented failure or negative result is useful evidence; an unsupported claim is not.

## Research after the demonstration

**Evaluator reliability:** freeze a second defect set that was not used to tune the checks. Compare incremental gates, report detection and false rejection by defect category, and measure validation overhead. Withheld defects and seeds are an experimental holdout, not a substitute for isolation from adversarial code.

**Measurement validity:** extend from three to five independent sessions, compare host and validated device measurements, and test whether kernel rankings change with shape and precision. Require consistent numerical contracts and equivalent baseline settings. Report uncertainty and all scheduled cases before any speedup headline.

**Feedback effectiveness:** after the verifier is stable, compare binary feedback, raw logs, and structured diagnostics on fixed initial candidates. Use a single generator/version, at most five repair attempts, matched budgets, and held-out final inputs. Run a small pilot before any paid sweep. Measure repair success, evaluations, elapsed time, and cost. Full RL training, distributed workers, Trainium2/3 claims, and end-to-end application acceleration remain deferred until the evidence justifies them.

## What can be automated

The agent can implement kernels and evaluator changes, run local and native tests, investigate failures, survey primary literature, operate the authorized AWS environment, preserve artifacts, generate plots and documents, and prepare reviewable GitHub changes. It can coordinate independent tracks while keeping performance runs isolated and bounded.

The Friday milestone is a defensible system demonstration and evidence package. Further compute sessions and model experiments should have an explicit budget and stop condition; messages to the PI require a separate instruction. Publication-level generality and performance remain research questions.
