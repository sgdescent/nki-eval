# nki-eval

**Evaluate an AWS NKI kernel against a PyTorch reference, then measure performance if it passes.**

An independent, MIT-licensed evaluator inspired by the correctness-first verification in
[MusaCoder / MooreEval](https://arxiv.org/abs/2606.04847). Supply two Python entry points and a
JSON workload. Get a pass/fail verdict, numerical diagnostics, latency distributions, speedup,
and reproducible evaluation artifacts.

This is a single-machine evaluator. It does not reproduce MooreEval's distributed scheduler,
training system, or adversarial anti-cheating protections.

## What it checks and measures

- Multiple shapes, dtypes, input distributions, and reproducible random trials.
- Exact output structure, shape, and dtype; configurable floating-point tolerances; exact integers.
- Maximum absolute/relative error over finite pairs, mismatch counts, and first failing element.
- Input mutation detection for functional kernels. A failure in **any** case withholds all performance results.
- Candidate **and** compiled PyTorch baseline checked against the eager CPU reference.
- Warmup, alternating measurement order, median/p90/p99/mean/std latency, raw samples, and speedup.
- Fresh inputs and oracle checks for every warmup and timed pair, outside the timed interval.
- Serial repeated suites with source fingerprints and no aggregate score for failed suites.
- Optional effective TFLOP/s and GB/s from user-supplied work/traffic estimates.
- Separate worker processes, whole-evaluation timeout, compiler logs, source hashes, SDK metadata.
- Conservative NKI import/JIT checks that catch obvious PyTorch/NumPy fallbacks.

## Execution modes

| Backend | Executes | Performance result |
| --- | --- | --- |
| `neuron` (default) | NKI and PyTorch compiled with `torch_neuronx.trace` on Neuron | Host request latency, including transfers, versus compiled PyTorch on Neuron |
| `simulator` | Actual NKI CPU simulator versus eager PyTorch | None; simulator time is not accelerator time |
| `cpu` | Ordinary Python/PyTorch callables | Explicitly labeled CPU debug timings; no NKI validation |

**Validation status:** The initial addition and fused-add/ReLU examples passed real simulator
and Trainium1 validation on 21 October 2026: six cases at commit `0ebffec7`, using NKI 0.3 and
the Neuron PyTorch 2.9 environment. See [recorded measurements](validation/trainium-20261021.json).
CPU and adapter tests run in CI; real SDK tests remain opt-in. New examples and evaluator
changes require their own validation; a CPU test pass never establishes hardware compatibility.
Trainium2 and Trainium3 have not been validated by these runs.

The [latest PI update](docs/pi-update-20261022.md), [research plan](docs/research-plan.md),
[literature survey](docs/literature-survey.md), and
[reliability experiment](docs/reliability-results.md) describe the research direction and evidence.

On 22 October the revised evaluator passed all **18 cases across six families**, repeated three
times on Trainium1. Two deliberately wrong native kernels were rejected in both simulator and
hardware checks. The [full repeated results](validation/repeated-suite/suite.md) retain all runs.
The host-request geometric mean is **0.984×** versus compiled PyTorch: validation evidence, not
a substantial speedup claim.

## Quick start without AWS

Python 3.10+ on Linux or macOS:

```bash
git clone https://github.com/sgdescent/nki-eval.git
cd nki-eval
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

nki-eval --backend cpu \
  --reference examples/add/reference.py:reference \
  --candidate examples/add/cpu_candidate.py:kernel \
  --spec examples/add/workload.json \
  --json runs/add-cpu.json

pytest -q
```

The CPU example exercises the evaluator, not an NKI kernel. To check actual NKI code, install
the Neuron SDK in a supported environment and use the simulator or hardware commands below.

## Evaluate on Trainium

Activate your Neuron DLAMI's PyTorch environment and install this repository there. Follow
[the setup and profiling guide](docs/trainium.md) to keep the SDK's matching PyTorch packages.

```bash
nki-eval --backend neuron \
  --reference examples/add/reference.py:reference \
  --candidate examples/add/kernel.py:kernel \
  --spec examples/add/workload.json \
  --compiler-arg=--target=trn1 --compiler-arg=--auto-cast=none \
  --trials 5 --warmup 20 --iterations 200 --timeout 1200 \
  --json runs/add-trainium.json
```

For simulator-only validation in an environment with NKI installed:

```bash
NEURON_PLATFORM_TARGET_OVERRIDE=trn1 nki-eval --backend simulator \
  --reference examples/add/reference.py:reference \
  --candidate examples/add/kernel.py:kernel \
  --spec examples/add/workload.json
```

The simulator uses NumPy inputs, preserves BF16 through `ml_dtypes`, and enables
`NKI_PRECISE_FP=1`. It supports `nki.simulate` and includes an adapter for older
`neuronxcc.nki.simulate_kernel`; the bundled kernels use the modern namespace.

## Bring your own code

Define a PyTorch function with positional arguments:

```python
# reference.py
import torch


def reference(x, bias):
    return torch.relu(x + bias)
```

Put your equivalent `@nki.jit def kernel(x, bias): ...` in a separate file. The entry point is
`file.py:symbol`. A zero-argument `torch.nn.Module` class is also supported for the reference;
for constructor parameters, export a prebuilt module instance. Evaluation uses `.eval()` and
`torch.no_grad()`. Both sides receive the same positional tensor/scalar arguments. Model
weights needed by the candidate should be explicit inputs.

Specify the domain you want to test:

```json
{
  "name": "my_add_relu",
  "rtol": 0.0001,
  "atol": 0.00001,
  "equal_nan": false,
  "cases": [
    {
      "name": "tail_shape",
      "inputs": [
        {"shape": [17, 31], "dtype": "float32", "distribution": "normal"},
        {"shape": [17, 31], "dtype": "float32", "distribution": "uniform", "low": -1, "high": 1}
      ],
      "flops": 1054,
      "bytes_moved": 6324
    }
  ]
}
```

Distributions: `normal`, `uniform`, `zeros`, `ones`, `randint` (exclusive `high`). Floating
inputs may specify `scale`. Dtypes: `float16`, `bfloat16`, `float32`, `float64`, `int8`, `int16`,
`int32`, `int64`, `bool`; actual NKI support depends on your kernel and hardware. Use literal
JSON scalars in `inputs` for compile-time constants, e.g. `[{"shape": [128, 512]}, 0.00001]`.
Inputs are independent contiguous CPU tensors; aliasing, arbitrary strides, complex numbers,
training/backward, stochastic operations, and stateful/in-place contracts are outside v0.1.

Outputs may be tensors or matching nested tuples/lists/dictionaries of tensors; hardware
outputs must also be representable by `torch_neuronx.trace`. Tolerances default to `rtol=1e-4`,
`atol=1e-5` and apply as `abs(actual-reference) <= atol + rtol*abs(reference)`. Choose tolerances
for the operation and dtype. Shape/dtype mismatches never broadcast or silently cast. Matching
infinities are allowed; matching NaNs require `equal_nan=true`. Integer/bool outputs are exact.

Random trials are reproducible examples, **not a proof** for all inputs. Add explicit boundary,
zero, extreme-range, and shape cases relevant to your workload. The add example includes tails,
BF16, and zeros; `examples/fused_relu` demonstrates an operation fusion.

The catalog also includes row sum, stable softmax, LayerNorm without affine weights, and tiled
matrix multiplication. Read each example's README for supported shapes, output precision, and
layout constraints. In particular, matmul expects a contiguous pretransposed left operand;
layout preparation is outside that operator's measured contract.

For repeated experiments and PI-ready result tables, see [suite instructions](docs/benchmarking.md).

## Python API and agent loops

```python
from nki_eval import evaluate

result = evaluate(
    reference="reference.py:reference",
    candidate="kernel.py:kernel",
    spec="workload.json",
    backend="neuron",
    trials=10,
    compiler_args=["--target=trn1", "--auto-cast=none"],
    timeout=1200,
)
if result["status"] == "passed":
    print(result["performance"])
else:
    print(result.get("error"), result["cases"])
```

Use `correctness_only=True` (CLI: `--correctness-only`) to skip timing. Each call creates a
unique `runs/<id>/` containing `request.json`, `result.json`, `progress.json`, and `worker.log`.
Hardware runs also save per-case candidate/reference compiler directories and `model.pt` files.
Reports record source SHA-256 hashes, seeds, tolerances, compiler arguments, package versions,
and selected Neuron environment settings. Source dependencies and model weights are not bundled;
retain the matching source checkout and SDK environment for reproduction.

CLI exit codes: **0** passed, **1** failed evaluation, **2** invalid invocation.
Report statuses: `passed`, `incorrect`, `error`, `timeout`. A `passed` simulator report means
only that its configured simulated tests passed. See [report semantics](docs/results.md).

## Performance interpretation

Hardware timing covers a **serial host request** with CPU inputs and completed CPU outputs.
It includes Python dispatch, host/device transfers, execution, and output-readiness checks;
it excludes compilation, input generation, and input cloning. The baseline is compiled PyTorch
on Neuron, not CPU PyTorch. Small kernels can be dominated by overhead. Speedup is the ratio of
median baseline/candidate latency; the cross-case summary is their unweighted geometric mean.

The `fresh_inputs_checked_v1` protocol checks every warmup and measured output against eager
PyTorch and checks both implementations for input mutation. Those checks are outside the timer
but affect surrounding cache/workload state. Any failure removes all partial performance.
Do not pool these measurements with the original fixed-input, unchecked benchmark protocol.

Compilation durations are reported separately and may reflect warm compiler caches. Estimated
TFLOP/s and GB/s use supplied operation/byte counts and are not measured hardware utilization.
The [completed Trn1 profiling check](docs/profiling-results.md) records actual device captures
and their limits. For device timelines and further analysis, use the retained NEFF with a
[compatible profiler](docs/trainium.md#device-profiling). No invented device counters are reported.

## Trust boundary

Imported code has the privileges of the worker's OS user. A subprocess, timeout, and static
import check are **not a security sandbox**, and the checks are not robust against adversarial
reward hacking. Evaluate untrusted generated code on a disposable, unprivileged worker with
restricted network/filesystem access and no credentials. Do not use this release as a public
multi-tenant execution service. See [SECURITY.md](SECURITY.md).

## Development

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -q

# Explicit opt-in with the relevant SDK/device configured:
NKI_EVAL_RUN_SIMULATOR=1 pytest tests/test_optional_neuron.py -q
NKI_EVAL_RUN_NEURON=1 pytest tests/test_optional_neuron.py -q
```

Contributions: [CONTRIBUTING.md](CONTRIBUTING.md). License: [MIT](LICENSE).

## References

- [MusaCoder paper, §4.3 and Appendix I](https://arxiv.org/html/2606.04847v1): motivation for
  correctness-gated evaluation and structured execution feedback. No paper implementation is copied.
- [AWS NKI quickstart](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.0/nki/get-started/quickstart-implement-run-kernel.html)
- [NKI CPU simulator](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/guides/nki_simulator.html)
- [PyTorch NeuronX tracing API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/inference/api-torch-neuronx-trace.html)
