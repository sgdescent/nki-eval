# Trainium setup and validation

## Environment

Use an instance with a matching AWS Neuron driver, runtime, compiler,
and PyTorch environment. Start from the
[AWS Neuron setup documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/index.html)
or a Neuron DLAMI. Activate its existing PyTorch virtual environment before installation.
The hardware adapter targets `torch_neuronx.trace`; it does not currently implement the newer
TorchNeuron native eager/`torch.compile` path. Use a release environment that provides this API.

Check the environment:

```bash
neuron-ls
python -c 'import torch, torch_neuronx, nki; print(torch.__version__)'
```

Install without replacing the SDK-provided PyTorch build:

```bash
git clone https://github.com/sgdescent/nki-eval.git
cd nki-eval
python -m pip install -e . --no-deps
python -m pip install 'numpy>=1.24,<3' 'ml-dtypes>=0.5'
```

Keep NumPy compatible with the selected SDK's constraints; if it is already installed and
satisfies the evaluator, no upgrade is needed. Do not install an unrelated PyPI package named
`nki`; use the AWS SDK's NKI distribution. No SDK, instance, or paid job is provisioned by this repo.

Run the simulator first when available, then the hardware backend. The NKI examples use modern
high-level language APIs (NKI 0.3+). For older kernels using `neuronxcc.nki`, keep the SDK that
matches their source. Presence of a legacy API does not guarantee current compiler compatibility.
The 21 October 2026 integration run validated the original six example cases on Trainium1:
`trn1.2xlarge`, Ubuntu 24.04, Python 3.12.3, NKI `0.3.0+23928721754.g18aa1271`,
neuronx-cc `2.24.5133.0+58f8de22`, torch `2.9.1`, and
torch-neuronx `2.9.0.2.13.24727+8e870898`. Activate the DLAMI's full virtual environment;
calling its Python executable without putting its bin directory on PATH failed to locate
`libneuronpjrt-path`. This evidence does not validate arbitrary SDK combinations or Trn2/Trn3.

## Hardware evaluation

```bash
unset NKI_SIMULATOR
nki-eval --backend neuron \
  --reference examples/add/reference.py:reference \
  --candidate examples/add/kernel.py:kernel \
  --spec examples/add/workload.json \
  --compiler-arg=--target=trn1 --compiler-arg=--auto-cast=none \
  --warmup 20 --iterations 200 --timeout 1200 \
  --json runs/hardware-add.json
```

Use a target matching your instance and SDK; Trn2/Trn3 need separate qualification.
Choose the SDK's appropriate logical NeuronCore configuration;
additional compiler options can be passed by repeating `--compiler-arg`, e.g.
`--compiler-arg=--lnc=1` where supported. Both candidate and baseline receive the same options.
Use the runtime's `NEURON_RT_VISIBLE_CORES` setting to reserve compatible cores. Avoid competing
jobs on those cores during measurement. The evaluator does not schedule or lock shared devices.

Each shape is compiled separately. Both traced implementations are numerically checked against
the eager PyTorch oracle on fresh inputs. Unsupported reference operators or compiler failures
stop the evaluation with a diagnostic rather than silently switching to a CPU timing baseline.
Weights should be explicit arguments where possible. Data-dependent Python control flow has the
usual tracing limitations; add input cases that expose them and inspect baseline checks.

For an SDK qualification run:

```bash
python -m pip install pytest
NKI_EVAL_RUN_SIMULATOR=1 NKI_EVAL_RUN_NEURON=1 \
  pytest tests/test_optional_neuron.py -q
python -m pip freeze > runs/environment.txt
```

Archive the test output, instance type, SDK image version, and result directories. Real SDK tests
remain explicitly skipped in ordinary CI. No Trainium pass or performance claim is implied by CI.

## Device profiling

On the tested Trn1 environment, `neuron-profile 2.29.18` captured and exported
both candidate and baseline traces. Use the [tested commands and evidence](profiling-results.md)
for that SDK. The newer Neuron Explorer example below requires a compatible installation.

The report lists a per-case `preparation.candidate_artifacts` directory containing compiler
outputs. Find its NEFF:

```bash
find runs -name '*.neff'
```

With a compatible Neuron Explorer installation, replace the path below with that artifact:

```bash
neuron-explorer capture \
  -n runs/RUN_ID/case_0/candidate/graph.neff \
  -s runs/profile.ntff --profile-nth-exec=2
```

Inspect the trace using Neuron Explorer for device execution and engine/DMA behavior. This is a
separate profiling run; it does not automatically share the evaluator's seeded inputs. Do not
substitute its timing into the request-latency ratio without establishing equivalent inputs and
measurement scope. Artifact filenames can vary by compiler version; use the actual emitted NEFF.
See [AWS's profiling guide](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/nki/guides/use-neuron-profile.html).

The evaluator intentionally does not depend on `nki.benchmark` or `nki.profile`, whose availability
changed during the NKI migration. See the
[NKI release notes](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/release-notes/components/nki.html).
