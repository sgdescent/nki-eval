"""SDK-specific execution. Simulator timings are deliberately never benchmarked."""

import copy
import importlib
import os
import time

import ml_dtypes
import numpy as np
import torch


class DependencyError(RuntimeError):
    pass


def import_nki():
    try:
        return importlib.import_module("nki")
    except ModuleNotFoundError as exc:
        if exc.name != "nki":
            raise
    try:
        return importlib.import_module("neuronxcc.nki")
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "Install the AWS Neuron SDK/NKI in this Python environment; see docs/trainium.md"
        ) from exc


def to_numpy(value):
    if not isinstance(value, torch.Tensor):
        return value
    if value.dtype == torch.bfloat16:
        return value.float().numpy().astype(ml_dtypes.bfloat16)
    return value.numpy().copy()


class CPUBackend:
    timing_scope = "cpu_debug_host_call"

    def prepare(self, candidate, reference, inputs, workdir, compiler_args):
        return candidate, copy.deepcopy(reference), {}


class SimulatorBackend:
    timing_scope = None

    def prepare(self, candidate, reference, inputs, workdir, compiler_args):
        os.environ["NKI_PRECISE_FP"] = "1"
        nki = import_nki()
        if hasattr(nki, "simulate"):
            simulated = nki.simulate(candidate)
        elif hasattr(nki, "simulate_kernel"):

            def simulated(*args):
                return nki.simulate_kernel(candidate, *args)
        else:
            raise DependencyError("Installed NKI has neither simulate nor simulate_kernel")

        def call(*args):
            arrays = tuple(to_numpy(x) for x in args)
            result = simulated(*arrays)
            # Detect input mutation even though the simulator received NumPy copies.
            for arg, array in zip(args, arrays):
                if isinstance(arg, torch.Tensor):
                    before = to_numpy(arg)
                    if not np.array_equal(before, array, equal_nan=True):
                        raise RuntimeError(
                            "Simulator kernel mutated an input; functional kernels only"
                        )
            return result

        return call, None, {"precise_fp": True}


class NeuronBackend:
    timing_scope = "neuron_host_request_including_transfers"

    def prepare(self, candidate, reference, inputs, workdir, compiler_args):
        try:
            import torch_neuronx
        except ImportError as exc:
            raise DependencyError(
                "Activate an AWS Neuron PyTorch environment with torch-neuronx"
            ) from exc
        if os.environ.get("NKI_SIMULATOR", "0") not in {"0", ""}:
            raise DependencyError("Unset NKI_SIMULATOR before hardware evaluation")
        positions = [i for i, x in enumerate(inputs) if isinstance(x, torch.Tensor)]
        tensors = tuple(inputs[i] for i in positions)

        class Bound(torch.nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.fn = fn

            def forward(self, *args):
                bound = list(inputs)
                for i, arg in zip(positions, args):
                    bound[i] = arg
                return self.fn(*bound)

        compiled = []
        metadata = {}
        for label, fn in (("candidate", candidate), ("reference", copy.deepcopy(reference))):
            folder = workdir / label
            folder.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            traced = torch_neuronx.trace(
                Bound(fn).eval(),
                tuple(x.clone() for x in tensors),
                compiler_workdir=str(folder),
                compiler_args=compiler_args,
            )
            metadata[f"{label}_compile_s"] = time.perf_counter() - start
            traced.save(str(folder / "model.pt"))
            metadata[f"{label}_artifacts"] = str(folder)

            def call(*args, _traced=traced):
                result = _traced(*(args[i] for i in positions))

                # Trace uses CPU inputs/outputs; disallow an unexpected async device tensor.
                def ready(value):
                    if isinstance(value, torch.Tensor):
                        if value.device.type != "cpu":
                            raise RuntimeError(
                                "Trace returned a non-CPU tensor; timing unsupported"
                            )
                        if value.numel():
                            value.reshape(-1)[0].item()
                    elif isinstance(value, dict):
                        for v in value.values():
                            ready(v)
                    elif isinstance(value, (tuple, list)):
                        for v in value:
                            ready(v)

                ready(result)
                return result

            compiled.append(call)
        return *compiled, metadata


BACKENDS = {"cpu": CPUBackend, "simulator": SimulatorBackend, "neuron": NeuronBackend}
