import sys
import types

import numpy as np
import pytest
import torch

from nki_eval.backends import NeuronBackend, SimulatorBackend, to_numpy
from nki_eval.compare import compare
from nki_eval.loading import lint_candidate


def test_simulator_contract(monkeypatch, tmp_path):
    def simulate(fn):
        def call(*args):
            assert all(isinstance(x, np.ndarray) for x in args)
            return fn(*args)

        return call

    monkeypatch.setitem(sys.modules, "nki", types.SimpleNamespace(simulate=simulate))
    fn, baseline, metadata = SimulatorBackend().prepare(np.add, None, (), tmp_path, [])
    inputs = (torch.ones(2), torch.ones(2))
    assert compare(torch.full((2,), 2.0), fn(*inputs))["passed"]
    assert baseline is None and metadata["precise_fp"]
    assert str(to_numpy(torch.ones(1, dtype=torch.bfloat16)).dtype) == "bfloat16"


def test_simulator_mutation(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "nki", types.SimpleNamespace(simulate=lambda fn: fn))

    def mutate(a):
        a[:] = 0
        return a

    fn, _, _ = SimulatorBackend().prepare(mutate, None, (), tmp_path, [])
    with pytest.raises(RuntimeError, match="mutated"):
        fn(torch.ones(2))


def test_neuron_trace_contract_scalar_binding_and_artifacts(monkeypatch, tmp_path):
    calls = []

    class Traced:
        def __init__(self, model):
            self.model = model

        def __call__(self, *args):
            return self.model(*args)

        def save(self, path):
            from pathlib import Path

            Path(path).write_text("fake-sdk-test-artifact")

    def trace(model, inputs, **kwargs):
        assert all(isinstance(x, torch.Tensor) for x in inputs)
        calls.append(kwargs)
        return Traced(model)

    monkeypatch.setitem(sys.modules, "torch_neuronx", types.SimpleNamespace(trace=trace))
    monkeypatch.delenv("NKI_SIMULATOR", raising=False)

    def function(x, scale):
        return x * scale

    candidate, reference, meta = NeuronBackend().prepare(
        function, function, (torch.ones(3), 2), tmp_path, ["--target=trn2"]
    )
    assert torch.equal(candidate(torch.ones(3), 2), torch.full((3,), 2.0))
    assert torch.equal(reference(torch.ones(3), 2), torch.full((3,), 2.0))
    assert len(calls) == 2 and calls[0]["compiler_args"] == ["--target=trn2"]
    assert (tmp_path / "candidate/model.pt").is_file()
    assert meta["reference_compile_s"] >= 0


def test_candidate_policy_aliases(tmp_path):
    path = tmp_path / "kernel.py"
    path.write_text(
        "from nki import jit as compile_kernel\n@compile_kernel\ndef kernel(a): return a"
    )
    assert lint_candidate(str(path) + ":kernel") == []
    path.write_text("from torch import add as plus\ndef kernel(a, b): return plus(a, b)")
    assert lint_candidate(str(path) + ":kernel")
