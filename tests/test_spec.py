import pytest
import torch

from nki_eval.spec import inputs_for, validate


def workload(inp):
    return {"cases": [{"inputs": [inp]}]}


@pytest.mark.parametrize(
    "inp",
    [
        {"shape": [-1]},
        {"shape": [1.5]},
        {"shape": [True]},
        {"shape": [2], "dtype": "nope"},
        {"shape": [2], "distribution": "typo"},
        {"shape": [2], "dytpe": "float16"},
        {"shape": [2], "scale": float("inf")},
        {"shape": [2], "distribution": "uniform", "low": 1, "high": 0},
        {"shape": [2], "dtype": "int8", "high": 1000},
        {"shape": [2], "dtype": "bool", "high": 3},
        {"shape": [2], "dtype": "int64", "scale": 0.5},
        {"shape": [2], "dtype": "int64", "distribution": "normal"},
    ],
)
def test_invalid_inputs(inp):
    with pytest.raises(ValueError):
        validate(workload(inp))


def test_seed_reproducibility_and_independence():
    case = {"inputs": [{"shape": [16]}, {"shape": [16]}, 0.5]}
    a, b = inputs_for(case, 1), inputs_for(case, 1)
    assert torch.equal(a[0], b[0])
    assert not torch.equal(a[0], a[1])
    assert not torch.equal(a[0], inputs_for(case, 2)[0])
    assert a[2] == 0.5


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "int64", "bool"])
def test_dtypes(dtype):
    spec = validate(workload({"shape": [3, 4], "dtype": dtype}))
    assert inputs_for(spec["cases"][0], 0)[0].dtype == getattr(torch, dtype)


@pytest.mark.parametrize(
    "spec",
    [
        {"cases": []},
        {"cases": [{"inputs": [1]}]},
        {"cases": [{"inputs": [{"shape": [1]}]}], "rtol": -1},
        {"cases": [{"inputs": [{"shape": [1]}]}], "atol": float("nan")},
    ],
)
def test_invalid_workloads(spec):
    with pytest.raises(ValueError):
        validate(spec)
