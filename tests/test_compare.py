import json

import ml_dtypes
import numpy as np
import pytest
import torch

from nki_eval.compare import compare


@pytest.mark.parametrize(
    "actual", [torch.ones(1), torch.ones(2, dtype=torch.float64), torch.zeros(2), (torch.ones(2),)]
)
def test_reject_shape_dtype_value_structure(actual):
    assert not compare(torch.ones(2), actual)["passed"]


def test_nested_outputs_and_numpy():
    ref = {"a": (torch.ones(2), [torch.zeros(1)])}
    out = {"a": (np.ones(2, dtype=np.float32), [np.zeros(1, dtype=np.float32)])}
    assert compare(ref, out)["passed"]
    assert not compare(ref, {"b": out["a"]})["passed"]
    assert not compare(ref, {"a": (out["a"][0], [])})["passed"]


def test_int64_is_not_compared_in_float():
    ref = torch.tensor([2**60, -(2**60)], dtype=torch.int64)
    out = torch.tensor([2**60 + 1, -(2**60)], dtype=torch.int64)
    result = compare(ref, out, rtol=1, atol=1e20)
    assert not result["passed"]
    assert result["tensors"][0]["mismatches"] == 1
    assert compare(ref, ref)["passed"]


def test_nonfinite_policy_and_json():
    ref = torch.tensor([float("nan"), float("inf"), -float("inf")])
    assert not compare(ref, ref)["passed"]
    assert compare(ref, ref, equal_nan=True)["passed"]
    out = torch.tensor([float("nan"), -float("inf"), 0])
    result = compare(ref, out)
    assert not result["passed"]
    json.dumps(result, allow_nan=False)


def test_bfloat16_preserved():
    ref = torch.tensor([0.1, 0.3], dtype=torch.bfloat16)
    array = ref.float().numpy().astype(ml_dtypes.bfloat16)
    assert compare(ref, array)["passed"]
    assert not compare(ref, array.astype(np.float32))["passed"]


def test_tolerance_is_relative_to_reference():
    assert compare(torch.tensor([100.0]), torch.tensor([101.0]), rtol=0.01, atol=0)["passed"]
    assert not compare(torch.tensor([100.0]), torch.tensor([101.1]), rtol=0.01, atol=0)["passed"]


def test_empty_tensor_allowed_empty_tree_rejected():
    assert compare(torch.empty(0), torch.empty(0))["passed"]
    assert not compare([], [])["passed"]
    assert not compare(None, None)["passed"]


def test_mismatch_has_actionable_location():
    result = compare(torch.ones(2, 3), torch.tensor([[1.0, 1.0, 1.0], [1.0, 2.0, 1.0]]))
    assert result["tensors"][0]["first_mismatch"]["flat_index"] == 4


def test_finite_error_overflow_still_serializes():
    result = compare(
        torch.tensor([1e308], dtype=torch.float64), torch.tensor([-1e308], dtype=torch.float64)
    )
    assert not result["passed"]
    json.dumps(result, allow_nan=False)
