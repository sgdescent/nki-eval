"""Validate example contracts on CPU; these tests do not execute NKI kernels."""

import json
from pathlib import Path

import pytest
import torch

from nki_eval.loading import lint_candidate, load
from nki_eval.spec import clone_inputs, inputs_for, validate

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
FAMILIES = ("row_sum", "softmax", "layernorm", "matmul")


def catalog_cases():
    cases = []
    for family in FAMILIES:
        spec = validate(json.loads((EXAMPLES / family / "workload.json").read_text()))
        cases.extend(
            pytest.param(family, case, id=f"{family}-{case['name']}") for case in spec["cases"]
        )
    return cases


@pytest.mark.parametrize("family,case", catalog_cases())
def test_catalog_reference_contract(family, case):
    inputs = inputs_for(case, seed=711)
    before = clone_inputs(inputs)
    reference = load(f"{EXAMPLES / family / 'reference.py'}:reference")
    with torch.no_grad():
        result = reference(*inputs)
    rows, cols = inputs[0].shape
    expected_shape = {
        "row_sum": (rows, 1),
        "softmax": (rows, cols),
        "layernorm": (rows, cols),
        "matmul": (cols, inputs[1].shape[1]) if family == "matmul" else (),
    }[family]
    assert result.shape == expected_shape
    assert result.dtype == (torch.bfloat16 if family == "matmul" else torch.float32)
    assert torch.isfinite(result).all()
    for actual, original in zip(inputs, before):
        if isinstance(actual, torch.Tensor):
            assert torch.equal(actual, original), "Reference mutated an input"

    if family == "softmax":
        assert torch.all(result >= 0)
        torch.testing.assert_close(result.sum(dim=1), torch.ones(rows))
    elif family == "layernorm":
        torch.testing.assert_close(result.mean(dim=1), torch.zeros(rows), atol=1e-6, rtol=0)
    elif family == "matmul":
        contraction, rows = inputs[0].shape
        cols = inputs[1].shape[1]
        assert case["flops"] == 2 * rows * contraction * cols
        # Algorithmic tensor bytes, not observed traffic or device bandwidth.
        assert case["bytes_moved"] == 2 * (rows * contraction + contraction * cols + rows * cols)


@pytest.mark.parametrize("family", FAMILIES)
def test_candidates_pass_static_import_check(family):
    path = EXAMPLES / family / "kernel.py"
    compile(path.read_text(), str(path), "exec")
    assert lint_candidate(f"{path}:kernel") == []


def test_sum_uses_fp32_output_for_bf16():
    reference = load(f"{EXAMPLES / 'row_sum' / 'reference.py'}:reference")
    values = torch.ones((3, 513), dtype=torch.bfloat16)
    # BF16 output would round 513 to 512; the declared FP32 contract preserves it.
    assert torch.equal(reference(values), torch.full((3, 1), 513.0))


def test_softmax_is_stable_under_large_constant_shift():
    reference = load(f"{EXAMPLES / 'softmax' / 'reference.py'}:reference")
    values = torch.tensor([[0.0, 1.0, 2.0], [-2.0, -1.0, 0.0]])
    torch.testing.assert_close(reference(values + 1000.0), reference(values))


def test_layernorm_constant_and_population_variance():
    reference = load(f"{EXAMPLES / 'layernorm' / 'reference.py'}:reference")
    assert torch.equal(reference(torch.ones((3, 17))), torch.zeros((3, 17)))
    values = torch.tensor([[1.0, 2.0, 3.0]])
    expected = torch.tensor([[-1.0, 0.0, 1.0]]) / (2.0 / 3.0 + 1e-5) ** 0.5
    torch.testing.assert_close(reference(values), expected)


def test_matmul_input_layout_is_pretransposed():
    reference = load(f"{EXAMPLES / 'matmul' / 'reference.py'}:reference")
    lhs_t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16)
    rhs = torch.eye(2, dtype=torch.bfloat16)
    assert torch.equal(reference(lhs_t, rhs), lhs_t.T)
