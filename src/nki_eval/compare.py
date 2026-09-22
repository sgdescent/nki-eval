"""Output-tree comparison without broadcasting or lossy integer equality checks."""

import math

import numpy as np
import torch


def as_tensor(value):
    if isinstance(value, np.ndarray):
        # NumPy's ml_dtypes bfloat16 is not accepted by torch.from_numpy.
        if str(value.dtype) == "bfloat16":
            return torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
        return torch.from_numpy(np.array(value, copy=True))
    return value


def compare(expected, actual, *, rtol=1e-4, atol=1e-5, equal_nan=False):
    leaves = []
    problems = []

    def walk(ref, out, path):
        ref, out = as_tensor(ref), as_tensor(out)
        if isinstance(ref, torch.Tensor):
            if not isinstance(out, torch.Tensor):
                problems.append(f"{path}: expected a tensor, got {type(out).__name__}")
                return
            if ref.shape != out.shape or ref.dtype != out.dtype:
                problems.append(
                    f"{path}: expected {list(ref.shape)} {ref.dtype}, "
                    f"got {list(out.shape)} {out.dtype}"
                )
                return
            ref, out = ref.detach().cpu(), out.detach().cpu()
            if ref.is_complex():
                problems.append(f"{path}: complex outputs are not supported")
                return
            if ref.is_floating_point():
                a, b = ref.double(), out.double()
                close = torch.isclose(b, a, rtol=rtol, atol=atol, equal_nan=equal_nan)
                finite = torch.isfinite(a) & torch.isfinite(b)
                abs_error = (a[finite] - b[finite]).abs()
                rel_error = abs_error / a[finite].abs().clamp_min(torch.finfo(torch.float64).tiny)
                max_abs = abs_error.max().item() if abs_error.numel() else 0.0
                max_rel = rel_error.max().item() if rel_error.numel() else 0.0
            else:
                close = out == ref  # Keep int64 exact, including values above 2**53.
                max_abs = max_rel = None
            mismatches = int((~close).sum())
            item = {
                "path": path,
                "shape": list(ref.shape),
                "dtype": str(ref.dtype),
                "elements": ref.numel(),
                "mismatches": mismatches,
                "max_abs_error_finite": max_abs
                if max_abs is None or math.isfinite(max_abs)
                else None,
                "max_rel_error_finite": max_rel
                if max_rel is None or math.isfinite(max_rel)
                else None,
            }
            if mismatches:
                flat = int((~close).reshape(-1).nonzero()[0])
                # Strings keep NaN/Inf and large integer diagnostics valid JSON.
                item["first_mismatch"] = {
                    "flat_index": flat,
                    "expected": str(ref.reshape(-1)[flat].item()),
                    "actual": str(out.reshape(-1)[flat].item()),
                }
            leaves.append(item)
        elif isinstance(ref, dict):
            if not isinstance(out, dict) or ref.keys() != out.keys():
                problems.append(f"{path}: dictionary keys differ")
                return
            for key in ref:
                walk(ref[key], out[key], f"{path}.{key}")
        elif isinstance(ref, (list, tuple)):
            if type(ref) is not type(out) or len(ref) != len(out):
                problems.append(f"{path}: sequence type or length differs")
                return
            for i, (a, b) in enumerate(zip(ref, out)):
                walk(a, b, f"{path}[{i}]")
        else:
            problems.append(f"{path}: outputs must be tensors or trees of tensors")

    walk(expected, actual, "output")
    if not leaves and not problems:
        problems.append("Output tree contains no tensors")
    return {
        "passed": not problems and all(x["mismatches"] == 0 for x in leaves),
        "problems": problems,
        "tensors": leaves,
    }
