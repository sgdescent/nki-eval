"""NKI 0.3+ row LayerNorm without affine weights; FP32 compute and output."""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def kernel(a, epsilon=1e-5):
    assert len(a.shape) == 2
    rows, cols = a.shape
    assert rows > 0 and 0 < cols <= 2048 and epsilon > 0
    assert a.dtype == nl.float32 or a.dtype == nl.bfloat16
    out = nl.ndarray(a.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    for r in nl.static_range((rows + 127) // 128):
        r0 = r * 128
        nr = min(128, rows - r0)
        values = nl.load(a[r0 : r0 + nr, :])
        total = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(total, op=nl.add, data=values, axis=1, keepdims=True)
        mean = nl.multiply(total, 1.0 / cols)
        centered = nl.ndarray((nr, cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(centered, data=values, op0=nl.subtract, operand0=mean)
        squared = nl.multiply(centered, centered)
        squared_sum = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(squared_sum, op=nl.add, data=squared, axis=1, keepdims=True)
        # Population variance (divide by N), matching torch layer_norm.
        variance = nl.multiply(squared_sum, 1.0 / cols)
        inverse_std = nl.rsqrt(nl.add(variance, epsilon))
        normalized = nl.ndarray((nr, cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(normalized, data=centered, op0=nl.multiply, operand0=inverse_std)
        nl.store(out[r0 : r0 + nr, :], value=normalized)
    return out
