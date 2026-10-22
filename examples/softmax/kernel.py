"""NKI 0.3+ stable row softmax; explicit ISA broadcasts and FP32 output."""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def kernel(a):
    assert len(a.shape) == 2
    rows, cols = a.shape
    assert rows > 0 and 0 < cols <= 2048
    assert a.dtype == nl.float32 or a.dtype == nl.bfloat16
    out = nl.ndarray(a.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    for r in nl.static_range((rows + 127) // 128):
        r0 = r * 128
        nr = min(128, rows - r0)
        values = nl.load(a[r0 : r0 + nr, :])
        maximum = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(maximum, op=nl.maximum, data=values, axis=1, keepdims=True)
        shifted = nl.ndarray((nr, cols), dtype=nl.float32, buffer=nl.sbuf)
        # nl.subtract does not support this row-vector broadcast in NKI 0.3.
        nisa.tensor_scalar(shifted, data=values, op0=nl.subtract, operand0=maximum)
        exponentials = nl.exp(shifted)
        total = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(total, op=nl.add, data=exponentials, axis=1, keepdims=True)
        inverse_total = nl.reciprocal(total)
        probabilities = nl.ndarray((nr, cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            probabilities, data=exponentials, op0=nl.multiply, operand0=inverse_total
        )
        nl.store(out[r0 : r0 + nr, :], value=probabilities)
    return out
