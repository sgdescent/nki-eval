"""NKI 0.3+ row reduction with FP32 accumulation and row-tail support."""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def kernel(a):
    assert len(a.shape) == 2
    rows, cols = a.shape
    assert rows > 0 and 0 < cols <= 2048
    assert a.dtype == nl.float32 or a.dtype == nl.bfloat16
    out = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    for r in nl.static_range((rows + 127) // 128):
        r0 = r * 128
        nr = min(128, rows - r0)
        values = nl.load(a[r0 : r0 + nr, :])
        total = nl.ndarray((nr, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(total, op=nl.add, data=values, axis=1, keepdims=True)
        nl.store(out[r0 : r0 + nr, :], value=total)
    return out
