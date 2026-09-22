"""NKI 0.3+ fused add and ReLU for a single tile."""

import nki
import nki.language as nl


@nki.jit
def kernel(a, b):
    assert len(a.shape) == 2 and a.shape == b.shape
    assert a.shape[0] <= 128 and a.shape[1] <= 512
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    x, y = nl.load(a), nl.load(b)
    nl.store(out, value=nl.maximum(nl.add(x, y), 0.0))
    return out
