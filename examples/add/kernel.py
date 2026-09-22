"""NKI 0.3+ tiled addition, including non-tile-aligned shapes."""

import nki
import nki.language as nl


@nki.jit
def kernel(a, b):
    assert len(a.shape) == 2 and a.shape == b.shape
    rows, cols = a.shape
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    for r in nl.static_range((rows + 127) // 128):
        for c in nl.static_range((cols + 511) // 512):
            r0, c0 = r * 128, c * 512
            nr, nc = min(128, rows - r0), min(512, cols - c0)
            x = nl.load(a[r0 : r0 + nr, c0 : c0 + nc])
            y = nl.load(b[r0 : r0 + nr, c0 : c0 + nc])
            nl.store(out[r0 : r0 + nr, c0 : c0 + nc], value=nl.add(x, y))
    return out
