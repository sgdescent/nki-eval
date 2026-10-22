"""NKI 0.3+ tiled lhs_t.T @ rhs with explicit FP32 PSUM accumulation.

The caller supplies a contiguous transposed LHS with shape [K, M]. Any cost of
preparing that layout is outside this operator's contract and measured timing.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def kernel(lhs_t, rhs):
    assert len(lhs_t.shape) == 2 and len(rhs.shape) == 2
    contraction, rows = lhs_t.shape
    rhs_contraction, cols = rhs.shape
    assert contraction == rhs_contraction and contraction > 0 and rows > 0 and cols > 0
    assert lhs_t.dtype == nl.bfloat16 and rhs.dtype == nl.bfloat16
    out = nl.ndarray((rows, cols), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for m in nl.static_range((rows + 127) // 128):
        m0 = m * 128
        nm = min(128, rows - m0)
        for n in nl.static_range((cols + 511) // 512):
            n0 = n * 512
            nn = min(512, cols - n0)
            accum = nl.ndarray((nm, nn), dtype=nl.float32, buffer=nl.psum)
            for k in nl.static_range((contraction + 127) // 128):
                k0 = k * 128
                nk = min(128, contraction - k0)
                lhs_tile = nl.load(lhs_t[k0 : k0 + nk, m0 : m0 + nm])
                rhs_tile = nl.load(rhs[k0 : k0 + nk, n0 : n0 + nn])
                nisa.nc_matmul(accum, lhs_tile, rhs_tile, accumulate=(k > 0))
            result = nl.ndarray((nm, nn), dtype=nl.bfloat16, buffer=nl.sbuf)
            # PSUM cannot be stored directly to HBM or copied using nl.copy.
            nisa.tensor_copy(result, accum)
            nl.store(out[m0 : m0 + nm, n0 : n0 + nn], value=result)
    return out
