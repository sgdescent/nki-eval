# Matrix multiplication with a pretransposed left input

Contract: contiguous BF16 `lhs_t[K, M]` and `rhs[K, N]` inputs to BF16 `[M, N]`
output, computing `lhs_t.T @ rhs`. All dimensions must be positive. Both the
candidate and compiled PyTorch reference receive the same already-prepared
input layout. **The cost of producing `lhs_t` from a conventional `[M, K]`
input is outside this operator's contract and benchmark.**

The kernel tiles M/K/N by 128/128/512, uses FP32 PSUM accumulation, and casts
to BF16 once per output tile. Exact-sized slices handle tails on every axis;
the first K tile overwrites PSUM and later tiles explicitly accumulate. The
PSUM result is copied into SBUF before storage to HBM.

The three cases cover a single tile, multiple tiles on all three axes, and
nonaligned dimensions including a contraction tail. The `rtol=0.01`,
`atol=0.001` policy allows BF16 rounding and accumulation-order differences;
it must be reported with results and is not a bitwise correctness claim.

FLOPs use the conventional `2*M*K*N` count. `bytes_moved` is the algorithmic
minimum `2*(M*K + K*N + M*N)` for BF16 input/output, not measured memory
traffic. This intentionally simple kernel reloads inputs across output tiles,
so effective GB/s from that count is not hardware bandwidth or utilization.

CPU catalog tests validate the reference and workload only. Native correctness
and performance require a saved real-SDK evaluation report. This baseline
implementation makes no claim to state-of-the-art throughput.

API source: [AWS NKI nc_matmul documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/api/generated/nki.isa.nc_matmul.html).
