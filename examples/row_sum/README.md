# Row sum

Contract: contiguous `[rows, cols]` FP32 or BF16 input to `[rows, 1]` FP32 output.
Rows must be positive; `1 <= cols <= 2048`. The reduction uses FP32 arithmetic
even for BF16 inputs. The kernel tiles rows by 128 and takes exact-sized tail
slices, so neither padded values nor out-of-bounds loads participate in a sum.

The three cases exercise an aligned FP32 tile, row/width tails, and BF16 input.
The `1e-4` absolute tolerance accommodates different FP32 reduction orders near
cancellation. It is a workload policy, not an assertion of bitwise equivalence.

This is a simple correctness-oriented example, not an optimized reduction
library. CPU catalog tests validate the reference and workload only. Native
correctness and performance require a saved real-SDK evaluation report.

API source: [AWS NKI tensor_reduce documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/api/generated/nki.isa.tensor_reduce.html).
