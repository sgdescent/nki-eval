# Stable row softmax

Contract: contiguous `[rows, cols]` FP32 or BF16 input to FP32 output of the same
shape, normalized along the last dimension. Rows must be positive;
`1 <= cols <= 2048`. The full row fits in one SBUF tile. Rows are tiled by 128
with exact tail slices. This does not implement a multi-tile/online softmax.

The kernel subtracts each row's maximum before exponentiation and multiplies by
the reciprocal of the exponential sum. It uses explicit ISA row-vector
broadcasts: NKI 0.3 language binary operations do not support implicit
broadcasting, and hardware division is unsupported. Computation and output are
FP32, including for BF16 input.

The three cases cover aligned normal FP32 values, nonaligned dimensions with
logits in `[80, 100)` (which expose overflow in an unstabilized implementation),
and BF16 input. Tolerances are `rtol=1e-4`, `atol=1e-6`. These finite-input cases
do not define semantics for NaN or infinity.

CPU catalog tests validate the reference and workload only. Native correctness
and performance require a saved real-SDK evaluation report.

API sources: [AWS NKI tensor_scalar](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/api/generated/nki.isa.tensor_scalar.html)
and [NKI language limitations](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.31.1/release-notes/components/nki.html).
