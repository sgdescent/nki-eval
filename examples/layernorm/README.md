# Row LayerNorm without affine parameters

Contract: contiguous `[rows, cols]` FP32 or BF16 input, plus a positive scalar
epsilon, to FP32 output of the same shape. Rows must be positive;
`1 <= cols <= 2048`. The reference is `torch.nn.functional.layer_norm` with
`weight=None` and `bias=None`. This example does not include learned affine
scale/bias, gradients, or arbitrary normalized dimensions.

The kernel computes the mean, then population variance from squared centered
values, and finally multiplies by `rsqrt(variance + epsilon)`. It uses FP32
intermediates/output and explicit ISA broadcasts. Rows are tiled by 128 with
exact tail slices. Epsilon is bound as a compile-time scalar by the evaluator.

The three cases cover an aligned FP32 input, constant input with row/width tails
(zero variance), and BF16 input using a second epsilon value. Tolerances are
`rtol=1e-4`, `atol=1e-5`. Extreme offsets and near-constant random inputs need
separate numerical-stability studies; they are not claimed as validated here.

CPU catalog tests validate the reference and workload only. Native correctness
and performance require a saved real-SDK evaluation report.

API sources: [AWS NKI tensor_reduce](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/api/generated/nki.isa.tensor_reduce.html)
and [tensor_scalar](https://awsdocs-neuron.readthedocs-hosted.com/en/v2.29.1/nki/api/generated/nki.isa.tensor_scalar.html).
