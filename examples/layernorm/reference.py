"""No-affine LayerNorm over the last dimension with FP32 compute and output."""

import torch.nn.functional as functional


def reference(a, epsilon=1e-5):
    return functional.layer_norm(a.float(), (a.shape[1],), eps=epsilon)
