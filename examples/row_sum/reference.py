"""Sum each row, returning an FP32 column tensor even for BF16 inputs."""

import torch


def reference(a):
    return torch.sum(a, dim=1, keepdim=True, dtype=torch.float32)
