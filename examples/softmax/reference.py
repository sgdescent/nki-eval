"""Row softmax computed and returned in FP32, including for BF16 inputs."""

import torch


def reference(a):
    return torch.softmax(a, dim=1, dtype=torch.float32)
