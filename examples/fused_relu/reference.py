import torch


def reference(a, b):
    return torch.relu(a + b)
