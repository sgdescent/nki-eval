"""Both implementations receive the same contiguous [K, M], [K, N] inputs."""


def reference(lhs_t, rhs):
    return lhs_t.transpose(0, 1) @ rhs
