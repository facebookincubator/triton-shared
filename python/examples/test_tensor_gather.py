# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver


@triton.jit
def gather2d_kernel(
    src_ptr, idx_ptr, out_ptr, AXIS: tl.constexpr, M: tl.constexpr, N: tl.constexpr
):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    offs = rows * N + cols

    src = tl.load(src_ptr + offs)
    idx = tl.load(idx_ptr + offs)
    out = tl.gather(src, idx, AXIS)
    tl.store(out_ptr + offs, out)


def gather2d(src, idx, axis):
    M, N = src.shape
    out = torch.empty_like(src)
    gather2d_kernel[1,](src, idx, out, AXIS=axis, M=M, N=N)
    return out


@pytest.mark.parametrize("axis", [0, 1])
def test_gather_2d(axis, device):
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    torch.manual_seed(0)
    M, N = 8, 16
    src = torch.randn(M, N, device=device)
    dim_size = M if axis == 0 else N
    idx = torch.randint(0, dim_size, (M, N), dtype=torch.int32, device=device)

    out = gather2d(src, idx, axis)
    ref = torch.gather(src, axis, idx.to(torch.int64))

    torch.testing.assert_close(out, ref)
