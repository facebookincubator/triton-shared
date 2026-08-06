# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
# Licensed under the MIT license.

import torch

import triton
import triton.language as tl


@triton.jit
def block_copy_kernel(a_ptr, b_ptr):
    rows = tl.arange(0, 2)
    cols = tl.arange(0, 2)
    offsets = rows[:, None] * 2 + cols[None, :] + 8
    a = tl.load(a_ptr + offsets)
    tl.store(b_ptr + rows[:, None] * 2 + cols[None, :], a)



def test(device):
    input = torch.arange(0, 16, device=device, dtype=torch.float32)
    output = torch.full((4,), -1, device=device, dtype=torch.float32)
    expected = torch.arange(8, 12, device=device)
    grid = lambda meta: (1,)

    block_copy_kernel[grid](input, output)
    torch.equal(expected, output)
