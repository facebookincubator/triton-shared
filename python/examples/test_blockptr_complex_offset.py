# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
# Licensed under the MIT license.

import torch

import triton
import triton.language as tl


@triton.jit
def block_copy_kernel(a_ptr, b_ptr):
    a_desc = tl.make_tensor_descriptor(
        base=a_ptr + 8,
        shape=[2, 4],
        strides=[4, 1],
        block_shape=[2, 4],
    )
    a = a_desc.load([0, 0])
    b_desc = tl.make_tensor_descriptor(
        base=b_ptr,
        shape=[2, 4],
        strides=[4, 1],
        block_shape=[2, 4],
    )
    b_desc.store([0, 0], a)



def test(device):
    input = torch.arange(0, 32, device=device, dtype=torch.float32)
    output = torch.full((8,), -1, device=device, dtype=torch.float32)
    expected = torch.arange(8, 16, device=device)
    grid = lambda meta: (1,)

    block_copy_kernel[grid](input, output)
    torch.equal(expected, output)
