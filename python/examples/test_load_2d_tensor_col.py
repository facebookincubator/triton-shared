# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
# Licensed under the MIT license.

import torch

import triton
import triton.language as tl


"""

|-----|-----|-----|-----|
|     |     |     |     |
|-----|-----|-----|-----|
|     |     |     |     |
|-----|-----|-----|-----|

Each instance loads BLOCK_SIZE_COL columns
"""


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    input_desc = tl.make_tensor_descriptor(
        base=x_ptr,
        shape=[n_rows, n_cols],
        strides=[n_cols, 1],
        block_shape=[BLOCK_SIZE_ROW, BLOCK_SIZE_COL],
    )
    x = input_desc.load([0, pid0 * BLOCK_SIZE_COL])
    output_desc = tl.make_tensor_descriptor(
        base=y_ptr,
        shape=[n_rows, n_cols],
        strides=[n_cols, 1],
        block_shape=[BLOCK_SIZE_ROW, BLOCK_SIZE_COL],
    )
    output_desc.store([0, pid0 * BLOCK_SIZE_COL], x)


def test(device):
    n_rows = 4
    n_cols = 8
    x = torch.arange(0, n_rows * n_cols, 1, device=device, dtype=torch.float32).reshape(
        [n_rows, n_cols]
    )
    output = torch.full([n_rows, n_cols], -1, device=device, dtype=x.dtype)
    BLOCK_SIZE_ROW = n_rows
    BLOCK_SIZE_COL = 4

    grid = lambda meta: (n_cols // BLOCK_SIZE_COL,)

    kernel[grid](
        x,
        output,
        n_rows,
        n_cols,
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )

    torch.testing.assert_close(output, x, rtol=0.001, atol=1e-5)
