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

Each instance loads BLOCK_SIZE_ROW * BLOCK_SIZE_COL
"""


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    stride_0,
    stride_1,
    BLOCK_SIZE_ROW: tl.constexpr,
    BLOCK_SIZE_COL: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)

    rows = tl.arange(0, BLOCK_SIZE_ROW)
    cols = tl.arange(0, BLOCK_SIZE_COL)
    offsets = (pid0 * BLOCK_SIZE_ROW + rows[:, None]) * stride_0 + (pid1 * BLOCK_SIZE_COL + cols[None, :]) * stride_1
    x = tl.load(x_ptr + offsets)
    x = (2 * x) + 1
    tl.store(y_ptr + offsets, x)


def test(device):
    n_rows = 512
    n_cols = 256
    x = torch.arange(0, n_rows * n_cols, 1, device=device, dtype=torch.float32).reshape(
        [n_rows, n_cols]
    )
    output = torch.full([n_rows, n_cols], -1, device=device, dtype=x.dtype)
    BLOCK_SIZE_ROW = 4
    BLOCK_SIZE_COL = 2

    grid = lambda meta: (n_rows // BLOCK_SIZE_ROW, n_cols // BLOCK_SIZE_COL)

    kernel[grid](
        x,
        output,
        n_rows,
        n_cols,
        x.stride(0),
        x.stride(1),
        BLOCK_SIZE_ROW=BLOCK_SIZE_ROW,
        BLOCK_SIZE_COL=BLOCK_SIZE_COL,
    )
    expected = (2 * x) + 1

    torch.testing.assert_close(output, expected, rtol=0.001, atol=1e-5)
