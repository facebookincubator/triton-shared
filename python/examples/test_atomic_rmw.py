# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
# Licensed under the MIT license.

import torch
import triton
import triton.language as tl

@triton.jit
def atomic_fadd_no_mask_kernel(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Atomically add 1.0 to every element in a BLOCK_SIZE-wide slice."""
    offs = tl.arange(0, BLOCK_SIZE)
    ptrs = out_ptr + offs
    tl.atomic_add(ptrs, tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32))


@triton.jit
def atomic_fadd_with_mask_kernel(
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Atomically add 1.0 only to the first N elements (mask = offs < N)."""
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    ptrs = out_ptr + offs
    tl.atomic_add(ptrs, tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32), mask=mask)


@triton.jit
def atomic_addi_no_mask_kernel(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Atomically add 1 (integer) to every element in a BLOCK_SIZE-wide slice."""
    offs = tl.arange(0, BLOCK_SIZE)
    ptrs = out_ptr + offs
    tl.atomic_add(ptrs, tl.full([BLOCK_SIZE], 1, dtype=tl.int32))


@triton.jit
def atomic_xchg_kernel(
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Atomically exchange every element with its index value."""
    offs = tl.arange(0, BLOCK_SIZE)
    ptrs = out_ptr + offs
    tl.atomic_xchg(ptrs, offs.to(tl.float32))

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_atomic_fadd_no_mask(device):
    """tl.atomic_add (f32, no mask) accumulates correctly."""
    BLOCK_SIZE = 64
    out = torch.zeros(BLOCK_SIZE, dtype=torch.float32, device=device)
    grid = lambda _: (1,)
    # Run the kernel twice so each element ends up as 2.0.
    atomic_fadd_no_mask_kernel[grid](out, BLOCK_SIZE=BLOCK_SIZE)
    atomic_fadd_no_mask_kernel[grid](out, BLOCK_SIZE=BLOCK_SIZE)
    expected = torch.full((BLOCK_SIZE,), 2.0, dtype=torch.float32, device=device)
    torch.testing.assert_close(out, expected)


def test_atomic_fadd_with_mask(device):
    """tl.atomic_add (f32, with mask) only updates masked lanes."""
    BLOCK_SIZE = 64
    N = 32  # only first 32 elements are updated
    out = torch.zeros(BLOCK_SIZE, dtype=torch.float32, device=device)
    grid = lambda _: (1,)
    atomic_fadd_with_mask_kernel[grid](out, N=N, BLOCK_SIZE=BLOCK_SIZE)
    expected = torch.cat([
        torch.ones(N, dtype=torch.float32, device=device),
        torch.zeros(BLOCK_SIZE - N, dtype=torch.float32, device=device),
    ])
    torch.testing.assert_close(out, expected)


def test_atomic_addi_no_mask(device):
    """tl.atomic_add (i32, no mask) accumulates correctly."""
    BLOCK_SIZE = 64
    out = torch.zeros(BLOCK_SIZE, dtype=torch.int32, device=device)
    grid = lambda _: (1,)
    atomic_addi_no_mask_kernel[grid](out, BLOCK_SIZE=BLOCK_SIZE)
    atomic_addi_no_mask_kernel[grid](out, BLOCK_SIZE=BLOCK_SIZE)
    expected = torch.full((BLOCK_SIZE,), 2, dtype=torch.int32, device=device)
    torch.testing.assert_close(out, expected)


def test_atomic_xchg(device):
    """tl.atomic_xchg replaces each element with its index."""
    BLOCK_SIZE = 64
    out = torch.full((BLOCK_SIZE,), -1.0, dtype=torch.float32, device=device)
    grid = lambda _: (1,)
    atomic_xchg_kernel[grid](out, BLOCK_SIZE=BLOCK_SIZE)
    expected = torch.arange(BLOCK_SIZE, dtype=torch.float32, device=device)
    torch.testing.assert_close(out, expected)
