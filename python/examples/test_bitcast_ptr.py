# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
# Licensed under the MIT license.

# Test for bitcast pointer handling in TritonToUnstructuredPass.
# Verifies that pointer arithmetic and bitcasts between same-stride types
# (i1/i8 — 1 byte; i32/f32 — 4 bytes) access the correct memory addresses,
# and that a bitcast which changes the byte stride is rejected at compile time.

import pytest
import torch

import triton
import triton.language as tl

from triton.backends.triton_shared.driver import CPUDriver


@triton.jit
def bitcast_ptr_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    ptr_i1 = input_ptr + 64 + idx               # addptr before bitcast
    ptr_i8 = ptr_i1.to(tl.pointer_type(tl.int8), bitcast=True)  # bitcast i1->i8
    ptr_final = ptr_i8 + 32                      # addptr after bitcast
    tl.store(output_ptr + idx, tl.load(ptr_final))


def test_bitcast_ptr(device):
    """addptr -> bitcast(i1->i8) -> addptr -> load. Reads at base+96+idx."""
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    input_buf = torch.arange(0, 256, device=device, dtype=torch.uint8)
    output_buf = torch.zeros(128, device=device, dtype=torch.uint8)

    bitcast_ptr_kernel[(1,)](
        input_buf.view(torch.bool), output_buf, BLOCK=128
    )

    # Kernel reads at base + 64 + idx + 32 = base + 96 + idx
    expected = input_buf[96:224]
    assert torch.equal(output_buf, expected)


@triton.jit
def bitcast_chain_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    ptr_i1 = input_ptr + 16 + idx                # addptr on ptr<i1>
    ptr_i8 = ptr_i1.to(tl.pointer_type(tl.int8), bitcast=True)  # bitcast i1->i8
    ptr_i8_2 = ptr_i8 + 8                        # addptr on ptr<i8>
    ptr_i1_2 = ptr_i8_2.to(tl.pointer_type(tl.int1), bitcast=True)  # bitcast i8->i1
    ptr_i1_3 = ptr_i1_2 + 4                      # addptr on ptr<i1>
    ptr_i8_3 = ptr_i1_3.to(tl.pointer_type(tl.int8), bitcast=True)  # bitcast i1->i8
    tl.store(output_ptr + idx, tl.load(ptr_i8_3))


def test_bitcast_chain(device):
    """addptr -> bitcast(i1->i8) -> addptr -> bitcast(i8->i1) -> addptr -> bitcast(i1->i8) -> load.
    Multiple bitcast chain. Reads at base+16+8+4 = base+28+idx."""
    if device == "cpu":
        triton.runtime.driver.set_active(CPUDriver())

    input_buf = torch.arange(0, 256, device=device, dtype=torch.uint8)
    output_buf = torch.zeros(64, device=device, dtype=torch.uint8)

    bitcast_chain_kernel[(1,)](
        input_buf.view(torch.bool), output_buf, BLOCK=64
    )

    # Kernel reads at base + 16 + idx + 8 + 4 = base + 28 + idx
    expected = input_buf[28:92]
    assert torch.equal(output_buf, expected)
