// RUN: triton-shared-opt --split-input-file --structured-to-memref %s | FileCheck %s

// Tests for the 1D circular-buffer split-pointer path (WRAP_1D).
//
// Pattern: ptr[i % N] — a 1D tts.make_tptr whose shape[0] is non-zero (set
// by PtrAnalysis::visitOperandRem) causes isSplitPtr()==true with
// parentShape.size()==1.  Before this fix that hit:
//   assert(parentShape.size() == 2 && "Only support split pointer for 2D tensors only")
//
// The fix (create1DCastOps) produces two contiguous memref chunks. The split
// SIZE math is done in element units (dividing by the stride, via the shared
// computeWrapSizes helper), so it also handles the non-unit-stride pattern
// ptr[stride * (i % N)] correctly:
//   xAddr   = startOffset % N          (start position, ADDRESS units)
//   xElem   = xAddr / stride           (start position, ELEMENT units)
//   elemN   = N / stride               (modulo period, in elements)
//   d1      = min(xElem + XBLOCK, elemN) - xElem  (elements before boundary)
//   d2      = XBLOCK - d1                          (elements after wrap, may 0)
// The 1D chunk offsets (unlike the 2D side-by-side case) are:
//   chunk1: reinterpret_cast base[xAddr], size=d1, stride=stride
//           (xAddr = startOffset % N correctly wraps a tile that starts a
//            whole period past the buffer, where raw startOffset would be OOB)
//   chunk2: reinterpret_cast base[0],     size=d2, stride=stride
//           (the period always restarts at flat address 0 in 1D)
// create1DCopies then assembles: dst[0..d1) <- chunk1, dst[d1..XBLOCK) <- chunk2.

// -----

// Unmasked 1D circular-buffer load.
// startOffset = 4 (static), XBLOCK = 8, modulo = %N (dynamic).

// CHECK-LABEL: tt.func public @wrap_1d_unmasked_load
module {
  tt.func public @wrap_1d_unmasked_load(%arg0: !tt.ptr<f32>, %N: index) -> tensor<8xf32> {
    %c4 = arith.constant 4 : index
    %ptr = tts.make_tptr %arg0 to sizes: [8], strides: [1], offsets: [%c4], shape: [%N], order: [] : <f32> to tensor<8x!tt.ptr<f32>>
    %result = "tts.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<8x!tt.ptr<f32>>) -> tensor<8xf32>
    tt.return %result : tensor<8xf32>
  }
}

// CHECK:     [[BASE:%.+]] = builtin.unrealized_conversion_cast {{%.+}} : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG: [[C4:%.+]] = arith.constant 4 : index
// CHECK-DAG: [[C8:%.+]] = arith.constant 8 : index
// CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
// CHECK:     [[XADDR:%.+]] = arith.remsi [[C4]], {{%.+}} : index
// CHECK:     [[XELEM:%.+]] = arith.divsi [[XADDR]], [[C1]] : index
// CHECK:     [[ELEMN:%.+]] = arith.divsi {{%.+}}, [[C1]] : index
// CHECK:     [[NEXT:%.+]] = arith.addi [[XELEM]], [[C8]] : index
// CHECK:     [[CLAMPED:%.+]] = arith.minsi [[NEXT]], [[ELEMN]] : index
// CHECK:     [[D1:%.+]] = arith.subi [[CLAMPED]], [[XELEM]] : index
// CHECK:     [[D2:%.+]] = arith.subi [[C8]], [[D1]] : index
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK:     [[CAST1:%.+]] = memref.reinterpret_cast [[BASE]] to offset: {{\[}}[[XADDR]]], sizes: {{\[}}[[D1]]], strides: {{\[}}[[C1]]] : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:     [[CAST2:%.+]] = memref.reinterpret_cast [[BASE]] to offset: {{\[}}[[C0]]], sizes: {{\[}}[[D2]]], strides: {{\[}}[[C1]]] : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:     {{%.+}} = builtin.unrealized_conversion_cast [[CAST1]], [[CAST2]] {{.*}} {wrap_1d}
// CHECK:     [[ALLOC:%.+]] = memref.alloc() : memref<8xf32>
// CHECK:     [[DIM1:%.+]] = memref.dim [[CAST1]], {{%.+}} : memref<?xf32, strided<[?], offset: ?>>
// CHECK:     [[DIM2:%.+]] = memref.dim [[CAST2]], {{%.+}} : memref<?xf32, strided<[?], offset: ?>>
// CHECK:     [[DST1:%.+]] = memref.subview [[ALLOC]]{{.*}} : memref<8xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:     [[DST2:%.+]] = memref.subview [[ALLOC]]{{.*}} : memref<8xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:     memref.copy [[CAST1]], [[DST1]]
// CHECK:     memref.copy [[CAST2]], [[DST2]]
// CHECK:     {{%.+}} = bufferization.to_tensor [[ALLOC]] restrict writable : memref<8xf32>

// -----

// Masked 1D circular-buffer load with static mask dim (6) and scalar other (0.0).
// The unrealized_conversion_cast is erased by rewriteMaskedLoad; the cast ops
// are referenced directly.

// CHECK-LABEL: tt.func public @wrap_1d_masked_load
module {
  tt.func public @wrap_1d_masked_load(%arg0: !tt.ptr<f32>, %N: index) -> tensor<8xf32> {
    %c4 = arith.constant 4 : index
    %cst = arith.constant 0.0 : f32
    %ptr = tts.make_tptr %arg0 to sizes: [8], strides: [1], offsets: [%c4], shape: [%N], order: [] : <f32> to tensor<8x!tt.ptr<f32>>
    %result = "tts.load"(%ptr, %cst) <{operandSegmentSizes = array<i32: 1, 0, 1>, static_mask_dims = array<i64: 6>}> : (tensor<8x!tt.ptr<f32>>, f32) -> tensor<8xf32>
    tt.return %result : tensor<8xf32>
  }
}

// CHECK:     [[BASE:%.+]] = builtin.unrealized_conversion_cast {{%.+}} : !tt.ptr<f32> to memref<*xf32>
// CHECK:     [[C4:%.+]] = arith.constant 4 : index
// CHECK:     [[CST:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: [[C8:%.+]] = arith.constant 8 : index
// CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
// CHECK:     [[XADDR:%.+]] = arith.remsi [[C4]], {{%.+}} : index
// CHECK:     [[XELEM:%.+]] = arith.divsi [[XADDR]], [[C1]] : index
// CHECK:     [[ELEMN:%.+]] = arith.divsi {{%.+}}, [[C1]] : index
// CHECK:     [[NEXT:%.+]] = arith.addi [[XELEM]], [[C8]] : index
// CHECK:     [[CLAMPED:%.+]] = arith.minsi [[NEXT]], [[ELEMN]] : index
// CHECK:     [[D1:%.+]] = arith.subi [[CLAMPED]], [[XELEM]] : index
// CHECK:     [[D2:%.+]] = arith.subi [[C8]], [[D1]] : index
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK:     [[CAST1:%.+]] = memref.reinterpret_cast [[BASE]] to offset: {{\[}}[[XADDR]]], sizes: {{\[}}[[D1]]], strides: {{\[}}[[C1]]] : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:     [[CAST2:%.+]] = memref.reinterpret_cast [[BASE]] to offset: {{\[}}[[C0]]], sizes: {{\[}}[[D2]]], strides: {{\[}}[[C1]]] : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:     [[ALLOC:%.+]] = memref.alloc() : memref<8xf32>
// CHECK:     scf.if
// CHECK:       linalg.fill ins([[CST]] : f32) outs([[ALLOC]] : memref<8xf32>)
// CHECK:     memref.copy
// CHECK:     memref.copy
// CHECK:     {{%.+}} = bufferization.to_tensor [[ALLOC]] restrict writable : memref<8xf32>

// -----

// 1D circular-buffer store.

// CHECK-LABEL: tt.func public @wrap_1d_store
module {
  tt.func public @wrap_1d_store(%arg0: !tt.ptr<f32>, %N: index, %val: tensor<8xf32>) {
    %c4 = arith.constant 4 : index
    %ptr = tts.make_tptr %arg0 to sizes: [8], strides: [1], offsets: [%c4], shape: [%N], order: [] : <f32> to tensor<8x!tt.ptr<f32>>
    "tts.store"(%ptr, %val) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64>}> : (tensor<8x!tt.ptr<f32>>, tensor<8xf32>) -> ()
    tt.return
  }
}

// CHECK:     [[BASE:%.+]] = builtin.unrealized_conversion_cast {{%.+}} : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG: [[C4:%.+]] = arith.constant 4 : index
// CHECK-DAG: [[C8:%.+]] = arith.constant 8 : index
// CHECK-DAG: [[C1:%.+]] = arith.constant 1 : index
// CHECK:     [[XADDR:%.+]] = arith.remsi [[C4]], {{%.+}} : index
// CHECK:     [[XELEM:%.+]] = arith.divsi [[XADDR]], [[C1]] : index
// CHECK:     [[ELEMN:%.+]] = arith.divsi {{%.+}}, [[C1]] : index
// CHECK:     [[NEXT:%.+]] = arith.addi [[XELEM]], [[C8]] : index
// CHECK:     [[CLAMPED:%.+]] = arith.minsi [[NEXT]], [[ELEMN]] : index
// CHECK:     [[D1:%.+]] = arith.subi [[CLAMPED]], [[XELEM]] : index
// CHECK:     [[D2:%.+]] = arith.subi [[C8]], [[D1]] : index
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK:     [[CAST1:%.+]] = memref.reinterpret_cast [[BASE]] to offset: {{\[}}[[XADDR]]], sizes: {{\[}}[[D1]]], strides: {{\[}}[[C1]]] : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:     [[CAST2:%.+]] = memref.reinterpret_cast [[BASE]] to offset: {{\[}}[[C0]]], sizes: {{\[}}[[D2]]], strides: {{\[}}[[C1]]] : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:     {{%.+}} = builtin.unrealized_conversion_cast [[CAST1]], [[CAST2]] {{.*}} {wrap_1d}
// CHECK:     [[DIM1:%.+]] = memref.dim [[CAST1]], {{%.+}} : memref<?xf32, strided<[?], offset: ?>>
// CHECK:     [[DIM2:%.+]] = memref.dim [[CAST2]], {{%.+}} : memref<?xf32, strided<[?], offset: ?>>
// CHECK:     [[SLICE1:%.+]] = tensor.extract_slice {{%.+}}[0] {{\[}}[[DIM1]]] [1]
// CHECK:     bufferization.materialize_in_destination [[SLICE1]] in writable [[CAST1]]
// CHECK:     [[SLICE2:%.+]] = tensor.extract_slice {{%.+}}{{\[}}[[DIM1]]] {{\[}}[[DIM2]]] [1]
// CHECK:     bufferization.materialize_in_destination [[SLICE2]] in writable [[CAST2]]

// -----

// Non-unit-stride 1D circular-buffer load: ptr[stride * (i % N)] with a
// constant stride of 3 (modulo period 96 in address units == 32 in elements).
// This is the pattern that reaches this path from a tiled rotary-embedding
// kernel (`cat(freqs, freqs)` lowered to `3*(x % 32)`). Before the stride-aware
// fix, create1DCastOps hardcoded stride 1 and compared the element block size
// directly against the address-unit modulo bound, collapsing the whole access
// to a plain contiguous stride-1 copy. The reinterpret_casts below must carry
// the real stride of 3, and the split SIZE math must divide by it (arith.divsi
// by the constant 3).

// CHECK-LABEL: tt.func public @wrap_1d_stride3_load
module {
  tt.func public @wrap_1d_stride3_load(%arg0: !tt.ptr<f32>) -> tensor<8xf32> {
    %c96 = arith.constant 96 : index
    %c0 = arith.constant 0 : index
    %ptr = tts.make_tptr %arg0 to sizes: [8], strides: [3], offsets: [%c0], shape: [%c96], order: [] : <f32> to tensor<8x!tt.ptr<f32>>
    %result = "tts.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<8x!tt.ptr<f32>>) -> tensor<8xf32>
    tt.return %result : tensor<8xf32>
  }
}

// CHECK:     [[BASE:%.+]] = builtin.unrealized_conversion_cast {{%.+}} : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG: [[C96:%.+]] = arith.constant 96 : index
// CHECK-DAG: [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[C8:%.+]] = arith.constant 8 : index
// CHECK-DAG: [[C3:%.+]] = arith.constant 3 : index
// CHECK:     [[XADDR:%.+]] = arith.remsi [[C0]], [[C96]] : index
// CHECK:     [[XELEM:%.+]] = arith.divsi [[XADDR]], [[C3]] : index
// CHECK:     [[ELEMN:%.+]] = arith.divsi [[C96]], [[C3]] : index
// CHECK:     [[NEXT:%.+]] = arith.addi [[XELEM]], [[C8]] : index
// CHECK:     [[CLAMPED:%.+]] = arith.minsi [[NEXT]], [[ELEMN]] : index
// CHECK:     [[D1:%.+]] = arith.subi [[CLAMPED]], [[XELEM]] : index
// CHECK:     [[D2:%.+]] = arith.subi [[C8]], [[D1]] : index
// CHECK-DAG: [[C0_2:%.+]] = arith.constant 0 : index
// CHECK:     [[CAST1:%.+]] = memref.reinterpret_cast [[BASE]] to offset: {{\[}}[[XADDR]]], sizes: {{\[}}[[D1]]], strides: {{\[}}[[C3]]] : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
// CHECK:     [[CAST2:%.+]] = memref.reinterpret_cast [[BASE]] to offset: {{\[}}[[C0_2]]], sizes: {{\[}}[[D2]]], strides: {{\[}}[[C3]]] : memref<*xf32> to memref<?xf32, strided<[?], offset: ?>>
