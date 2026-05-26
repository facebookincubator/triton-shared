// RUN: triton-shared-opt --triton-to-structured="run-prepass-only=true" --split-input-file %s | FileCheck %s

// The TritonToStructured prepass inserts tts.get_structured_state around
// scf.for iter_args whose type the converter treats as "structured". The
// converter must only treat pointer tensors and i32/i64 integer tensors as
// structured -- TT_IndexTensorLike in TritonStructuredDialect.td is defined
// as AnyTypeOf<[I32Tensor, I64Tensor]>, so wrapping a tensor with any other
// integer/index element type produces an op that fails verification.
//
// These tests pin the pass behavior on the boundary: the pointer iter_arg is
// expected to be wrapped (one tts.get_structured_state before the loop, one
// inside the loop body), while the non-i32/i64 integer tensor iter_arg passes
// through untouched.

// Test 1: tensor<...xi8> iter_arg must NOT be wrapped.
module {
  tt.func public @i8_tensor_iter_arg_not_wrapped(%arg0: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %cst_i8 = arith.constant dense<0> : tensor<4xi8>
    %cst_one_i8 = arith.constant dense<1> : tensor<4xi8>
    %offs = arith.constant dense<1> : tensor<4xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %0:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%pa = %p, %ia = %cst_i8) -> (tensor<4x!tt.ptr<f32>>, tensor<4xi8>) : i32 {
      %v = tt.load %pa : tensor<4x!tt.ptr<f32>>
      tt.store %pa, %v : tensor<4x!tt.ptr<f32>>
      %nextp = tt.addptr %pa, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %nexti = arith.addi %ia, %cst_one_i8 : tensor<4xi8>
      scf.yield %nextp, %nexti : tensor<4x!tt.ptr<f32>>, tensor<4xi8>
    }
    tt.return
  }
}

// Pointer iter_arg gets wrapped twice (once before the loop, once at the yield).
// The i8 iter_arg must not be wrapped, so the total count stays at 2.
// CHECK-LABEL: @i8_tensor_iter_arg_not_wrapped
// CHECK-COUNT-2: tts.get_structured_state
// CHECK-NOT:     tts.get_structured_state
// CHECK-NOT:     tensor<4xi8>{{.*}}tts.get_structured_state
// CHECK-NOT:     tts.get_structured_state{{.*}}tensor<4xi8>

// -----

// Test 2: tensor<...xindex> iter_arg must NOT be wrapped.
module {
  tt.func public @index_tensor_iter_arg_not_wrapped(%arg0: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %cst_idx = arith.constant dense<0> : tensor<4xindex>
    %cst_one_idx = arith.constant dense<1> : tensor<4xindex>
    %offs = arith.constant dense<1> : tensor<4xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %0:2 = scf.for %i = %c0 to %c4 step %c1 iter_args(%pa = %p, %ia = %cst_idx) -> (tensor<4x!tt.ptr<f32>>, tensor<4xindex>) : i32 {
      %v = tt.load %pa : tensor<4x!tt.ptr<f32>>
      tt.store %pa, %v : tensor<4x!tt.ptr<f32>>
      %nextp = tt.addptr %pa, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %nexti = arith.addi %ia, %cst_one_idx : tensor<4xindex>
      scf.yield %nextp, %nexti : tensor<4x!tt.ptr<f32>>, tensor<4xindex>
    }
    tt.return
  }
}

// CHECK-LABEL: @index_tensor_iter_arg_not_wrapped
// CHECK-COUNT-2: tts.get_structured_state
// CHECK-NOT:     tts.get_structured_state
// CHECK-NOT:     tensor<4xindex>{{.*}}tts.get_structured_state
// CHECK-NOT:     tts.get_structured_state{{.*}}tensor<4xindex>
