// RUN: triton-shared-opt --triton-to-structured="run-prepass-only=true" --split-input-file %s | FileCheck %s

// The TritonToStructured prepass inserts tts.get_structured_state around
// scf.for iter_args whose type the converter treats as structured. The
// converter must only treat pointer tensors and i32/i64 integer tensors as
// structured because TT_IndexTensorLike only accepts I32Tensor/I64Tensor.

module {
  tt.func public @i8_tensor_iter_arg_not_wrapped(%arg0: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %cst_i8 = arith.constant dense<0> : tensor<4xi8>
    %cst_one_i8 = arith.constant dense<1> : tensor<4xi8>
    %offs = arith.constant dense<1> : tensor<4xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %0:2 = scf.for %i = %c0 to %c4 step %c1
        iter_args(%pa = %p, %ia = %cst_i8)
        -> (tensor<4x!tt.ptr<f32>>, tensor<4xi8>) : i32 {
      %v = tt.load %pa : tensor<4x!tt.ptr<f32>>
      tt.store %pa, %v : tensor<4x!tt.ptr<f32>>
      %nextp = tt.addptr %pa, %offs : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %nexti = arith.addi %ia, %cst_one_i8 : tensor<4xi8>
      scf.yield %nextp, %nexti : tensor<4x!tt.ptr<f32>>, tensor<4xi8>
    }
    tt.return
  }
}

// CHECK-LABEL: @i8_tensor_iter_arg_not_wrapped
// CHECK-COUNT-2: tts.get_structured_state
// CHECK-NOT: tts.get_structured_state

// -----

module {
  tt.func public @index_tensor_iter_arg_not_wrapped(%arg0: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %cst_idx = arith.constant dense<0> : tensor<4xindex>
    %cst_one_idx = arith.constant dense<1> : tensor<4xindex>
    %offs = arith.constant dense<1> : tensor<4xi32>
    %p = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %0:2 = scf.for %i = %c0 to %c4 step %c1
        iter_args(%pa = %p, %ia = %cst_idx)
        -> (tensor<4x!tt.ptr<f32>>, tensor<4xindex>) : i32 {
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
// CHECK-NOT: tts.get_structured_state
