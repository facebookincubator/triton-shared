// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s --check-prefixes=ALLOC
// RUN: triton-shared-opt -triton-arith-to-linalg="use-alloc-tensor=false" %s | FileCheck %s --check-prefixes=EMPTY

// EMPTY-LABEL: func.func @reduce_sum
// EMPTY: tensor.empty() : tensor<f32>
// EMPTY: tensor.insert
// EMPTY: linalg.reduce
// ALLOC-LABEL: func.func @reduce_sum
// ALLOC: bufferization.alloc_tensor() : tensor<f32>
// ALLOC: tensor.insert
// ALLOC: linalg.reduce
module {
  tt.func @reduce_sum(%arg0: tensor<8xf32>) -> f32 {
    %0 = "tt.reduce"(%arg0) ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.addf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) {axis = 0 : i32} : (tensor<8xf32>) -> f32
    tt.return %0 : f32
  }
}
