// Multi-result `tt.reduce` where each result uses a DIFFERENT (but
// independently supported) reduction op — result 0 is a sum, result 1 is a
// product. This exercises that the per-result reductionOps[i] is threaded
// through correctly (distinct identity constants, distinct combine ops in
// the shared linalg.reduce region) rather than assuming all results share
// one op.

// RUN: triton-shared-opt --triton-arith-to-linalg="transpose-reduce-to-rank0=false" %s | FileCheck %s

module {
  tt.func public @mixed_sum_prod(%x: tensor<32x512xf32>, %y: tensor<32x512xf32>) -> (tensor<32xf32>, tensor<32xf32>) {
    %red:2 = "tt.reduce"(%x, %y) <{axis = 1 : i32}> ({
    ^bb0(%a0: f32, %a1: f32, %b0: f32, %b1: f32):
      %s0 = arith.addf %a0, %b0 : f32
      %s1 = arith.mulf %a1, %b1 : f32
      tt.reduce.return %s0, %s1 : f32, f32
    }) : (tensor<32x512xf32>, tensor<32x512xf32>) -> (tensor<32xf32>, tensor<32xf32>)
    tt.return %red#0, %red#1 : tensor<32xf32>, tensor<32xf32>
  }
}

// CHECK-LABEL:   func.func @mixed_sum_prod(
// CHECK:           %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[FILL0:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs({{.*}}) -> tensor<32xf32>
// CHECK:           %[[FILL1:.*]] = linalg.fill ins(%[[ONE]] : f32) outs({{.*}}) -> tensor<32xf32>
// CHECK:           %[[RED:.*]]:2 = linalg.reduce ins(%[[ARG0:.*]], %[[ARG1:.*]] : tensor<32x512xf32>, tensor<32x512xf32>) outs(%[[FILL0]], %[[FILL1]] : tensor<32xf32>, tensor<32xf32>) dimensions = [1]
// CHECK:             (%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[INIT0:.*]]: f32, %[[INIT1:.*]]: f32) {
// CHECK:               %[[A0:.*]] = arith.addf %[[IN0]], %[[INIT0]] : f32
// CHECK:               %[[A1:.*]] = arith.mulf %[[IN1]], %[[INIT1]] : f32
// CHECK:               linalg.yield %[[A0]], %[[A1]] : f32, f32
// CHECK:             }
// CHECK:           return %[[RED]]#0, %[[RED]]#1 : tensor<32xf32>, tensor<32xf32>
