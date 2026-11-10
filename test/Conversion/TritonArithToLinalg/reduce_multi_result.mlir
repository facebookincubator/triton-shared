// Multi-operand / multi-result tt.reduce must lower to a SINGLE multi-result
// linalg.reduce that keeps both sub-reductions in one loop nest over the
// reduction axis.
//
// Two configs are exercised:
//   - @fused_sum_sumsq:    transpose-reduce-to-rank0=false -> the reduce
//                          keeps dimensions = [1].
//   - @fused_sum_sumsq_t:  transpose-reduce-to-rank0=true (pass default) ->
//                          BOTH inputs are transposed with the same
//                          permutation and the reduce runs on dimensions = [0].

// RUN: triton-shared-opt --triton-arith-to-linalg="transpose-reduce-to-rank0=false" %s | FileCheck %s --check-prefix=NOTRANS
// RUN: triton-shared-opt --triton-arith-to-linalg="transpose-reduce-to-rank0=true" %s | FileCheck %s --check-prefix=TRANS

module {
  tt.func public @fused_sum_sumsq(%x: tensor<32x512xf32>) -> (tensor<32xf32>, tensor<32xf32>) {
    %xx = arith.mulf %x, %x : tensor<32x512xf32>
    %red:2 = "tt.reduce"(%x, %xx) <{axis = 1 : i32}> ({
    ^bb0(%a0: f32, %a1: f32, %b0: f32, %b1: f32):
      %s0 = arith.addf %a0, %b0 : f32
      %s1 = arith.addf %a1, %b1 : f32
      tt.reduce.return %s0, %s1 : f32, f32
    }) : (tensor<32x512xf32>, tensor<32x512xf32>) -> (tensor<32xf32>, tensor<32xf32>)
    tt.return %red#0, %red#1 : tensor<32xf32>, tensor<32xf32>
  }
}

// One multi-result linalg.reduce, no transpose, reducing the innermost dim.
// NOTRANS-LABEL:   func.func @fused_sum_sumsq(
// NOTRANS:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// NOTRANS:           %[[XX:.*]] = linalg.generic
// NOTRANS:           %[[FILL0:.*]] = linalg.fill ins(%[[CST]] : f32) outs({{.*}}) -> tensor<32xf32>
// NOTRANS:           %[[FILL1:.*]] = linalg.fill ins(%[[CST]] : f32) outs({{.*}}) -> tensor<32xf32>
// NOTRANS:           %[[RED:.*]]:2 = linalg.reduce ins(%[[ARG0:.*]], %[[XX]] : tensor<32x512xf32>, tensor<32x512xf32>) outs(%[[FILL0]], %[[FILL1]] : tensor<32xf32>, tensor<32xf32>) dimensions = [1]
// NOTRANS:             (%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[INIT0:.*]]: f32, %[[INIT1:.*]]: f32) {
// NOTRANS:               %[[A0:.*]] = arith.addf %[[IN0]], %[[INIT0]] : f32
// NOTRANS:               %[[A1:.*]] = arith.addf %[[IN1]], %[[INIT1]] : f32
// NOTRANS:               linalg.yield %[[A0]], %[[A1]] : f32, f32
// NOTRANS:             }
// NOTRANS:           return %[[RED]]#0, %[[RED]]#1 : tensor<32xf32>, tensor<32xf32>

// Both inputs are transposed with the same permutation, then reduced on dim 0.
// TRANS-LABEL:   func.func @fused_sum_sumsq(
// TRANS:           %[[XX:.*]] = linalg.generic
// TRANS:           %[[T0:.*]] = linalg.transpose ins(%[[ARG0:.*]] : tensor<32x512xf32>) outs({{.*}} : tensor<512x32xf32>) permutation = [1, 0]
// TRANS:           %[[T1:.*]] = linalg.transpose ins(%[[XX]] : tensor<32x512xf32>) outs({{.*}} : tensor<512x32xf32>) permutation = [1, 0]
// TRANS:           %[[RED:.*]]:2 = linalg.reduce ins(%[[T0]], %[[T1]] : tensor<512x32xf32>, tensor<512x32xf32>) outs({{.*}}, {{.*}} : tensor<32xf32>, tensor<32xf32>) dimensions = [0]
// TRANS:             (%[[IN0:.*]]: f32, %[[IN1:.*]]: f32, %[[INIT0:.*]]: f32, %[[INIT1:.*]]: f32) {
// TRANS:               %[[A0:.*]] = arith.addf %[[IN0]], %[[INIT0]] : f32
// TRANS:               %[[A1:.*]] = arith.addf %[[IN1]], %[[INIT1]] : f32
// TRANS:               linalg.yield %[[A0]], %[[A1]] : f32, f32
// TRANS:             }
// TRANS:           return %[[RED]]#0, %[[RED]]#1 : tensor<32xf32>, tensor<32xf32>
