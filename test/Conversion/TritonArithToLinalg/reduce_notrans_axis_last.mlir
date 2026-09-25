// Single-result `tt.reduce` on the innermost axis under
// transpose-reduce-to-rank0=false: no transpose is applied and
// linalg.reduce runs directly on the original (non-transposed) axis. This
// locks in the current behavior (rather than the legacy transpose-to-
// rank0-minus-1 lowering this pass used before multi-result support was
// added) so a future refactor of the transpose logic can't silently regress
// it. See reduce_multi_result.mlir for the same axis under
// transpose-reduce-to-rank0=true.

// RUN: triton-shared-opt --triton-arith-to-linalg="transpose-reduce-to-rank0=false" %s | FileCheck %s

module {
  tt.func public @sum_axis_last(%x: tensor<32x512xf32>) -> tensor<32xf32> {
    %red = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a0: f32, %b0: f32):
      %s0 = arith.addf %a0, %b0 : f32
      tt.reduce.return %s0 : f32
    }) : (tensor<32x512xf32>) -> tensor<32xf32>
    tt.return %red : tensor<32xf32>
  }
}

// CHECK-LABEL:   func.func @sum_axis_last(
// CHECK:           %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NOT:       linalg.transpose
// CHECK:           %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs({{.*}}) -> tensor<32xf32>
// CHECK:           %[[RED:.*]] = linalg.reduce ins(%[[ARG0:.*]] : tensor<32x512xf32>) outs(%[[FILL]] : tensor<32xf32>) dimensions = [1]
// CHECK:             (%[[IN:.*]]: f32, %[[INIT:.*]]: f32) {
// CHECK:               %[[A:.*]] = arith.addf %[[IN]], %[[INIT]] : f32
// CHECK:               linalg.yield %[[A]] : f32
// CHECK:             }
// CHECK:           return %[[RED]] : tensor<32xf32>
