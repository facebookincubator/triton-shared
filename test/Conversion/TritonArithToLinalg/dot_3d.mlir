// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @batch_dot(%a: tensor<2x4x8xf32>,
                            %b: tensor<2x8x16xf32>,
                            %c: tensor<2x4x16xf32>)
      -> tensor<2x4x16xf32> {
    %0 = tt.dot %a, %b, %c {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32}
        : tensor<2x4x8xf32> * tensor<2x8x16xf32> -> tensor<2x4x16xf32>
    tt.return %0 : tensor<2x4x16xf32>
  }
}

// CHECK-LABEL: func.func @batch_dot(
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<2x4x16xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
// CHECK: %[[BATCH_MATMUL:.*]] = linalg.batch_matmul ins(%{{.*}}, %{{.*}} : tensor<2x4x8xf32>, tensor<2x8x16xf32>) outs(%[[FILL]] : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
// CHECK: linalg.generic
// CHECK-SAME: ins(%{{.*}}, %[[BATCH_MATMUL]] : tensor<2x4x16xf32>, tensor<2x4x16xf32>)
