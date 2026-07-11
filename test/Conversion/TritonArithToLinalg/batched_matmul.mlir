// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0: tensor<32x64x128x!tt.ptr<f16>>, %arg1: tensor<32x128x256x!tt.ptr<f16>>, %arg2: tensor<32x64x256x!tt.ptr<f32>>, %arg3: tensor<32x64x256x!tt.ptr<f32>>
  )
  {
    %0 = tt.load %arg0: tensor<32x64x128x!tt.ptr<f16>>
    %1 = tt.load %arg1: tensor<32x128x256x!tt.ptr<f16>>
    %2 = tt.load %arg2: tensor<32x64x256x!tt.ptr<f32>>
    %3 = tt.dot %0, %1, %2 : tensor<32x64x128xf16> * tensor<32x128x256xf16> -> tensor<32x64x256xf32>
    tt.store %arg3, %3 : tensor<32x64x256x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x64x128x!tt.ptr<f16>>, %[[ARG1:.*]]: tensor<32x128x256x!tt.ptr<f16>>, %[[ARG2:.*]]: tensor<32x64x256x!tt.ptr<f32>>, %[[ARG3:.*]]: tensor<32x64x256x!tt.ptr<f32>>,
// CHECK:           %[[LOAD_A:.*]] = tt.load %[[ARG0]] : tensor<32x64x128x!tt.ptr<f16>>
// CHECK:           %[[LOAD_B:.*]] = tt.load %[[ARG1]] : tensor<32x128x256x!tt.ptr<f16>>
// CHECK:           %[[LOAD_C:.*]] = tt.load %[[ARG2]] : tensor<32x64x256x!tt.ptr<f32>>
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<32x64x256xf32>
// CHECK:           %[[FILL:.*]] = linalg.fill {{.*}} outs(%[[EMPTY]] : tensor<32x64x256xf32>) -> tensor<32x64x256xf32>
// CHECK:           %[[BATCH_MATMUL:.*]] = linalg.batch_matmul ins(%[[LOAD_A]], %[[LOAD_B]] : tensor<32x64x128xf16>, tensor<32x128x256xf16>) outs(%[[FILL]] : tensor<32x64x256xf32>) -> tensor<32x64x256xf32>
// CHECK:           %[[RESULT:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[LOAD_C]], %[[BATCH_MATMUL]] : tensor<32x64x256xf32>, tensor<32x64x256xf32>)
// CHECK:           tt.store %[[ARG3]], %[[RESULT]] : tensor<32x64x256x!tt.ptr<f32>>
// CHECK:           return
