// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

// Verifies that a 3D tt.dot (batched matmul) lowers to linalg.batch_matmul
// instead of linalg.matmul, which only supports 2D operands.
module {
  tt.func @kernel(
    %arg0: tensor<32x64x128x!tt.ptr<bf16>>, %arg1: tensor<32x128x256x!tt.ptr<bf16>>, %arg2: tensor<32x64x256x!tt.ptr<bf16>>, %arg3: tensor<32x64x256x!tt.ptr<bf16>>
  )
  {
    %0 = tt.load %arg0: tensor<32x64x128x!tt.ptr<bf16>>
    %1 = tt.load %arg1: tensor<32x128x256x!tt.ptr<bf16>>
    %2 = tt.load %arg2: tensor<32x64x256x!tt.ptr<bf16>>
    %3 = tt.dot %0, %1, %2 : tensor<32x64x128xbf16> * tensor<32x128x256xbf16> -> tensor<32x64x256xbf16>
    tt.store %arg3, %3 : tensor<32x64x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<32x64x128x!tt.ptr<bf16>>, %[[ARG1:.*]]: tensor<32x128x256x!tt.ptr<bf16>>, %[[ARG2:.*]]: tensor<32x64x256x!tt.ptr<bf16>>, %[[ARG3:.*]]: tensor<32x64x256x!tt.ptr<bf16>>,
// CHECK:           %[[LOAD_A:.*]] = tt.load %[[ARG0]] : tensor<32x64x128x!tt.ptr<bf16>>
// CHECK:           %[[LOAD_B:.*]] = tt.load %[[ARG1]] : tensor<32x128x256x!tt.ptr<bf16>>
// CHECK:           %[[LOAD_C:.*]] = tt.load %[[ARG2]] : tensor<32x64x256x!tt.ptr<bf16>>
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<32x64x256xbf16>
// CHECK:           %[[FILL:.*]] = linalg.fill {{.*}} outs(%[[EMPTY]] : tensor<32x64x256xbf16>) -> tensor<32x64x256xbf16>
// CHECK:           %[[BATCH_MATMUL:.*]] = linalg.batch_matmul ins(%[[LOAD_A]], %[[LOAD_B]] : tensor<32x64x128xbf16>, tensor<32x128x256xbf16>) outs(%[[FILL]] : tensor<32x64x256xbf16>) -> tensor<32x64x256xbf16>
// CHECK:           %[[RESULT:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[LOAD_C]], %[[BATCH_MATMUL]] : tensor<32x64x256xbf16>, tensor<32x64x256xbf16>)
// CHECK:           tt.store %[[ARG3]], %[[RESULT]] : tensor<32x64x256x!tt.ptr<bf16>>
// CHECK:           return
