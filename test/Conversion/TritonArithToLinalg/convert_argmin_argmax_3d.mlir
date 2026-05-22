// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

// Test argmax on a 3D tensor reducing along the last axis (axis=2).
// Verifies that getTransposedValue moves the reduction axis to rank 0
// (permutation = [2, 0, 1]) and reduces along axis 0.

module {
  tt.func public @test_argmax_3d(%arg0: tensor<2x4x8xf32>) -> (tensor<2x4xf32>, tensor<2x4xi32>) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
    %3 = tt.broadcast %2 : tensor<1x1x8xi32> -> tensor<2x4x8xi32>
    %4:2 = "tt.reduce"(%arg0, %3) <{axis = 2 : i32}> ({
    ^bb0(%arg1: f32, %arg2: i32, %arg3: f32, %arg4: i32):
      %5 = arith.cmpf oeq, %arg1, %arg3 : f32
      %6 = arith.cmpi slt, %arg2, %arg4 : i32
      %7 = arith.andi %5, %6 : i1
      %8 = arith.cmpf ogt, %arg1, %arg3 : f32
      %9 = arith.ori %8, %7 : i1
      %10 = arith.select %9, %arg1, %arg3 : f32
      %11 = arith.select %9, %arg2, %arg4 : i32
      tt.reduce.return %10, %11 : f32, i32
    }) : (tensor<2x4x8xf32>, tensor<2x4x8xi32>) -> (tensor<2x4xf32>, tensor<2x4xi32>)
    tt.return %4#0, %4#1 : tensor<2x4xf32>, tensor<2x4xi32>
  }
}

// CHECK-LABEL:   func.func @test_argmax_3d(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<2x4x8xf32>,
// CHECK-DAG:       %[[VAL_INIT:.*]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       %[[IDX_INIT:.*]] = arith.constant -1 : i32
// CHECK:           %[[TRANSPOSED_VAL:.*]] = linalg.transpose ins(%[[INPUT]] : tensor<2x4x8xf32>) outs(%{{.*}} : tensor<8x2x4xf32>) permutation = [2, 0, 1]
// CHECK:           %[[TRANSPOSED_IDX:.*]] = linalg.transpose ins(%{{.*}} : tensor<2x4x8xi32>) outs(%{{.*}} : tensor<8x2x4xi32>) permutation = [2, 0, 1]
// CHECK:           linalg.reduce ins(%[[TRANSPOSED_VAL]], %[[TRANSPOSED_IDX]] : tensor<8x2x4xf32>, tensor<8x2x4xi32>) outs(%{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xi32>) dimensions = [0]
// CHECK:             arith.cmpf ogt,
// CHECK:             linalg.yield

// -----

// Test argmin on a 3D tensor reducing along the last axis (axis=2).

module {
  tt.func public @test_argmin_3d(%arg0: tensor<2x4x8xf32>) -> (tensor<2x4xf32>, tensor<2x4xi32>) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
    %3 = tt.broadcast %2 : tensor<1x1x8xi32> -> tensor<2x4x8xi32>
    %4:2 = "tt.reduce"(%arg0, %3) <{axis = 2 : i32}> ({
    ^bb0(%arg1: f32, %arg2: i32, %arg3: f32, %arg4: i32):
      %5 = arith.cmpf oeq, %arg1, %arg3 : f32
      %6 = arith.cmpi slt, %arg2, %arg4 : i32
      %7 = arith.andi %5, %6 : i1
      %8 = arith.cmpf olt, %arg1, %arg3 : f32
      %9 = arith.ori %8, %7 : i1
      %10 = arith.select %9, %arg1, %arg3 : f32
      %11 = arith.select %9, %arg2, %arg4 : i32
      tt.reduce.return %10, %11 : f32, i32
    }) : (tensor<2x4x8xf32>, tensor<2x4x8xi32>) -> (tensor<2x4xf32>, tensor<2x4xi32>)
    tt.return %4#0, %4#1 : tensor<2x4xf32>, tensor<2x4xi32>
  }
}

// CHECK-LABEL:   func.func @test_argmin_3d(
// CHECK-SAME:      %[[INPUT:.*]]: tensor<2x4x8xf32>,
// CHECK-DAG:       %[[VAL_INIT:.*]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       %[[IDX_INIT:.*]] = arith.constant -1 : i32
// CHECK:           %[[TRANSPOSED_VAL:.*]] = linalg.transpose ins(%[[INPUT]] : tensor<2x4x8xf32>) outs(%{{.*}} : tensor<8x2x4xf32>) permutation = [2, 0, 1]
// CHECK:           %[[TRANSPOSED_IDX:.*]] = linalg.transpose ins(%{{.*}} : tensor<2x4x8xi32>) outs(%{{.*}} : tensor<8x2x4xi32>) permutation = [2, 0, 1]
// CHECK:           linalg.reduce ins(%[[TRANSPOSED_VAL]], %[[TRANSPOSED_IDX]] : tensor<8x2x4xf32>, tensor<8x2x4xi32>) outs(%{{.*}}, %{{.*}} : tensor<2x4xf32>, tensor<2x4xi32>) dimensions = [0]
// CHECK:             arith.cmpf olt,
// CHECK:             linalg.yield
