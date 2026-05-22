// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @argmin_i64(%arg0: tensor<4x8xi64>, %arg1: tensor<4x8xi32>) -> (tensor<8xi64>, tensor<8xi32>) {
    %0:2 = "tt.reduce"(%arg0, %arg1) <{axis = 0 : i32}> ({
    ^bb0(%arg2: i64, %arg3: i32, %arg4: i64, %arg5: i32):
      %1 = arith.cmpi eq, %arg2, %arg4 : i64
      %2 = arith.cmpi slt, %arg3, %arg5 : i32
      %3 = arith.andi %1, %2 : i1
      %4 = arith.cmpi slt, %arg2, %arg4 : i64
      %5 = arith.ori %4, %3 : i1
      %6 = arith.select %5, %arg2, %arg4 : i64
      %7 = arith.select %5, %arg3, %arg5 : i32
      tt.reduce.return %6, %7 : i64, i32
    }) : (tensor<4x8xi64>, tensor<4x8xi32>) -> (tensor<8xi64>, tensor<8xi32>)
    tt.return %0#0, %0#1 : tensor<8xi64>, tensor<8xi32>
  }
}

// CHECK-LABEL: func.func @argmin_i64
// CHECK: arith.constant 9223372036854775807 : i64
// CHECK: arith.constant -1 : i32
// CHECK: linalg.reduce ins(%{{.*}}, %{{.*}} : tensor<4x8xi64>, tensor<4x8xi32>) outs(%{{.*}}, %{{.*}} : tensor<8xi64>, tensor<8xi32>) dimensions = [0]
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi slt
// CHECK: arith.select

// -----

module {
  tt.func public @argmin_i32(%arg0: tensor<4x8xi32>, %arg1: tensor<4x8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
    %0:2 = "tt.reduce"(%arg0, %arg1) <{axis = 0 : i32}> ({
    ^bb0(%arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32):
      %1 = arith.cmpi eq, %arg2, %arg4 : i32
      %2 = arith.cmpi slt, %arg3, %arg5 : i32
      %3 = arith.andi %1, %2 : i1
      %4 = arith.cmpi slt, %arg2, %arg4 : i32
      %5 = arith.ori %4, %3 : i1
      %6 = arith.select %5, %arg2, %arg4 : i32
      %7 = arith.select %5, %arg3, %arg5 : i32
      tt.reduce.return %6, %7 : i32, i32
    }) : (tensor<4x8xi32>, tensor<4x8xi32>) -> (tensor<8xi32>, tensor<8xi32>)
    tt.return %0#0, %0#1 : tensor<8xi32>, tensor<8xi32>
  }
}

// CHECK-LABEL: func.func @argmin_i32
// CHECK: arith.constant 2147483647 : i32
// CHECK: arith.constant -1 : i32
// CHECK: linalg.reduce ins(%{{.*}}, %{{.*}} : tensor<4x8xi32>, tensor<4x8xi32>) outs(%{{.*}}, %{{.*}} : tensor<8xi32>, tensor<8xi32>) dimensions = [0]
// CHECK: arith.cmpi eq
// CHECK: arith.cmpi slt
// CHECK: arith.select
