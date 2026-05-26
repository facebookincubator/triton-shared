// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

module {
  tt.func public @simple_cf_into_structured_load(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.cmpi eq, %arg2, %c1_i32 : i32
    %1 = scf.if %0 -> (!tt.ptr<f32>) {
      %9 = arith.muli %arg2, %c2_i32 : i32
      %10 = tt.addptr %arg0, %9 : !tt.ptr<f32>, i32
      scf.yield %10 : !tt.ptr<f32>
    } else {
      %9 = tt.addptr %arg0, %arg2 : !tt.ptr<f32>, i32
      scf.yield %9 : !tt.ptr<f32>
    }
    %2 = tt.addptr %1, %c6_i32 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %6 = tt.load %5 : tensor<4x!tt.ptr<f32>>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %8 = tt.addptr %7, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %8, %6 : tensor<4x!tt.ptr<f32>>
    tt.return
  }

  tt.func public @select_into_structured_load(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: !tt.ptr<f32>) attributes {noinline = false} {
    %c6_i32 = arith.constant 6 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.cmpi eq, %arg2, %c1_i32 : i32
    %1 = arith.select %0, %arg0, %arg3 : !tt.ptr<f32>
    %2 = tt.addptr %1, %c6_i32 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %6 = tt.load %5 : tensor<4x!tt.ptr<f32>>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %8 = tt.addptr %7, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %8, %6 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func public @simple_cf_into_structured_load([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK:           [[VAR_0_:%.+]] = arith.cmpi eq
// CHECK-DAG:       [[VAR_1_:%.+]] = scf.if [[VAR_0_]] -> (index) {
// CHECK:           } else {
// CHECK:           }
// CHECK:           [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_6_]] : index
// CHECK:           tts.make_tptr [[PARAM_0_]] to sizes: [4], strides: [1], offsets: {{.}}[[VAR_2_]]{{.}}, shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>

// CHECK:         tt.func public @select_into_structured_load([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK:           [[VAR_0_:%.+]] = arith.cmpi eq
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.select [[VAR_0_]], [[PARAM_0_]], [[PARAM_3_]] : !tt.ptr<f32>
// CHECK:           tts.make_tptr [[VAR_1_]] to sizes: [4], strides: [1], offsets: [{{%c6|6}}], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
