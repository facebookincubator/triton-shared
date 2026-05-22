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
}

// Check that scf.if yielding scalar pointer now yields a scalar offset.
// CHECK-LABEL:   tt.func public @simple_cf_into_structured_load
// CHECK-SAME:    ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK:           [[VAR_0_:%.+]] = arith.cmpi eq, [[PARAM_2_]], [[CST_1_]] : i32
// CHECK:           [[VAR_1_:%.+]] = scf.if [[VAR_0_]] -> (index) {
// CHECK:             [[VAR_6_:%.+]] = arith.muli [[PARAM_2_]], [[CST_2_]] : i32
// CHECK:             [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : i32 to index
// CHECK:             scf.yield [[VAR_7_]] : index
// CHECK:           } else {
// CHECK:             [[VAR_6_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:             scf.yield [[VAR_6_]] : index
// CHECK:           }
// CHECK:           [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_6_]] : index
// CHECK:           [[VAR_3_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4], strides: [1], offsets: {{\[}}[[VAR_2_]]{{\]}}, shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:           [[VAR_4_:%.+]] = "tts.load"([[VAR_3_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
// CHECK:           [[VAR_5_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:           "tts.store"([[VAR_5_]], [[VAR_4_]]) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
