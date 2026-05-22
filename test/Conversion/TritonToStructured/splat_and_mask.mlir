// RUN: triton-shared-opt --triton-to-structured --canonicalize --cse %s | FileCheck %s

tt.func public @splat_and_mask(%arg0: !tt.ptr<f32>) -> tensor<1x64xf32> {
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<1x64xf32>
  %cst_0 = arith.constant dense<64> : tensor<1x64xi64>
  %c64_i64 = arith.constant 64 : i64
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id y : i32
  %1 = arith.cmpi slt, %0, %c1_i32 : i32
  %2 = tt.get_program_id x : i32
  %3 = arith.muli %2, %c64_i32 : i32
  %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
  %6 = tt.splat %3 : i32 -> tensor<1x64xi32>
  %7 = arith.addi %6, %5 : tensor<1x64xi32>
  %8 = arith.extsi %7 : tensor<1x64xi32> to tensor<1x64xi64>
  %9 = arith.cmpi slt, %8, %cst_0 : tensor<1x64xi64>
  %10 = tt.splat %1 : i1 -> tensor<1x64xi1>
  %11 = arith.andi %9, %10 : tensor<1x64xi1>
  %12 = arith.muli %0, %c64_i32 : i32
  %13 = tt.splat %12 : i32 -> tensor<1x64xi32>
  %14 = arith.addi %13, %7 : tensor<1x64xi32>
  %15 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
  %16 = tt.addptr %15, %14 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
  %17 = tt.load %16, %11, %cst evictionPolicy = evict_last : tensor<1x64x!tt.ptr<f32>>
  tt.return %17 : tensor<1x64xf32>
}

// CHECK-LABEL: tt.func public @splat_and_mask
// CHECK-SAME:    (%[[ARG0:.*]]: !tt.ptr<f32>) -> tensor<1x64xf32>
// CHECK-DAG:     %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C64:.*]] = arith.constant 64 : index
// CHECK-DAG:     %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK-DAG:     %[[C64_I32:.*]] = arith.constant 64 : i32
// CHECK:         %[[PROGRAM_ID_Y:.*]] = tt.get_program_id y : i32
// CHECK:         %[[DIM0_CHECK:.*]] = arith.cmpi slt, %[[PROGRAM_ID_Y]], %[[C1_I32]] : i32
// CHECK:         %[[PROGRAM_ID_X:.*]] = tt.get_program_id x : i32
// CHECK:         %[[MUL1:.*]] = arith.muli %[[PROGRAM_ID_X]], %[[C64_I32]] : i32
// CHECK:         %[[CAST1:.*]] = arith.index_cast %[[MUL1]] : i32 to index
// CHECK:         %[[MUL2:.*]] = arith.muli %[[PROGRAM_ID_Y]], %[[C64_I32]] : i32
// CHECK:         %[[CAST2:.*]] = arith.index_cast %[[MUL2]] : i32 to index
// CHECK:         %[[ADD1:.*]] = arith.addi %[[CAST2]], %[[CAST1]] : index
// CHECK:         %[[TPTR:.*]] = tts.make_tptr %[[ARG0]] to sizes: [1, 64], strides: [64, 1], offsets: [%[[ADD1]], 0], shape: [0, 0], order: [] : <f32> to tensor<1x64x!tt.ptr<f32>>
// CHECK:         %[[ADD2:.*]] = arith.addi %[[CAST1]], %[[C64]] : index
// CHECK:         %[[MIN1:.*]] = arith.minsi %[[ADD2]], %[[C64]] : index
// CHECK:         %[[MAX:.*]] = arith.maxsi %[[MIN1]], %[[CAST1]] : index
// CHECK:         %[[SUB:.*]] = arith.subi %[[MAX]], %[[CAST1]] : index
// CHECK:         %[[SELECT_DIM0:.*]] = arith.select %[[DIM0_CHECK]], %[[C1]], %[[C0]] : index
// CHECK:         %[[SELECT_DIM1:.*]] = arith.select %[[DIM0_CHECK]], %[[SUB]], %[[C0]] : index
// CHECK:         %[[LOAD:.*]] = "tts.load"(%[[TPTR]], %[[SELECT_DIM0]], %[[SELECT_DIM1]], %[[CST]]) {{.*}} : (tensor<1x64x!tt.ptr<f32>>, index, index, f32) -> tensor<1x64xf32>
// CHECK:         tt.return %[[LOAD]] : tensor<1x64xf32>
