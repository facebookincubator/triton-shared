// RUN: triton-shared-opt --triton-to-unstructured --canonicalize %s | FileCheck %s

module {
  tt.func public @gather(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<4> : tensor<4xi32>
    %cst_0 = arith.constant dense<64> : tensor<4xi32>
    %cst_1 = arith.constant dense<3> : tensor<4xi32>
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %4 = arith.divsi %arg3, %cst_1 : tensor<4xi32>
      %5 = tt.splat %arg2 : i32 -> tensor<4xi32>
      %6 = arith.addi %4, %5 : tensor<4xi32>
      %7 = arith.cmpi slt, %6, %cst_0 : tensor<4xi32>
      %8 = tt.addptr %1, %6 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %9 = tt.load %8, %7 : tensor<4x!tt.ptr<f32>>
      %10 = tt.addptr %2, %6 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %10, %9 : tensor<4x!tt.ptr<f32>>
      %11 = arith.addi %6, %cst : tensor<4xi32>
      %12 = arith.addi %arg4, %cst : tensor<4xi32>
      %13 = arith.addi %arg2, %c1_i32 : i32
      %14:2 = scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg6 = %11, %arg7 = %12) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
        %23 = arith.addi %arg5, %c1_i32 : i32
        %24 = arith.muli %13, %23 : i32
        %25 = tt.splat %24 : i32 -> tensor<4xi32>
        %26 = arith.divsi %arg6, %25 : tensor<4xi32>
        %27 = arith.addi %26, %5 : tensor<4xi32>
        %28 = arith.cmpi slt, %27, %cst_0 : tensor<4xi32>
        %29 = tt.addptr %1, %27 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        %30 = tt.load %29, %28 : tensor<4x!tt.ptr<f32>>
        %31 = tt.addptr %2, %27 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        tt.store %31, %30 : tensor<4x!tt.ptr<f32>>
        %32 = arith.addi %27, %cst : tensor<4xi32>
        %33 = arith.addi %arg7, %cst : tensor<4xi32>
        scf.yield %32, %33 : tensor<4xi32>, tensor<4xi32>
      }
      %15 = arith.divsi %14#0, %cst_1 : tensor<4xi32>
      %16 = arith.addi %15, %5 : tensor<4xi32>
      %17 = arith.cmpi slt, %16, %cst_0 : tensor<4xi32>
      %18 = tt.addptr %1, %16 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %19 = tt.load %18, %17 : tensor<4x!tt.ptr<f32>>
      %20 = tt.addptr %2, %16 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %20, %19 : tensor<4x!tt.ptr<f32>>
      %21 = arith.addi %16, %cst : tensor<4xi32>
      %22 = arith.addi %14#1, %cst : tensor<4xi32>
      scf.yield %21, %22 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @gather([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<4> : tensor<4xi32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<64> : tensor<4xi32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<3> : tensor<4xi32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = scf.for [[VAR_arg2_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg3_:%.+]] = [[VAR_0_]]) -> (tensor<4xi32>)  : i32 {
// CHECK-DAG:         [[VAR_2_:%.+]] = arith.divsi [[VAR_arg3_]], [[VAR_cst_1_]] : tensor<4xi32>
// CHECK-DAG:         [[VAR_3_:%.+]] = tt.splat [[VAR_arg2_]] : i32 -> tensor<4xi32>
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_2_]], [[VAR_3_]] : tensor<4xi32>
// CHECK:             [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[VAR_cst_0_]] : tensor<4xi32>
// CHECK:             [[VAR_6_:%.+]] = tts.gather [[PARAM_0_]]{{.}}[[VAR_4_]]{{.}} mask = [[VAR_5_]] : (<f32>, tensor<4xi32>) -> tensor<4xf32>
// CHECK:             tts.scatter [[VAR_6_]] into [[PARAM_1_]]{{.}}[[VAR_4_]]{{.}} : tensor<4xf32> into (<f32>, tensor<4xi32>)
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_4_]], [[VAR_cst_]] : tensor<4xi32>
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_arg2_]], [[CST_1_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]] = scf.for [[VAR_arg4_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg5_:%.+]] = [[VAR_7_]]) -> (tensor<4xi32>)  : i32 {
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.addi [[VAR_arg4_]], [[CST_1_]] : i32
// CHECK:               [[VAR_16_:%.+]] = arith.muli [[VAR_8_]], [[VAR_15_]] : i32
// CHECK:               [[VAR_17_:%.+]] = tt.splat [[VAR_16_]] : i32 -> tensor<4xi32>
// CHECK:               [[VAR_18_:%.+]] = arith.divsi [[VAR_arg5_]], [[VAR_17_]] : tensor<4xi32>
// CHECK:               [[VAR_19_:%.+]] = arith.addi [[VAR_18_]], [[VAR_3_]] : tensor<4xi32>
// CHECK:               [[VAR_20_:%.+]] = arith.cmpi slt, [[VAR_19_]], [[VAR_cst_0_]] : tensor<4xi32>
// CHECK:               [[VAR_21_:%.+]] = tts.gather [[PARAM_0_]]{{.}}[[VAR_19_]]{{.}} mask = [[VAR_20_]] : (<f32>, tensor<4xi32>) -> tensor<4xf32>
// CHECK:               tts.scatter [[VAR_21_]] into [[PARAM_1_]]{{.}}[[VAR_19_]]{{.}} : tensor<4xf32> into (<f32>, tensor<4xi32>)
// CHECK:               [[VAR_22_:%.+]] = arith.addi [[VAR_19_]], [[VAR_cst_]] : tensor<4xi32>
// CHECK:               scf.yield [[VAR_22_]] : tensor<4xi32>
// CHECK:             }
// CHECK:             [[VAR_10_:%.+]] = arith.divsi [[VAR_9_]], [[VAR_cst_1_]] : tensor<4xi32>
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[VAR_10_]], [[VAR_3_]] : tensor<4xi32>
// CHECK:             [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_11_]], [[VAR_cst_0_]] : tensor<4xi32>
// CHECK:             [[VAR_13_:%.+]] = tts.gather [[PARAM_0_]]{{.}}[[VAR_11_]]{{.}} mask = [[VAR_12_]] : (<f32>, tensor<4xi32>) -> tensor<4xf32>
// CHECK:             tts.scatter [[VAR_13_]] into [[PARAM_1_]]{{.}}[[VAR_11_]]{{.}} : tensor<4xf32> into (<f32>, tensor<4xi32>)
// CHECK:             [[VAR_14_:%.+]] = arith.addi [[VAR_11_]], [[VAR_cst_]] : tensor<4xi32>
// CHECK:             scf.yield [[VAR_14_]] : tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
