// RUN: triton-shared-opt --triton-to-structured %s | FileCheck %s

module {
  tt.func public @tensor_pointer_in_conditional_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c16_i32 : i32
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.splat %1 : i32 -> tensor<16xi32>
    %4 = arith.addi %3, %2 : tensor<16xi32>
    %5 = arith.cmpi ne, %arg3, %c0_i32 : i32
    %6 = scf.if %5 -> (tensor<16x!tt.ptr<f16>>) {
      %10 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %11 = tt.addptr %10, %4 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      scf.yield %11 : tensor<16x!tt.ptr<f16>>
    } else {
      %10 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %11 = tt.addptr %10, %4 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      scf.yield %11 : tensor<16x!tt.ptr<f16>>
    }
    %7 = tt.load %6 : tensor<16x!tt.ptr<f16>>
    %8 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
    %9 = tt.addptr %8, %4 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
    tt.store %9, %7 : tensor<16x!tt.ptr<f16>>
    tt.return
  }

// CHECK:         tt.func public @tensor_pointer_in_conditional_kernel([[PARAM_0_:%.+]]: !tt.ptr<f16>, [[PARAM_1_:%.+]]: !tt.ptr<f16>, [[PARAM_2_:%.+]]: !tt.ptr<f16>, [[PARAM_3_:%.+]]: i32)
// CHECK:           [[VAR_0_:%.+]] = arith.cmpi ne
// CHECK:           [[VAR_1_:%.+]]:4 = scf.if [[VAR_0_]] -> (tensor<16x!tt.ptr<f16>>, index, index, !tt.ptr<f16>) {
// CHECK:           } else {
// CHECK:           }
// CHECK:           [[VAR_2_:%.+]] = tts.make_tptr [[VAR_1_]]#3 to sizes: [16], strides: [[[VAR_1_]]#2], offsets: [[[VAR_1_]]#1], shape: [0], order: [] : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK:           "tts.load"([[VAR_2_]])

  tt.func public @tensor_pointer_in_nested_conditional_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c16_i32 : i32
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.splat %1 : i32 -> tensor<16xi32>
    %4 = arith.addi %3, %2 : tensor<16xi32>
    %5 = arith.cmpi ne, %arg3, %c0_i32 : i32
    %6 = scf.if %5 -> (tensor<16x!tt.ptr<f16>>) {
      %10 = arith.cmpi ne, %arg4, %c0_i32 : i32
      %11 = scf.if %10 -> (tensor<16x!tt.ptr<f16>>) {
        %12 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
        scf.yield %12 : tensor<16x!tt.ptr<f16>>
      } else {
        %12 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
        %13 = tt.addptr %12, %4 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
        scf.yield %13 : tensor<16x!tt.ptr<f16>>
      }
      scf.yield %11 : tensor<16x!tt.ptr<f16>>
    } else {
      %10 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %11 = tt.addptr %10, %4 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      scf.yield %11 : tensor<16x!tt.ptr<f16>>
    }
    %7 = tt.load %6 : tensor<16x!tt.ptr<f16>>
    %8 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
    %9 = tt.addptr %8, %4 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
    tt.store %9, %7 : tensor<16x!tt.ptr<f16>>
    tt.return
  }

// CHECK:         tt.func public @tensor_pointer_in_nested_conditional_kernel([[PARAM_0_:%.+]]: !tt.ptr<f16>, [[PARAM_1_:%.+]]: !tt.ptr<f16>, [[PARAM_2_:%.+]]: !tt.ptr<f16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32)
// CHECK:           [[VAR_0_:%.+]] = arith.cmpi ne
// CHECK:           [[VAR_1_:%.+]]:4 = scf.if [[VAR_0_]] -> (tensor<16x!tt.ptr<f16>>, index, index, !tt.ptr<f16>) {
// CHECK:           } else {
// CHECK:           }
// CHECK:           [[VAR_2_:%.+]] = tts.make_tptr [[VAR_1_]]#3 to sizes: [16], strides: [[[VAR_1_]]#2], offsets: [[[VAR_1_]]#1], shape: [0], order: [] : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK:           "tts.load"([[VAR_2_]])
}
