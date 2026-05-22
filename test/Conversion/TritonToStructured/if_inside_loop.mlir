// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: tt.func public @tensor_offset_in_conditional_kernel
// CHECK-DAG:     %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C16_I32:.*]] = arith.constant 16 : i32
// CHECK-DAG:     %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:         %{{.*}}:2 = scf.for %{{.*}} = %[[C0_I32]] to %[[C16_I32]] step %[[C1_I32]] iter_args(%[[ITER_ARG5:.*]] = %[[C0]], %[[ITER_ARG6:.*]] = %[[C0]]) -> (index, index) : i32 {
// CHECK:           %[[STORE_PTR:.*]] = tts.make_tptr %{{.*}} to sizes: [16], strides: [1], offsets: [%[[ITER_ARG5]]], shape: [0], order: [] : <f32> to tensor<16x!tt.ptr<f32>>
// CHECK:           %[[CMP:.*]] = arith.cmpi ne, %{{.*}}, %[[C0_I32]] : i32
// CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %{{.*}}, %{{.*}} : !tt.ptr<f32>
// CHECK-NOT:       scf.if{{.*}}tensor<{{.*}}!tt.ptr
// CHECK:           %[[LOAD_PTR:.*]] = tts.make_tptr %[[SEL]] to sizes: [16], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<16x!tt.ptr<f32>>
// CHECK:           %[[LOAD:.*]] = "tts.load"(%[[LOAD_PTR]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<16x!tt.ptr<f32>>) -> tensor<16xf32>
// CHECK:           "tts.store"(%[[STORE_PTR]], %[[LOAD]]) <{static_mask_dims = array<i64>}> : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>) -> ()
// CHECK:           %[[NEXT_OFFSET:.*]] = arith.addi %[[ITER_ARG6]], %[[C16]] : index
// CHECK:           scf.yield %[[ITER_ARG6]], %[[NEXT_OFFSET]] : index, index
// CHECK:         }
// CHECK:         tt.return

module {
  tt.func public @tensor_offset_in_conditional_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<16> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %3:2 = scf.for %arg4 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg5 = %2, %arg6 = %2) -> (tensor<16x!tt.ptr<f32>>, tensor<16x!tt.ptr<f32>>) : i32 {
      %4 = arith.cmpi ne, %arg2, %c0_i32 : i32
      %5 = scf.if %4 -> (tensor<16x!tt.ptr<f32>>) {
        %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %9 = tt.addptr %8, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %9 : tensor<16x!tt.ptr<f32>>
      } else {
        %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
        %9 = tt.addptr %8, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %9 : tensor<16x!tt.ptr<f32>>
      }
      %6 = tt.load %5 : tensor<16x!tt.ptr<f32>>
      tt.store %arg5, %6 : tensor<16x!tt.ptr<f32>>
      %7 = tt.addptr %arg6, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      scf.yield %arg6, %7 : tensor<16x!tt.ptr<f32>>, tensor<16x!tt.ptr<f32>>
    }
    tt.return
  }
}
