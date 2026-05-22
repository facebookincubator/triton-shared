// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: tt.func public @tensor_offset_in_conditional_kernel
// CHECK-DAG: %c1 = arith.constant 1 : index
// CHECK-DAG: %c0 = arith.constant 0 : index
// CHECK-DAG: %c1_i32 = arith.constant 1 : i32
// CHECK-DAG: %c16_i32 = arith.constant 16 : i32
// CHECK-DAG: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[PTR:.*]] = arith.select{{.*}}: !tt.ptr<f32>
// CHECK-NOT: scf.if{{.*}}tensor<{{.*}}!tt.ptr
// CHECK: scf.for{{.*}}iter_args(%[[OFFSET:.*]] = %c0) -> (index)
// CHECK: tts.make_tptr %[[PTR]] to sizes: [16], strides: [1], offsets: [%[[OFFSET]]]{{.*}}: <f32> to tensor<16x!tt.ptr<f32>>

module {
  tt.func public @tensor_offset_in_conditional_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst = arith.constant dense<1> : tensor<16xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg2, %c0_i32 : i32
    %1 = scf.if %0 -> (tensor<16x!tt.ptr<f32>>) {
      %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %5 = tt.addptr %4, %3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      scf.yield %5 : tensor<16x!tt.ptr<f32>>
    } else {
      %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %5 = tt.addptr %4, %3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      scf.yield %5 : tensor<16x!tt.ptr<f32>>
    }
    %2 = scf.for %arg4 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg5 = %1) -> (tensor<16x!tt.ptr<f32>>)  : i32 {
      %3 = tt.load %arg5 : tensor<16x!tt.ptr<f32>>
      tt.store %arg5, %3 : tensor<16x!tt.ptr<f32>>
      %4 = tt.addptr %arg5, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      scf.yield %4 : tensor<16x!tt.ptr<f32>>
    }
    tt.return
  }
}
