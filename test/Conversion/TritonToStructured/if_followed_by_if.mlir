// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: tt.func public @tensor_pointer_in_conditional_kernel
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[V0:.*]] = scf.if{{.*}}-> (index) {
// CHECK: scf.yield{{.*}}: index
// CHECK: } else {
// CHECK: scf.yield{{.*}}: index
// CHECK: }
// CHECK-NOT: scf.if{{.*}}tensor<{{.*}}!tt.ptr
// CHECK: tts.make_tptr %arg0 to sizes: [16], strides: [%[[C1]]], offsets: [%[[V0]]]

module {
  tt.func public @tensor_pointer_in_conditional_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<3> : tensor<16xi32>
    %c2_i32 = arith.constant 2 : i32
    %cst_0 = arith.constant dense<1> : tensor<16xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
    %3 = tt.get_program_id x : i32
    %4 = tt.splat %3 : i32 -> tensor<16xi32>
    %5 = tt.addptr %2, %4 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
    %6 = arith.cmpi ne, %arg3, %c0_i32 : i32
    %7 = scf.if %6 -> (tensor<16x!tt.ptr<f16>>) {
      %12 = tt.addptr %5, %cst_0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %13 = arith.remsi %3, %c2_i32 : i32
      %14 = arith.cmpi eq, %13, %c0_i32 : i32
      %15 = scf.if %14 -> (tensor<16x!tt.ptr<f16>>) {
        %16 = tt.addptr %5, %cst : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
        scf.yield %16 : tensor<16x!tt.ptr<f16>>
      } else {
        scf.yield %12 : tensor<16x!tt.ptr<f16>>
      }
      scf.yield %15 : tensor<16x!tt.ptr<f16>>
    } else {
      scf.yield %5 : tensor<16x!tt.ptr<f16>>
    }
    %8 = scf.if %6 -> (tensor<16x!tt.ptr<f16>>) {
      %12 = tt.addptr %7, %cst_0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      scf.yield %12 : tensor<16x!tt.ptr<f16>>
    } else {
      scf.yield %7 : tensor<16x!tt.ptr<f16>>
    }
    %9 = tt.load %8 : tensor<16x!tt.ptr<f16>>
    %10 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
    %11 = tt.addptr %10, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
    tt.store %11, %9 : tensor<16x!tt.ptr<f16>>
    tt.return
  }
}
