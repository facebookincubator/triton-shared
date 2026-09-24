// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: tt.func public @tensor_pointer_in_conditional_kernel
// CHECK-DAG: %c0_i32 = arith.constant 0 : i32
// CHECK-DAG: %c2_i32 = arith.constant 2 : i32
// CHECK: %[[RESULT:.*]]:2 = scf.if %{{.*}} -> (index, !tt.ptr<f16>) {
// CHECK: } else {
// CHECK:   %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : index
// CHECK:   %{{.*}} = arith.select %{{.*}}, %arg1, %arg2 : !tt.ptr<f16>
// CHECK: }
// CHECK-NOT: scf.if{{.*}}tensor<{{.*}}!tt.ptr
// CHECK: %{{.*}} = tts.make_tptr %[[RESULT]]#1 to sizes: [16, 32], strides: [{{%c16|16}}, {{%c32|32}}], offsets: [%[[RESULT]]#0, {{%c0|0}}], shape: [0, 0], order: []

module {
  tt.func public @tensor_pointer_in_conditional_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant dense<10> : tensor<16x32xi32>
    %cst_0 = arith.constant dense<32> : tensor<1x32xi32>
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<16> : tensor<16x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %3 = arith.muli %2, %cst_1 : tensor<16x1xi32>
    %4 = tt.get_program_id x : i32
    %5 = tt.splat %4 : i32 -> tensor<16x1xi32>
    %6 = arith.addi %3, %5 : tensor<16x1xi32>
    %7 = arith.cmpi ne, %arg4, %c0_i32 : i32
    %8 = scf.if %7 -> (tensor<16x32x!tt.ptr<f16>>) {
      %10 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>>
      %11 = tt.addptr %10, %6 : tensor<16x1x!tt.ptr<f16>>, tensor<16x1xi32>
      %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
      %13 = arith.muli %12, %cst_0 : tensor<1x32xi32>
      %14 = tt.broadcast %11 : tensor<16x1x!tt.ptr<f16>> -> tensor<16x32x!tt.ptr<f16>>
      %15 = tt.broadcast %13 : tensor<1x32xi32> -> tensor<16x32xi32>
      %16 = tt.addptr %14, %15 : tensor<16x32x!tt.ptr<f16>>, tensor<16x32xi32>
      %17 = tt.addptr %16, %cst : tensor<16x32x!tt.ptr<f16>>, tensor<16x32xi32>
      scf.yield %17 : tensor<16x32x!tt.ptr<f16>>
    } else {
      %10 = arith.remsi %4, %c2_i32 : i32
      %11 = arith.cmpi eq, %10, %c0_i32 : i32
      %12 = scf.if %11 -> (tensor<16x32x!tt.ptr<f16>>) {
        %13 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>>
        %14 = tt.addptr %13, %6 : tensor<16x1x!tt.ptr<f16>>, tensor<16x1xi32>
        %15 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
        %16 = arith.muli %15, %cst_0 : tensor<1x32xi32>
        %17 = tt.broadcast %14 : tensor<16x1x!tt.ptr<f16>> -> tensor<16x32x!tt.ptr<f16>>
        %18 = tt.broadcast %16 : tensor<1x32xi32> -> tensor<16x32xi32>
        %19 = tt.addptr %17, %18 : tensor<16x32x!tt.ptr<f16>>, tensor<16x32xi32>
        scf.yield %19 : tensor<16x32x!tt.ptr<f16>>
      } else {
        %13 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>>
        %14 = tt.addptr %13, %6 : tensor<16x1x!tt.ptr<f16>>, tensor<16x1xi32>
        %15 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
        %16 = arith.muli %15, %cst_0 : tensor<1x32xi32>
        %17 = tt.broadcast %14 : tensor<16x1x!tt.ptr<f16>> -> tensor<16x32x!tt.ptr<f16>>
        %18 = tt.broadcast %16 : tensor<1x32xi32> -> tensor<16x32xi32>
        %19 = tt.addptr %17, %18 : tensor<16x32x!tt.ptr<f16>>, tensor<16x32xi32>
        scf.yield %19 : tensor<16x32x!tt.ptr<f16>>
      }
      scf.yield %12 : tensor<16x32x!tt.ptr<f16>>
    }
    %9 = tt.load %8 : tensor<16x32x!tt.ptr<f16>>
    tt.store %8, %9 : tensor<16x32x!tt.ptr<f16>>
    tt.return
  }
}
