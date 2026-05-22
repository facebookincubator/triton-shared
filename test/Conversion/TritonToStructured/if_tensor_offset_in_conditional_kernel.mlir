// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: tt.func public @tensor_offset_in_conditional_kernel
// CHECK-DAG: %c0 = arith.constant 0 : index
// CHECK-DAG: %c1 = arith.constant 1 : index
// CHECK-DAG: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[PTR:.*]] = scf.if %{{.*}} -> (!tt.ptr<f32>) {
// CHECK: } else {
// CHECK:   %{{.*}} = arith.select %{{.*}}, %arg1, %arg2 : !tt.ptr<f32>
// CHECK: }
// CHECK: %[[OFFSET_STRIDE:.*]]:2 = scf.if %{{.*}} -> (index, index) {
// CHECK: } else {
// CHECK: }
// CHECK: %[[COL_OFFSET:.*]] = scf.if %{{.*}} -> (index) {
// CHECK: } else {
// CHECK: }
// CHECK: %[[ROW_OFFSET:.*]] = arith.addi %[[OFFSET_STRIDE]]#0, %[[COL_OFFSET]] : index
// CHECK-NOT: scf.if{{.*}}tensor<{{.*}}!tt.ptr
// CHECK: %{{.*}} = tts.make_tptr %[[PTR]] to sizes: [32, 16], strides: [%[[OFFSET_STRIDE]]#1, %c1], offsets: [%[[ROW_OFFSET]], %c0], shape: [0, 0], order: []
// CHECK: %[[STRIDE:.*]] = scf.if %{{.*}} -> (index) {
// CHECK: } else {
// CHECK: }
// CHECK: %{{.*}} = tts.make_tptr %arg20 to sizes: [32, 16], strides: [%[[STRIDE]], %c1], offsets: [%c0, %c0], shape: [0, 0], order: []

module {
  tt.func public @tensor_offset_in_conditional_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32, %arg17: i32, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<1x16xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg3, %c0_i32 : i32
    %1 = scf.if %0 -> (!tt.ptr<f32>) {
      scf.yield %arg0 : !tt.ptr<f32>
    } else {
      %21 = arith.cmpi ne, %arg4, %c0_i32 : i32
      %22 = arith.select %21, %arg1, %arg2 : !tt.ptr<f32>
      scf.yield %22 : !tt.ptr<f32>
    }
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = arith.cmpi ne, %arg12, %c0_i32 : i32
    %4 = scf.if %3 -> (tensor<32x1xi32>) {
      %21 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %22 = tt.splat %arg9 : i32 -> tensor<32x1xi32>
      %23 = arith.muli %21, %22 : tensor<32x1xi32>
      %24 = tt.splat %arg6 : i32 -> tensor<32x1xi32>
      %25 = arith.addi %24, %23 : tensor<32x1xi32>
      scf.yield %25 : tensor<32x1xi32>
    } else {
      %21 = arith.cmpi ne, %arg13, %c0_i32 : i32
      %22 = scf.if %21 -> (tensor<32x1xi32>) {
        %23 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %24 = tt.splat %arg10 : i32 -> tensor<32x1xi32>
        %25 = arith.muli %23, %24 : tensor<32x1xi32>
        %26 = tt.splat %arg7 : i32 -> tensor<32x1xi32>
        %27 = arith.addi %26, %25 : tensor<32x1xi32>
        scf.yield %27 : tensor<32x1xi32>
      } else {
        %23 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %24 = tt.splat %arg11 : i32 -> tensor<32x1xi32>
        %25 = arith.muli %23, %24 : tensor<32x1xi32>
        %26 = tt.splat %arg8 : i32 -> tensor<32x1xi32>
        %27 = arith.addi %26, %25 : tensor<32x1xi32>
        scf.yield %27 : tensor<32x1xi32>
      }
      scf.yield %22 : tensor<32x1xi32>
    }
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %6 = arith.cmpi ne, %arg17, %c0_i32 : i32
    %7 = scf.if %6 -> (tensor<1x16xi32>) {
      %21 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
      %22 = tt.splat %arg15 : i32 -> tensor<1x16xi32>
      %23 = arith.addi %22, %21 : tensor<1x16xi32>
      scf.yield %23 : tensor<1x16xi32>
    } else {
      %21 = arith.cmpi ne, %arg18, %c0_i32 : i32
      %22 = scf.if %21 -> (tensor<1x16xi32>) {
        %23 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        %24 = arith.addi %23, %cst : tensor<1x16xi32>
        scf.yield %24 : tensor<1x16xi32>
      } else {
        %23 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        %24 = tt.splat %arg16 : i32 -> tensor<1x16xi32>
        %25 = arith.addi %24, %23 : tensor<1x16xi32>
        scf.yield %25 : tensor<1x16xi32>
      }
      scf.yield %22 : tensor<1x16xi32>
    }
    %8 = tt.splat %1 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %9 = tt.addptr %8, %4 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %10 = tt.broadcast %9 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x16x!tt.ptr<f32>>
    %11 = tt.broadcast %7 : tensor<1x16xi32> -> tensor<32x16xi32>
    %12 = tt.addptr %10, %11 : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
    %13 = tt.load %12 : tensor<32x16x!tt.ptr<f32>>
    %14 = scf.if %3 -> (tensor<32x1xi32>) {
      %21 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
      %22 = tt.splat %arg9 : i32 -> tensor<32x1xi32>
      %23 = arith.muli %21, %22 : tensor<32x1xi32>
      scf.yield %23 : tensor<32x1xi32>
    } else {
      %21 = arith.cmpi ne, %arg13, %c0_i32 : i32
      %22 = scf.if %21 -> (tensor<32x1xi32>) {
        %23 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %24 = tt.splat %arg10 : i32 -> tensor<32x1xi32>
        %25 = arith.muli %23, %24 : tensor<32x1xi32>
        scf.yield %25 : tensor<32x1xi32>
      } else {
        %23 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
        %24 = tt.splat %arg11 : i32 -> tensor<32x1xi32>
        %25 = arith.muli %23, %24 : tensor<32x1xi32>
        scf.yield %25 : tensor<32x1xi32>
      }
      scf.yield %22 : tensor<32x1xi32>
    }
    %15 = scf.if %6 -> (tensor<1x16xi32>) {
      %21 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
      scf.yield %21 : tensor<1x16xi32>
    } else {
      %21 = arith.cmpi ne, %arg18, %c0_i32 : i32
      %22 = scf.if %21 -> (tensor<1x16xi32>) {
        %23 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        scf.yield %23 : tensor<1x16xi32>
      } else {
        %23 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        scf.yield %23 : tensor<1x16xi32>
      }
      scf.yield %22 : tensor<1x16xi32>
    }
    %16 = tt.splat %arg20 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %17 = tt.addptr %16, %14 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %18 = tt.broadcast %17 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x16x!tt.ptr<f32>>
    %19 = tt.broadcast %15 : tensor<1x16xi32> -> tensor<32x16xi32>
    %20 = tt.addptr %18, %19 : tensor<32x16x!tt.ptr<f32>>, tensor<32x16xi32>
    tt.store %20, %13 : tensor<32x16x!tt.ptr<f32>>
    tt.return
  }
}
