
// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s
// Test that scf.if returning multiple pointers both unstructured and structured are handled
// even with nested if statements.

// CHECK-LABEL: tt.func public @test_multiple_ptrs_if_kernel
// CHECK: %[[IF_RESULT:.*]]:4 = scf.if %{{.*}} -> (index, index, !tt.ptr<f32>, tensor<16x16x!tt.ptr<f32>>) {
// CHECK:   scf.yield %{{.*}}, %{{.*}}, %arg0, %{{.*}} : index, index, !tt.ptr<f32>, tensor<16x16x!tt.ptr<f32>>
// CHECK: } else {
// CHECK:   %{{.*}} = arith.select %{{.*}}, %arg1, %arg2 : !tt.ptr<f32>
// CHECK:   %{{.*}}:3 = scf.if %{{.*}} -> (index, index, tensor<16x16x!tt.ptr<f32>>) {
// CHECK:     scf.yield %{{.*}}, %{{.*}}, %{{.*}} : index, index, tensor<16x16x!tt.ptr<f32>>
// CHECK:   } else {
// CHECK:     scf.yield %{{.*}}, %{{.*}}, %{{.*}} : index, index, tensor<16x16x!tt.ptr<f32>>
// CHECK:   }
// CHECK:   scf.yield %{{.*}}#0, %{{.*}}#1, %{{.*}}, %{{.*}}#2 : index, index, !tt.ptr<f32>, tensor<16x16x!tt.ptr<f32>>
// CHECK: }
// CHECK: %{{.*}} = tts.make_tptr %[[IF_RESULT]]#2 to sizes: [16, 16], strides: [%[[IF_RESULT]]#1, %{{.*}}], offsets: [%[[IF_RESULT]]#0, %{{.*}}]
// CHECK: %[[LOAD:.*]] = "tts.load"(%{{.*}})
// CHECK: tt.store %[[IF_RESULT]]#3, %[[LOAD]]

module {
  tt.func public @test_multiple_ptrs_if_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16x16xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %3 = arith.cmpi ne, %arg13, %c0_i32 : i32
    %4:2 = scf.if %3 -> (tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>) {
      %6 = tt.splat %arg6 : i32 -> tensor<16x1xi32>
      %7 = arith.muli %1, %6 : tensor<16x1xi32>
      %8 = tt.splat %arg7 : i32 -> tensor<16x1xi32>
      %9 = arith.addi %8, %7 : tensor<16x1xi32>
      %10 = tt.splat %arg10 : i32 -> tensor<16x1xi32>
      %11 = arith.addi %9, %10 : tensor<16x1xi32>
      %12 = tt.broadcast %11 : tensor<16x1xi32> -> tensor<16x16xi32>
      %13 = tt.broadcast %2 : tensor<1x16xi32> -> tensor<16x16xi32>
      %14 = arith.addi %12, %13 : tensor<16x16xi32>
      %15 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
      %16 = tt.addptr %15, %14 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
      %17 = arith.divsi %14, %cst : tensor<16x16xi32>
      %18 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
      %19 = tt.addptr %18, %17 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
      scf.yield %16, %19 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
    } else {
      %6 = arith.cmpi ne, %arg14, %c0_i32 : i32
      %7:2 = scf.if %6 -> (tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>) {
        %8 = tt.splat %arg6 : i32 -> tensor<16x1xi32>
        %9 = arith.muli %1, %8 : tensor<16x1xi32>
        %10 = tt.splat %arg8 : i32 -> tensor<16x1xi32>
        %11 = arith.addi %10, %9 : tensor<16x1xi32>
        %12 = tt.splat %arg11 : i32 -> tensor<16x1xi32>
        %13 = arith.addi %11, %12 : tensor<16x1xi32>
        %14 = tt.broadcast %13 : tensor<16x1xi32> -> tensor<16x16xi32>
        %15 = tt.broadcast %2 : tensor<1x16xi32> -> tensor<16x16xi32>
        %16 = arith.addi %14, %15 : tensor<16x16xi32>
        %17 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
        %18 = tt.addptr %17, %16 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %19 = arith.divsi %16, %cst : tensor<16x16xi32>
        %20 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
        %21 = tt.addptr %20, %19 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        scf.yield %18, %21 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
      } else {
        %8 = tt.splat %arg6 : i32 -> tensor<16x1xi32>
        %9 = arith.muli %1, %8 : tensor<16x1xi32>
        %10 = tt.splat %arg9 : i32 -> tensor<16x1xi32>
        %11 = arith.addi %10, %9 : tensor<16x1xi32>
        %12 = tt.splat %arg12 : i32 -> tensor<16x1xi32>
        %13 = arith.addi %11, %12 : tensor<16x1xi32>
        %14 = tt.broadcast %13 : tensor<16x1xi32> -> tensor<16x16xi32>
        %15 = tt.broadcast %2 : tensor<1x16xi32> -> tensor<16x16xi32>
        %16 = arith.addi %14, %15 : tensor<16x16xi32>
        %17 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
        %18 = tt.addptr %17, %16 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %19 = arith.divsi %16, %cst : tensor<16x16xi32>
        %20 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
        %21 = tt.addptr %20, %19 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        scf.yield %18, %21 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
      }
      scf.yield %7#0, %7#1 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
    }
    %5 = tt.load %4#0 : tensor<16x16x!tt.ptr<f32>>
    tt.store %4#1, %5 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
