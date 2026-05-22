// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: tt.func public @test_multiple_ptrs_if_kernel
// CHECK-DAG: %c1 = arith.constant 1 : index
// CHECK-DAG: %c0 = arith.constant 0 : index
// CHECK-DAG: %cst = arith.constant dense<2> : tensor<16x16xi32>
// CHECK-DAG: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[PTR:.*]] = arith.select{{.*}}: !tt.ptr<f32>
// CHECK: %[[IF_RESULT:.*]]:3 = scf.if{{.*}}-> (index, index, tensor<16x16x!tt.ptr<f32>>)
// CHECK: tts.make_tptr %[[PTR]] to sizes: [16, 16], strides: [%[[IF_RESULT]]#1, %c1], offsets: [%[[IF_RESULT]]#0, %c0]{{.*}}: <f32> to tensor<16x16x!tt.ptr<f32>>
// CHECK: tt.store %[[IF_RESULT]]#2

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
      %6 = tt.splat %arg6 : i32 -> tensor<16x1xi32>
      %7 = arith.muli %1, %6 : tensor<16x1xi32>
      %8 = tt.splat %arg9 : i32 -> tensor<16x1xi32>
      %9 = arith.addi %8, %7 : tensor<16x1xi32>
      %10 = tt.splat %arg12 : i32 -> tensor<16x1xi32>
      %11 = arith.addi %9, %10 : tensor<16x1xi32>
      %12 = tt.broadcast %11 : tensor<16x1xi32> -> tensor<16x16xi32>
      %13 = tt.broadcast %2 : tensor<1x16xi32> -> tensor<16x16xi32>
      %14 = arith.addi %12, %13 : tensor<16x16xi32>
      %15 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
      %16 = tt.addptr %15, %14 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
      %17 = arith.divsi %14, %cst : tensor<16x16xi32>
      %18 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
      %19 = tt.addptr %18, %17 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
      scf.yield %16, %19 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
    }
    %5 = tt.load %4#0 : tensor<16x16x!tt.ptr<f32>>
    tt.store %4#1, %5 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
