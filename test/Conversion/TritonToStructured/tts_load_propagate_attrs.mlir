// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg2: i32,
      %arg3: i32,
      %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 {tt.divisibility = dense<16> : tensor<1xi32>} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %8 = tt.load %5 : tensor<1024x!tt.ptr<f32>>
    %17 = math.exp %8 : tensor<1024xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %21 = tt.addptr %20, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %21, %17 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: [[PTR:%.+]] = tts.make_tptr
// CHECK-SAME: tt.divisibility = dense<16> : tensor<1xi32>
// CHECK: "tts.load"([[PTR]])
