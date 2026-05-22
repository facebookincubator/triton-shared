// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @test_forward_cumsum_accepted(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %3 = tt.load %2 : tensor<64x!tt.ptr<f32>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %5 = arith.addf %arg1, %arg2 : f32
      tt.scan.return %5 : f32
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %7 = tt.addptr %6, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %7, %4 : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: @test_forward_cumsum_accepted
// CHECK: ttx.cumsum
// CHECK-NOT: tt.scan

// -----

module {
  tt.func public @test_reverse_cumsum_rejected(%arg0: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %3 = tt.load %2 : tensor<64x!tt.ptr<f32>>
    %4 = "tt.scan"(%3) <{axis = 0 : i32, reverse = true}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %5 = arith.addf %arg1, %arg2 : f32
      tt.scan.return %5 : f32
    }) : (tensor<64xf32>) -> tensor<64xf32>
    %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %7 = tt.addptr %6, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %7, %4 : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: @test_reverse_cumsum_rejected
// CHECK-NOT: ttx.cumsum
// CHECK: tt.scan
// CHECK-SAME: reverse = true
// CHECK-NOT: ttx.cumsum
