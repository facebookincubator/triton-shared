// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: @cf_postincrement
// CHECK: %[[SELECT:.*]] = arith.select {{.*}} : !tt.ptr<f32>
// CHECK: %[[IF_RESULT:.*]] = scf.if {{.*}} -> (index)
// CHECK: tts.make_tptr %[[SELECT]] to sizes: [16], strides: [{{.*}}], offsets: [%[[IF_RESULT]]]

module {
  tt.func public @cf_postincrement(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<16xi32>
    %c0_i32 = arith.constant 0 : i32
    %c11_i32 = arith.constant 11 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = arith.cmpi eq, %arg3, %c11_i32 : i32
    %2 = scf.if %1 -> (tensor<16x!tt.ptr<f32>>) {
      %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %7 = tt.addptr %6, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %8 = arith.cmpi ne, %arg4, %c0_i32 : i32
      %9 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
        %10 = tt.addptr %7, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %10 : tensor<16x!tt.ptr<f32>>
      } else {
        scf.yield %7 : tensor<16x!tt.ptr<f32>>
      }
      scf.yield %9 : tensor<16x!tt.ptr<f32>>
    } else {
      %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %7 = tt.addptr %6, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      scf.yield %7 : tensor<16x!tt.ptr<f32>>
    }
    %3 = tt.load %2 : tensor<16x!tt.ptr<f32>>
    %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %5, %3 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
