// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: tt.func public @simple_addptr
// CHECK-DAG: %c32 = arith.constant 32 : index
// CHECK-DAG: %c16 = arith.constant 16 : index
// CHECK: %[[OFF:.*]] = arith.select{{.*}}%c16, %c32 : index
// CHECK-NOT: scf.if{{.*}}tensor<{{.*}}!tt.ptr
// CHECK: tts.make_tptr %arg0 to sizes: [16], strides: [{{%c1|1}}], offsets: [%[[OFF]]]{{.*}}: <f32> to tensor<16x!tt.ptr<f32>>

module {
  tt.func public @simple_addptr(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.get_program_id x : i32
    %2 = arith.remsi %1, %c2_i32 : i32
    %3 = arith.cmpi eq, %2, %c0_i32 : i32
    %4 = scf.if %3 -> (!tt.ptr<f32>) {
      %10 = tt.addptr %arg0, %c16_i32 : !tt.ptr<f32>, i32
      scf.yield %10 : !tt.ptr<f32>
    } else {
      %10 = tt.addptr %arg0, %c32_i32 : !tt.ptr<f32>, i32
      scf.yield %10 : !tt.ptr<f32>
    }
    %5 = tt.splat %4 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %6 = tt.addptr %5, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %7 = tt.load %6 : tensor<16x!tt.ptr<f32>>
    %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %9 = tt.addptr %8, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %9, %7 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
