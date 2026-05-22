// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: tt.func public @tensor_pointer_in_conditional_kernel
// CHECK-DAG: %c5 = arith.constant 5 : index
// CHECK-DAG: %c3 = arith.constant 3 : index
// CHECK-DAG: %c1 = arith.constant 1 : index
// CHECK-DAG: %c10_i32 = arith.constant 10 : i32
// CHECK-DAG: %c2_i32 = arith.constant 2 : i32
// CHECK-DAG: %c1_i32 = arith.constant 1 : i32
// CHECK-DAG: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[CMP0:.*]] = arith.cmpi ne, %arg3, %c0_i32 : i32
// CHECK: %[[SEL:.*]] = arith.select %[[CMP0]], %arg0, %arg1 : !tt.ptr<f16>
// CHECK: %[[IF0:.*]] = scf.if %[[CMP0]] -> (index) {
// CHECK:   %[[PID:.*]] = tt.get_program_id x : i32
// CHECK:   %[[IDX:.*]] = arith.index_cast %[[PID]] : i32 to index
// CHECK:   tts.make_tptr %arg0 to sizes: [16], strides: [1], offsets: [%[[IDX]]], {{.*}} : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK:   tts.make_tptr %arg2 to sizes: [16], strides: [1], offsets: [0], {{.*}} : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK:   %[[ADD1:.*]] = arith.addi %[[IDX]], %c1 : index
// CHECK:   %[[REM2:.*]] = arith.remsi %[[PID]], %c2_i32 : i32
// CHECK:   %[[CMP2:.*]] = arith.cmpi eq, %[[REM2]], %c0_i32 : i32
// CHECK:   %[[IF1:.*]] = scf.if %[[CMP2]] -> (index) {
// CHECK:     %[[ADD3:.*]] = arith.addi %[[IDX]], %c3 : index
// CHECK:     tts.make_tptr %arg0 to sizes: [16], strides: [1], offsets: [%[[ADD3]]], {{.*}} : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK:     %[[REM10:.*]] = arith.remsi %[[PID]], %c10_i32 : i32
// CHECK:     %[[CMP10:.*]] = arith.cmpi eq, %[[REM10]], %c1_i32 : i32
// CHECK:     %[[IF2:.*]] = scf.if %[[CMP10]] -> (index) {
// CHECK:       %[[ADD5:.*]] = arith.addi %[[IDX]], %c5 : index
// CHECK:       tts.make_tptr %arg0 to sizes: [16], strides: [1], offsets: [%[[ADD5]]], {{.*}} : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK:       scf.yield %[[ADD5]] : index
// CHECK:     } else {
// CHECK:       scf.yield %[[ADD3]] : index
// CHECK:     }
// CHECK:     scf.yield %[[IF2]] : index
// CHECK:   } else {
// CHECK:     scf.yield %[[ADD1]] : index
// CHECK:   }
// CHECK:   %[[FINAL:.*]] = arith.addi %[[IF1]], %c1 : index
// CHECK:   tts.make_tptr %[[SEL]] to sizes: [16], strides: [%c1], offsets: [%[[FINAL]]], {{.*}} : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK:   tts.make_tptr %arg2 to sizes: [16], strides: [1], offsets: [0], {{.*}} : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK:   scf.yield %[[FINAL]] : index
// CHECK: } else {
// CHECK:   %[[PID:.*]] = tt.get_program_id x : i32
// CHECK:   %[[IDX:.*]] = arith.index_cast %[[PID]] : i32 to index
// CHECK:   tts.make_tptr %arg1 to sizes: [16], strides: [1], offsets: [%[[IDX]]], {{.*}} : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK:   tts.make_tptr %arg2 to sizes: [16], strides: [1], offsets: [0], {{.*}} : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK:   scf.yield %[[IDX]] : index
// CHECK: }
// CHECK-NOT: scf.if{{.*}}tensor<{{.*}}!tt.ptr
// CHECK: tts.make_tptr %[[SEL]] to sizes: [16], strides: [%c1], offsets: [%[[IF0]]], {{.*}} : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK: tts.make_tptr %arg2 to sizes: [16], strides: [1], offsets: [0], {{.*}} : <f16> to tensor<16x!tt.ptr<f16>>
// CHECK: tt.return

module {
  tt.func public @tensor_pointer_in_conditional_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<16xi32>
    %cst_0 = arith.constant dense<3> : tensor<16xi32>
    %c10_i32 = arith.constant 10 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant dense<1> : tensor<16xi32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = arith.cmpi ne, %arg3, %c0_i32 : i32
    %2 = scf.if %1 -> (tensor<16x!tt.ptr<f16>>) {
      %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %8 = tt.addptr %7, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %9 = tt.get_program_id x : i32
      %10 = tt.splat %9 : i32 -> tensor<16xi32>
      %11 = tt.addptr %8, %10 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %12 = tt.load %11 : tensor<16x!tt.ptr<f16>>
      %13 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %14 = tt.addptr %13, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      tt.store %14, %12 : tensor<16x!tt.ptr<f16>>
      %15 = tt.addptr %11, %cst_1 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %16 = arith.remsi %9, %c2_i32 : i32
      %17 = arith.cmpi eq, %16, %c0_i32 : i32
      %18 = scf.if %17 -> (tensor<16x!tt.ptr<f16>>) {
        %19 = tt.addptr %11, %cst_0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
        %20 = tt.load %19 : tensor<16x!tt.ptr<f16>>
        tt.store %14, %20 : tensor<16x!tt.ptr<f16>>
        %21 = arith.remsi %9, %c10_i32 : i32
        %22 = arith.cmpi eq, %21, %c1_i32 : i32
        %23 = scf.if %22 -> (tensor<16x!tt.ptr<f16>>) {
          %24 = tt.addptr %11, %cst : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
          %25 = tt.load %24 : tensor<16x!tt.ptr<f16>>
          tt.store %14, %25 : tensor<16x!tt.ptr<f16>>
          scf.yield %24 : tensor<16x!tt.ptr<f16>>
        } else {
          scf.yield %19 : tensor<16x!tt.ptr<f16>>
        }
        scf.yield %23 : tensor<16x!tt.ptr<f16>>
      } else {
        scf.yield %15 : tensor<16x!tt.ptr<f16>>
      }
      scf.yield %18 : tensor<16x!tt.ptr<f16>>
    } else {
      %7 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %8 = tt.addptr %7, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %9 = tt.get_program_id x : i32
      %10 = tt.splat %9 : i32 -> tensor<16xi32>
      %11 = tt.addptr %8, %10 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %12 = tt.load %11 : tensor<16x!tt.ptr<f16>>
      %13 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %14 = tt.addptr %13, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      tt.store %14, %12 : tensor<16x!tt.ptr<f16>>
      scf.yield %11 : tensor<16x!tt.ptr<f16>>
    }
    %3 = scf.if %1 -> (tensor<16x!tt.ptr<f16>>) {
      %7 = tt.addptr %2, %cst_1 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %8 = tt.load %7 : tensor<16x!tt.ptr<f16>>
      %9 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %10 = tt.addptr %9, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      tt.store %10, %8 : tensor<16x!tt.ptr<f16>>
      scf.yield %7 : tensor<16x!tt.ptr<f16>>
    } else {
      scf.yield %2 : tensor<16x!tt.ptr<f16>>
    }
    %4 = tt.load %3 : tensor<16x!tt.ptr<f16>>
    %5 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
    %6 = tt.addptr %5, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
    tt.store %6, %4 : tensor<16x!tt.ptr<f16>>
    tt.return
  }
}
