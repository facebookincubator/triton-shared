// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: tt.func public @tensor_pointer_in_conditional_kernel
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[V0:.*]]:2 = scf.if{{.*}}-> (index, !tt.ptr<f16>) {
// CHECK: scf.yield{{.*}}: index, !tt.ptr<f16>
// CHECK: } else {
// CHECK: scf.yield{{.*}}: index, !tt.ptr<f16>
// CHECK: }
// CHECK-NOT: scf.if{{.*}}tensor<{{.*}}!tt.ptr
// CHECK: tts.make_tptr %{{.*}}#1 to sizes: [16], strides: [%[[C1]]], offsets: [%{{.*}}#0]

module {
  tt.func public @tensor_pointer_in_conditional_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %cst = arith.constant dense<5> : tensor<16xi32>
    %cst_0 = arith.constant dense<3> : tensor<16xi32>
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c10_i32 = arith.constant 10 : i32
    %c2_i32 = arith.constant 2 : i32
    %cst_1 = arith.constant dense<1> : tensor<16xi32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = arith.cmpi ne, %arg4, %c0_i32 : i32
    %2 = scf.if %1 -> (tensor<16x!tt.ptr<f16>>) {
      %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %8 = tt.addptr %7, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %9 = tt.get_program_id x : i32
      %10 = tt.splat %9 : i32 -> tensor<16xi32>
      %11 = tt.addptr %8, %10 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %12 = tt.load %11 : tensor<16x!tt.ptr<f16>>
      %13 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
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
      %7 = tt.get_program_id x : i32
      %8 = arith.remsi %7, %c2_i32 : i32
      %9 = arith.cmpi eq, %8, %c0_i32 : i32
      %10 = scf.if %9 -> (tensor<16x!tt.ptr<f16>>) {
        %11 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
        %12 = tt.addptr %11, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
        %13 = tt.splat %7 : i32 -> tensor<16xi32>
        %14 = tt.addptr %12, %13 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
        %15 = tt.load %14 : tensor<16x!tt.ptr<f16>>
        %16 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
        %17 = tt.addptr %16, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
        tt.store %17, %15 : tensor<16x!tt.ptr<f16>>
        scf.yield %14 : tensor<16x!tt.ptr<f16>>
      } else {
        %11 = arith.remsi %7, %c3_i32 : i32
        %12 = arith.cmpi eq, %11, %c0_i32 : i32
        %13 = scf.if %12 -> (tensor<16x!tt.ptr<f16>>) {
          %14 = tt.load %arg0 : !tt.ptr<f16>
          %15 = arith.extf %14 : f16 to f32
          %16 = arith.fptosi %15 : f32 to i64
          %17 = tt.int_to_ptr %16 : i64 -> !tt.ptr<f16>
          %18 = tt.addptr %arg0, %c2_i32 : !tt.ptr<f16>, i32
          %19 = tt.load %18 : !tt.ptr<f16>
          %20 = arith.extf %19 : f16 to f32
          %21 = arith.fptosi %20 : f32 to i64
          %22 = tt.int_to_ptr %21 : i64 -> !tt.ptr<f16>
          %23 = tt.addptr %arg0, %c3_i32 : !tt.ptr<f16>, i32
          %24 = tt.load %23 : !tt.ptr<f16>
          %25 = arith.extf %24 : f16 to f32
          %26 = arith.fptosi %25 : f32 to i64
          %27 = tt.int_to_ptr %26 : i64 -> !tt.ptr<f16>
          %28 = arith.remsi %7, %c4_i32 : i32
          %29 = arith.cmpi eq, %28, %c0_i32 : i32
          %30 = scf.if %29 -> (tensor<16x!tt.ptr<f16>>) {
            %35 = tt.splat %17 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
            %36 = tt.addptr %35, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
            %37 = tt.splat %7 : i32 -> tensor<16xi32>
            %38 = tt.addptr %36, %37 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
            scf.yield %38 : tensor<16x!tt.ptr<f16>>
          } else {
            %35 = arith.remsi %7, %c5_i32 : i32
            %36 = arith.cmpi eq, %35, %c0_i32 : i32
            %37 = scf.if %36 -> (tensor<16x!tt.ptr<f16>>) {
              %38 = tt.splat %22 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
              %39 = tt.addptr %38, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
              %40 = tt.splat %7 : i32 -> tensor<16xi32>
              %41 = tt.addptr %39, %40 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
              scf.yield %41 : tensor<16x!tt.ptr<f16>>
            } else {
              %38 = tt.splat %27 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
              %39 = tt.addptr %38, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
              %40 = tt.splat %7 : i32 -> tensor<16xi32>
              %41 = tt.addptr %39, %40 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
              scf.yield %41 : tensor<16x!tt.ptr<f16>>
            }
            scf.yield %37 : tensor<16x!tt.ptr<f16>>
          }
          %31 = scf.if %29 -> (tensor<16x!tt.ptr<f16>>) {
            %35 = tt.splat %7 : i32 -> tensor<16xi32>
            %36 = tt.addptr %30, %35 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
            scf.yield %36 : tensor<16x!tt.ptr<f16>>
          } else {
            %35 = arith.remsi %7, %c5_i32 : i32
            %36 = arith.cmpi eq, %35, %c0_i32 : i32
            %37 = scf.if %36 -> (tensor<16x!tt.ptr<f16>>) {
              %38 = tt.get_program_id y : i32
              %39 = tt.splat %38 : i32 -> tensor<16xi32>
              %40 = tt.addptr %30, %39 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
              scf.yield %40 : tensor<16x!tt.ptr<f16>>
            } else {
              %38 = tt.get_program_id z : i32
              %39 = tt.splat %38 : i32 -> tensor<16xi32>
              %40 = tt.addptr %30, %39 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
              scf.yield %40 : tensor<16x!tt.ptr<f16>>
            }
            scf.yield %37 : tensor<16x!tt.ptr<f16>>
          }
          %32 = tt.load %31 : tensor<16x!tt.ptr<f16>>
          %33 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
          %34 = tt.addptr %33, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
          tt.store %34, %32 : tensor<16x!tt.ptr<f16>>
          scf.yield %31 : tensor<16x!tt.ptr<f16>>
        } else {
          %14 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
          %15 = tt.addptr %14, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
          scf.for %arg5 = %c0_i32 to %7 step %c1_i32  : i32 {
            %16 = tt.load %15 : tensor<16x!tt.ptr<f16>>
            %17 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
            %18 = tt.addptr %17, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
            tt.store %18, %16 : tensor<16x!tt.ptr<f16>>
            scf.for %arg6 = %c0_i32 to %7 step %c1_i32  : i32 {
              %19 = tt.load %15 : tensor<16x!tt.ptr<f16>>
              tt.store %18, %19 : tensor<16x!tt.ptr<f16>>
            }
          }
          scf.yield %15 : tensor<16x!tt.ptr<f16>>
        }
        scf.yield %13 : tensor<16x!tt.ptr<f16>>
      }
      scf.yield %10 : tensor<16x!tt.ptr<f16>>
    }
    %3 = scf.if %1 -> (tensor<16x!tt.ptr<f16>>) {
      %7 = tt.addptr %2, %cst_1 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      %8 = tt.load %7 : tensor<16x!tt.ptr<f16>>
      %9 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
      %10 = tt.addptr %9, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
      tt.store %10, %8 : tensor<16x!tt.ptr<f16>>
      scf.yield %7 : tensor<16x!tt.ptr<f16>>
    } else {
      scf.yield %2 : tensor<16x!tt.ptr<f16>>
    }
    %4 = tt.load %3 : tensor<16x!tt.ptr<f16>>
    %5 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<16x!tt.ptr<f16>>
    %6 = tt.addptr %5, %0 : tensor<16x!tt.ptr<f16>>, tensor<16xi32>
    tt.store %6, %4 : tensor<16x!tt.ptr<f16>>
    tt.return
  }
}
