// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// This stress test verifies that the triton-to-structured pass can handle
// complex nested control flow with loops and conditionals.

// CHECK-LABEL: module {
// CHECK:   tt.func public @if_and_loop_stress_test
// Verify no scf.yield operations yield tensor of pointers
// CHECK-NOT:     scf.yield {{.*}} : tensor<{{.*}}x!tt.ptr
// CHECK-NOT: tts.get_structured_state
// CHECK:     tt.return
// CHECK:   }
// CHECK: }

module {
  tt.func public @if_and_loop_stress_test(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %cst = arith.constant dense<4> : tensor<16x16xi32>
    %c16_i32 = arith.constant 16 : i32
    %cst_0 = arith.constant dense<3> : tensor<16x16xi32>
    %cst_1 = arith.constant dense<2> : tensor<16x16xi32>
    %cst_2 = arith.constant dense<1> : tensor<16x16xi32>
    %cst_3 = arith.constant dense<13> : tensor<16x16xi32>
    %cst_4 = arith.constant dense<12> : tensor<16x16xi32>
    %cst_5 = arith.constant dense<11> : tensor<16x16xi32>
    %cst_6 = arith.constant dense<128> : tensor<1x16xi32>
    %cst_7 = arith.constant dense<64> : tensor<1x16xi32>
    %cst_8 = arith.constant dense<32> : tensor<1x16xi32>
    %cst_9 = arith.constant dense<128> : tensor<16x1xi32>
    %cst_10 = arith.constant dense<64> : tensor<16x1xi32>
    %cst_11 = arith.constant dense<32> : tensor<16x1xi32>
    %c12_i32 = arith.constant 12 : i32
    %c11_i32 = arith.constant 11 : i32
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.get_program_id x : i32
    %2 = arith.remsi %1, %c2_i32 : i32
    %3 = arith.cmpi eq, %2, %c1_i32 : i32
    %4 = scf.if %3 -> (!tt.ptr<f16>) {
      %17 = tt.addptr %arg0, %c10_i32 : !tt.ptr<f16>, i32
      scf.yield %17 : !tt.ptr<f16>
    } else {
      %17 = arith.cmpi eq, %2, %c0_i32 : i32
      %18 = scf.if %17 -> (!tt.ptr<f16>) {
        %19 = tt.addptr %arg1, %c11_i32 : !tt.ptr<f16>, i32
        scf.yield %19 : !tt.ptr<f16>
      } else {
        %19 = tt.addptr %arg2, %c12_i32 : !tt.ptr<f16>, i32
        scf.yield %19 : !tt.ptr<f16>
      }
      scf.yield %18 : !tt.ptr<f16>
    }
    %5 = arith.remsi %arg4, %c2_i32 : i32
    %6 = arith.cmpi eq, %5, %c0_i32 : i32
    %7 = scf.if %6 -> (tensor<16x1xi32>) {
      %17 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
      %18 = arith.muli %17, %cst_11 : tensor<16x1xi32>
      scf.yield %18 : tensor<16x1xi32>
    } else {
      %17 = arith.cmpi eq, %2, %c0_i32 : i32
      %18 = scf.if %17 -> (tensor<16x1xi32>) {
        %19 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
        %20 = arith.muli %19, %cst_10 : tensor<16x1xi32>
        scf.yield %20 : tensor<16x1xi32>
      } else {
        %19 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
        %20 = arith.muli %19, %cst_9 : tensor<16x1xi32>
        scf.yield %20 : tensor<16x1xi32>
      }
      scf.yield %18 : tensor<16x1xi32>
    }
    %8 = scf.if %6 -> (tensor<1x16xi32>) {
      %17 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
      %18 = arith.muli %17, %cst_8 : tensor<1x16xi32>
      scf.yield %18 : tensor<1x16xi32>
    } else {
      %17 = arith.cmpi eq, %2, %c0_i32 : i32
      %18 = scf.if %17 -> (tensor<1x16xi32>) {
        %19 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        %20 = arith.muli %19, %cst_7 : tensor<1x16xi32>
        scf.yield %20 : tensor<1x16xi32>
      } else {
        %19 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
        %20 = arith.muli %19, %cst_6 : tensor<1x16xi32>
        scf.yield %20 : tensor<1x16xi32>
      }
      scf.yield %18 : tensor<1x16xi32>
    }
    %9 = scf.if %6 -> (tensor<16x16x!tt.ptr<f16>>) {
      %17 = tt.splat %4 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>>
      %18 = tt.addptr %17, %7 : tensor<16x1x!tt.ptr<f16>>, tensor<16x1xi32>
      %19 = tt.broadcast %18 : tensor<16x1x!tt.ptr<f16>> -> tensor<16x16x!tt.ptr<f16>>
      %20 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<16x16xi32>
      %21 = tt.addptr %19, %20 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
      %22 = tt.addptr %21, %cst_5 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
      scf.yield %22 : tensor<16x16x!tt.ptr<f16>>
    } else {
      %17 = arith.cmpi eq, %2, %c0_i32 : i32
      %18 = scf.if %17 -> (tensor<16x16x!tt.ptr<f16>>) {
        %19 = tt.splat %4 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>>
        %20 = tt.addptr %19, %7 : tensor<16x1x!tt.ptr<f16>>, tensor<16x1xi32>
        %21 = tt.broadcast %20 : tensor<16x1x!tt.ptr<f16>> -> tensor<16x16x!tt.ptr<f16>>
        %22 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<16x16xi32>
        %23 = tt.addptr %21, %22 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
        %24 = tt.addptr %23, %cst_4 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
        scf.yield %24 : tensor<16x16x!tt.ptr<f16>>
      } else {
        %19 = tt.splat %4 : !tt.ptr<f16> -> tensor<16x1x!tt.ptr<f16>>
        %20 = tt.addptr %19, %7 : tensor<16x1x!tt.ptr<f16>>, tensor<16x1xi32>
        %21 = tt.broadcast %20 : tensor<16x1x!tt.ptr<f16>> -> tensor<16x16x!tt.ptr<f16>>
        %22 = tt.broadcast %8 : tensor<1x16xi32> -> tensor<16x16xi32>
        %23 = tt.addptr %21, %22 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
        %24 = tt.addptr %23, %cst_3 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
        scf.yield %24 : tensor<16x16x!tt.ptr<f16>>
      }
      scf.yield %18 : tensor<16x16x!tt.ptr<f16>>
    }
    %10 = scf.for %arg5 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg6 = %9) -> (tensor<16x16x!tt.ptr<f16>>)  : i32 {
      %17 = tt.load %arg6 : tensor<16x16x!tt.ptr<f16>>
      tt.store %arg6, %17 : tensor<16x16x!tt.ptr<f16>>
      %18 = tt.splat %arg5 : i32 -> tensor<16x16xi32>
      %19 = tt.addptr %arg6, %18 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
      %20 = scf.for %arg7 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg8 = %19) -> (tensor<16x16x!tt.ptr<f16>>)  : i32 {
        %21 = tt.load %arg8 : tensor<16x16x!tt.ptr<f16>>
        tt.store %arg8, %21 : tensor<16x16x!tt.ptr<f16>>
        %22 = tt.splat %arg7 : i32 -> tensor<16x16xi32>
        %23 = tt.addptr %arg8, %22 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
        %24 = arith.remsi %arg7, %c2_i32 : i32
        %25 = arith.cmpi eq, %24, %c0_i32 : i32
        %26 = scf.if %25 -> (tensor<16x16x!tt.ptr<f16>>) {
          %28 = arith.remsi %arg5, %c2_i32 : i32
          %29 = arith.cmpi eq, %28, %c1_i32 : i32
          %30 = scf.if %29 -> (tensor<16x16x!tt.ptr<f16>>) {
            %31 = tt.addptr %23, %cst_0 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
            scf.yield %31 : tensor<16x16x!tt.ptr<f16>>
          } else {
            %31 = scf.if %25 -> (tensor<16x16x!tt.ptr<f16>>) {
              %32 = tt.addptr %23, %cst_1 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
              scf.yield %32 : tensor<16x16x!tt.ptr<f16>>
            } else {
              %32 = tt.addptr %23, %cst : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
              scf.yield %32 : tensor<16x16x!tt.ptr<f16>>
            }
            scf.yield %31 : tensor<16x16x!tt.ptr<f16>>
          }
          scf.yield %30 : tensor<16x16x!tt.ptr<f16>>
        } else {
          scf.yield %23 : tensor<16x16x!tt.ptr<f16>>
        }
        %27 = scf.for %arg9 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg10 = %26) -> (tensor<16x16x!tt.ptr<f16>>)  : i32 {
          %28 = tt.load %arg10 : tensor<16x16x!tt.ptr<f16>>
          tt.store %arg10, %28 : tensor<16x16x!tt.ptr<f16>>
          %29 = tt.splat %arg9 : i32 -> tensor<16x16xi32>
          %30 = tt.addptr %arg10, %29 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
          %31 = arith.remsi %arg9, %c2_i32 : i32
          %32 = arith.cmpi eq, %31, %c0_i32 : i32
          %33 = scf.if %32 -> (tensor<16x16x!tt.ptr<f16>>) {
            %34 = arith.remsi %arg5, %c2_i32 : i32
            %35 = arith.cmpi eq, %34, %c1_i32 : i32
            %36 = scf.if %35 -> (tensor<16x16x!tt.ptr<f16>>) {
              %37 = tt.addptr %30, %cst_0 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
              scf.yield %37 : tensor<16x16x!tt.ptr<f16>>
            } else {
              %37 = scf.if %25 -> (tensor<16x16x!tt.ptr<f16>>) {
                %38 = tt.addptr %30, %cst_1 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
                scf.yield %38 : tensor<16x16x!tt.ptr<f16>>
              } else {
                %38 = tt.addptr %30, %cst : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
                scf.yield %38 : tensor<16x16x!tt.ptr<f16>>
              }
              scf.yield %37 : tensor<16x16x!tt.ptr<f16>>
            }
            scf.yield %36 : tensor<16x16x!tt.ptr<f16>>
          } else {
            scf.yield %30 : tensor<16x16x!tt.ptr<f16>>
          }
          scf.yield %33 : tensor<16x16x!tt.ptr<f16>>
        }
        scf.yield %27 : tensor<16x16x!tt.ptr<f16>>
      }
      scf.yield %20 : tensor<16x16x!tt.ptr<f16>>
    }
    %11 = arith.cmpi eq, %2, %c0_i32 : i32
    %12 = scf.if %11 -> (tensor<16x16x!tt.ptr<f16>>) {
      %17 = tt.addptr %10, %cst_2 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
      scf.yield %17 : tensor<16x16x!tt.ptr<f16>>
    } else {
      scf.yield %10 : tensor<16x16x!tt.ptr<f16>>
    }
    %13 = arith.remsi %2, %c2_i32 : i32
    %14 = arith.cmpi eq, %13, %c1_i32 : i32
    %15 = scf.if %14 -> (tensor<16x16x!tt.ptr<f16>>) {
      %17 = tt.addptr %12, %cst_1 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
      scf.yield %17 : tensor<16x16x!tt.ptr<f16>>
    } else {
      %17 = arith.cmpi eq, %13, %c0_i32 : i32
      %18 = scf.if %17 -> (tensor<16x16x!tt.ptr<f16>>) {
        %19 = tt.addptr %12, %cst_2 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
        scf.yield %19 : tensor<16x16x!tt.ptr<f16>>
      } else {
        %19 = tt.addptr %12, %cst_0 : tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi32>
        scf.yield %19 : tensor<16x16x!tt.ptr<f16>>
      }
      scf.yield %18 : tensor<16x16x!tt.ptr<f16>>
    }
    %16 = tt.load %15 : tensor<16x16x!tt.ptr<f16>>
    tt.store %15, %16 : tensor<16x16x!tt.ptr<f16>>
    tt.return
  }
}
