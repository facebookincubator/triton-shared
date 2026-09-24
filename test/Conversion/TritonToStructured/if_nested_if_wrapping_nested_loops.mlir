// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: @tensor_offset_in_conditional_kernel
// CHECK-NOT: tts.get_structured_state
// Verify no scf.yield operations yield tensor of pointers
// CHECK-NOT:     scf.yield {{.*}} : tensor<{{.*}}x!tt.ptr

module {
  tt.func public @tensor_offset_in_conditional_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<14> : tensor<16xi32>
    %cst_0 = arith.constant dense<8> : tensor<16xi32>
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_1 = arith.constant dense<4> : tensor<16xi32>
    %cst_2 = arith.constant dense<3> : tensor<16xi32>
    %c3_i32 = arith.constant 3 : i32
    %cst_3 = arith.constant dense<2> : tensor<16xi32>
    %c2_i32 = arith.constant 2 : i32
    %cst_4 = arith.constant dense<1> : tensor<16xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi ne, %arg2, %c0_i32 : i32
    %1 = scf.if %0 -> (tensor<16x!tt.ptr<f32>>) {
      %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %5 = tt.addptr %4, %3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      scf.yield %5 : tensor<16x!tt.ptr<f32>>
    } else {
      %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
      %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %5 = tt.addptr %4, %3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      scf.yield %5 : tensor<16x!tt.ptr<f32>>
    }
    %2 = scf.for %arg4 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg5 = %1) -> (tensor<16x!tt.ptr<f32>>)  : i32 {
      %3 = scf.if %0 -> (tensor<16x!tt.ptr<f32>>) {
        %8 = tt.addptr %arg5, %cst_4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %8 : tensor<16x!tt.ptr<f32>>
      } else {
        %8 = tt.addptr %arg5, %cst_3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %8 : tensor<16x!tt.ptr<f32>>
      }
      %4 = tt.load %3 : tensor<16x!tt.ptr<f32>>
      tt.store %3, %4 : tensor<16x!tt.ptr<f32>>
      %5:2 = scf.for %arg6 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg7 = %3, %arg8 = %4) -> (tensor<16x!tt.ptr<f32>>, tensor<16xf32>)  : i32 {
        %8:2 = scf.if %0 -> (tensor<16x!tt.ptr<f32>>, tensor<16xf32>) {
          %9 = tt.addptr %arg7, %cst_2 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
          %10 = tt.get_program_id x : i32
          %11 = arith.remsi %10, %c2_i32 : i32
          %12 = arith.cmpi eq, %11, %c0_i32 : i32
          %13:2 = scf.if %12 -> (tensor<16x!tt.ptr<f32>>, tensor<16xf32>) {
            %14 = tt.addptr %arg7, %cst_0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
            %15 = tt.load %14 : tensor<16x!tt.ptr<f32>>
            tt.store %14, %15 : tensor<16x!tt.ptr<f32>>
            %16 = arith.remsi %10, %c3_i32 : i32
            %17 = arith.cmpi eq, %16, %c0_i32 : i32
            %18:2 = scf.if %17 -> (tensor<16x!tt.ptr<f32>>, tensor<16xf32>) {
              %19 = tt.addptr %arg7, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
              %20:2 = scf.for %arg9 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg10 = %15, %arg11 = %19) -> (tensor<16xf32>, tensor<16x!tt.ptr<f32>>)  : i32 {
                %21 = tt.load %arg11 : tensor<16x!tt.ptr<f32>>
                tt.store %arg11, %21 : tensor<16x!tt.ptr<f32>>
                %22 = tt.addptr %arg11, %cst_4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
                scf.yield %21, %22 : tensor<16xf32>, tensor<16x!tt.ptr<f32>>
              }
              scf.yield %20#1, %20#0 : tensor<16x!tt.ptr<f32>>, tensor<16xf32>
            } else {
              scf.yield %14, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xf32>
            }
            scf.yield %18#0, %18#1 : tensor<16x!tt.ptr<f32>>, tensor<16xf32>
          } else {
            scf.yield %9, %arg8 : tensor<16x!tt.ptr<f32>>, tensor<16xf32>
          }
          scf.yield %13#0, %13#1 : tensor<16x!tt.ptr<f32>>, tensor<16xf32>
        } else {
          %9 = tt.addptr %arg7, %cst_1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
          %10 = tt.load %9 : tensor<16x!tt.ptr<f32>>
          tt.store %9, %10 : tensor<16x!tt.ptr<f32>>
          scf.yield %9, %10 : tensor<16x!tt.ptr<f32>>, tensor<16xf32>
        }
        scf.yield %8#0, %8#1 : tensor<16x!tt.ptr<f32>>, tensor<16xf32>
      }
      %6 = scf.if %0 -> (tensor<16x!tt.ptr<f32>>) {
        %8 = tt.addptr %5#0, %cst_4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %8 : tensor<16x!tt.ptr<f32>>
      } else {
        %8 = tt.addptr %5#0, %cst_3 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %8 : tensor<16x!tt.ptr<f32>>
      }
      %7 = tt.load %6 : tensor<16x!tt.ptr<f32>>
      tt.store %6, %7 : tensor<16x!tt.ptr<f32>>
      scf.yield %6 : tensor<16x!tt.ptr<f32>>
    }
    tt.return
  }
}
