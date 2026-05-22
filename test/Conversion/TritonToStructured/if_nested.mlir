// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// CHECK-LABEL: tt.func public @cf_nested
// CHECK: %[[CMP:.*]] = arith.cmpi eq, %arg3
// CHECK: %[[PTR_SELECT:.*]] = arith.select %[[CMP]], %arg0, %arg1 : !tt.ptr<f32>
// CHECK: %[[OFFSET:.*]] = scf.if %[[CMP]] -> (index)
// CHECK-NOT: scf.if {{.*}} -> (tensor<{{.*}}x!tt.ptr
// CHECK: tts.make_tptr %[[PTR_SELECT]] to sizes: [16], strides: [1], offsets: [%[[OFFSET]]]

module {
  tt.func public @cf_nested(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<16> : tensor<16xi32>
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c11_i32 = arith.constant 11 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = arith.cmpi eq, %arg3, %c11_i32 : i32
    %2 = scf.if %1 -> (tensor<16x!tt.ptr<f32>>) {
      %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %7 = tt.addptr %6, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %8 = arith.cmpi ne, %arg4, %c0_i32 : i32
      %9 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
        %14 = arith.muli %arg4, %c16_i32 : i32
        %15 = tt.splat %14 : i32 -> tensor<16xi32>
        %16 = tt.addptr %7, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %16 : tensor<16x!tt.ptr<f32>>
      } else {
        scf.yield %7 : tensor<16x!tt.ptr<f32>>
      }
      %10 = tt.addptr %9, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %11 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
        %14 = arith.muli %arg4, %c32_i32 : i32
        %15 = tt.splat %14 : i32 -> tensor<16xi32>
        %16 = tt.addptr %10, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %17 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
          %20 = arith.muli %arg4, %c16_i32 : i32
          %21 = tt.splat %20 : i32 -> tensor<16xi32>
          %22 = tt.addptr %16, %21 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
          scf.yield %22 : tensor<16x!tt.ptr<f32>>
        } else {
          scf.yield %16 : tensor<16x!tt.ptr<f32>>
        }
        %18 = tt.addptr %17, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %19 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
          %20 = tt.addptr %18, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
          scf.yield %20 : tensor<16x!tt.ptr<f32>>
        } else {
          scf.yield %18 : tensor<16x!tt.ptr<f32>>
        }
        scf.yield %19 : tensor<16x!tt.ptr<f32>>
      } else {
        scf.yield %10 : tensor<16x!tt.ptr<f32>>
      }
      %12 = tt.addptr %11, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %13 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
        %14 = arith.muli %arg4, %c32_i32 : i32
        %15 = tt.splat %14 : i32 -> tensor<16xi32>
        %16 = tt.addptr %12, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %17 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
          %20 = arith.muli %arg4, %c16_i32 : i32
          %21 = tt.splat %20 : i32 -> tensor<16xi32>
          %22 = tt.addptr %16, %21 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
          scf.yield %22 : tensor<16x!tt.ptr<f32>>
        } else {
          scf.yield %16 : tensor<16x!tt.ptr<f32>>
        }
        %18 = tt.addptr %17, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %19 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
          %20 = tt.addptr %18, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
          scf.yield %20 : tensor<16x!tt.ptr<f32>>
        } else {
          scf.yield %18 : tensor<16x!tt.ptr<f32>>
        }
        scf.yield %19 : tensor<16x!tt.ptr<f32>>
      } else {
        scf.yield %12 : tensor<16x!tt.ptr<f32>>
      }
      scf.yield %13 : tensor<16x!tt.ptr<f32>>
    } else {
      %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
      %7 = tt.addptr %6, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %8 = arith.cmpi ne, %arg4, %c0_i32 : i32
      %9 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
        %14 = arith.muli %arg4, %c16_i32 : i32
        %15 = tt.splat %14 : i32 -> tensor<16xi32>
        %16 = tt.addptr %7, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        scf.yield %16 : tensor<16x!tt.ptr<f32>>
      } else {
        scf.yield %7 : tensor<16x!tt.ptr<f32>>
      }
      %10 = tt.addptr %9, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %11 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
        %14 = arith.muli %arg4, %c32_i32 : i32
        %15 = tt.splat %14 : i32 -> tensor<16xi32>
        %16 = tt.addptr %10, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %17 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
          %20 = arith.muli %arg4, %c16_i32 : i32
          %21 = tt.splat %20 : i32 -> tensor<16xi32>
          %22 = tt.addptr %16, %21 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
          scf.yield %22 : tensor<16x!tt.ptr<f32>>
        } else {
          scf.yield %16 : tensor<16x!tt.ptr<f32>>
        }
        %18 = tt.addptr %17, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %19 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
          %20 = tt.addptr %18, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
          scf.yield %20 : tensor<16x!tt.ptr<f32>>
        } else {
          scf.yield %18 : tensor<16x!tt.ptr<f32>>
        }
        scf.yield %19 : tensor<16x!tt.ptr<f32>>
      } else {
        scf.yield %10 : tensor<16x!tt.ptr<f32>>
      }
      %12 = tt.addptr %11, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %13 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
        %14 = arith.muli %arg4, %c32_i32 : i32
        %15 = tt.splat %14 : i32 -> tensor<16xi32>
        %16 = tt.addptr %12, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %17 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
          %20 = arith.muli %arg4, %c16_i32 : i32
          %21 = tt.splat %20 : i32 -> tensor<16xi32>
          %22 = tt.addptr %16, %21 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
          scf.yield %22 : tensor<16x!tt.ptr<f32>>
        } else {
          scf.yield %16 : tensor<16x!tt.ptr<f32>>
        }
        %18 = tt.addptr %17, %cst : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
        %19 = scf.if %8 -> (tensor<16x!tt.ptr<f32>>) {
          %20 = tt.addptr %18, %15 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
          scf.yield %20 : tensor<16x!tt.ptr<f32>>
        } else {
          scf.yield %18 : tensor<16x!tt.ptr<f32>>
        }
        scf.yield %19 : tensor<16x!tt.ptr<f32>>
      } else {
        scf.yield %12 : tensor<16x!tt.ptr<f32>>
      }
      scf.yield %13 : tensor<16x!tt.ptr<f32>>
    }
    %3 = tt.load %2 : tensor<16x!tt.ptr<f32>>
    %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %5, %3 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
