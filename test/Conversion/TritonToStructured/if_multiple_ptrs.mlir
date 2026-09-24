// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// IR of the following program:
// @triton.jit
// def multiple_ptrs(
//     x_ptr,
//     y_ptr,
//     z_ptr,
//     x_out_ptr,
//     y_out_ptr,
//     z_out_ptr,
//     stride_row,
//     stride_col,
//     x_row_offset,
//     y_row_offset,
//     z_row_offset,
//     x_col_offset,
//     y_col_offset,
//     z_col_offset,
//     use_x,
//     use_y,
//     use_z,
//     BLOCK_SIZE_ROW: tl.constexpr,
//     BLOCK_SIZE_COL: tl.constexpr,
// ):
//     row_offsets = tl.arange(0, BLOCK_SIZE_ROW)[:, None]
//     col_offsets = tl.arange(0, BLOCK_SIZE_COL)[None, :]
//     out_offsets = row_offsets * stride_row + col_offsets * stride_col
//     if use_x:
//         offsets = (
//             x_row_offset
//             + row_offsets * stride_row
//             + x_col_offset
//             + col_offsets * stride_col
//         )
//         # in_ptrs is unstructured in one branch, this should make all other branches unstructured
//         in_ptrs = x_ptr + offsets // 2
//         out_ptrs = x_out_ptr + out_offsets
//     elif use_y:
//         offsets = (
//             y_row_offset
//             + row_offsets * stride_row
//             + y_col_offset
//             + col_offsets * stride_col
//         )
//         in_ptrs = y_ptr + offsets
//         out_ptrs = y_out_ptr + out_offsets
//     else:
//         offsets = (
//             z_row_offset
//             + row_offsets * stride_row
//             + z_col_offset
//             + col_offsets * stride_col
//         )
//         in_ptrs = z_ptr + offsets
//         out_ptrs = z_out_ptr + out_offsets
//
//     vals = tl.load(in_ptrs)
//     tl.store(out_ptrs, vals)
//
// We have one branch that is unstructured, currently we bail out entirely.

// CHECK-NOT: tts

module {
  tt.func public @multiple_ptrs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16x16xi32>
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %3 = tt.splat %arg6 : i32 -> tensor<16x1xi32>
    %4 = arith.muli %1, %3 : tensor<16x1xi32>
    %5 = tt.broadcast %4 : tensor<16x1xi32> -> tensor<16x16xi32>
    %6 = tt.broadcast %2 : tensor<1x16xi32> -> tensor<16x16xi32>
    %7 = arith.addi %5, %6 : tensor<16x16xi32>
    %8 = arith.cmpi ne, %arg13, %c0_i32 : i32
    %9:2 = scf.if %8 -> (tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>) {
      %11 = tt.splat %arg7 : i32 -> tensor<16x1xi32>
      %12 = arith.addi %11, %4 : tensor<16x1xi32>
      %13 = tt.splat %arg10 : i32 -> tensor<16x1xi32>
      %14 = arith.addi %12, %13 : tensor<16x1xi32>
      %15 = tt.broadcast %14 : tensor<16x1xi32> -> tensor<16x16xi32>
      %16 = arith.addi %15, %6 : tensor<16x16xi32>
      %17 = arith.divsi %16, %cst : tensor<16x16xi32>
      %18 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
      %19 = tt.addptr %18, %17 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
      %20 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
      %21 = tt.addptr %20, %7 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
      scf.yield %19, %21 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>> // unstructured, structured
    } else {
      %11 = arith.cmpi ne, %arg14, %c0_i32 : i32
      %12:2 = scf.if %11 -> (tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>) {
        %13 = tt.splat %arg8 : i32 -> tensor<16x1xi32>
        %14 = arith.addi %13, %4 : tensor<16x1xi32>
        %15 = tt.splat %arg11 : i32 -> tensor<16x1xi32>
        %16 = arith.addi %14, %15 : tensor<16x1xi32>
        %17 = tt.broadcast %16 : tensor<16x1xi32> -> tensor<16x16xi32>
        %18 = arith.addi %17, %6 : tensor<16x16xi32>
        %19 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
        %20 = tt.addptr %19, %18 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %21 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
        %22 = tt.addptr %21, %7 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        scf.yield %20, %22 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
      } else {
        %13 = tt.splat %arg9 : i32 -> tensor<16x1xi32>
        %14 = arith.addi %13, %4 : tensor<16x1xi32>
        %15 = tt.splat %arg12 : i32 -> tensor<16x1xi32>
        %16 = arith.addi %14, %15 : tensor<16x1xi32>
        %17 = tt.broadcast %16 : tensor<16x1xi32> -> tensor<16x16xi32>
        %18 = arith.addi %17, %6 : tensor<16x16xi32>
        %19 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
        %20 = tt.addptr %19, %18 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        %21 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
        %22 = tt.addptr %21, %7 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
        scf.yield %20, %22 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
      }
      scf.yield %12#0, %12#1 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16x!tt.ptr<f32>>
    }
    %10 = tt.load %9#0 : tensor<16x16x!tt.ptr<f32>>
    tt.store %9#1, %10 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}
