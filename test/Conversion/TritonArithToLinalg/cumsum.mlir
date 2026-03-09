// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

// @triton.jit
// def test_cumsum_op(
//     input_ptr, output_ptr, n_columns
// ):
//     row = tl.program_id(axis=0)
//     row_start = row * n_columns
//     columns = tl.arange(0, 4096)
//     offsets = row_start + columns
//     data = tl.load(input_ptr + offsets)
//     result = tl.cumsum(data, axis=0)
//     tl.store(output_ptr + offsets, result)
//
// ret = triton.compiler.compile(
//     test_cumsum_op,
//     signature=" *fp32,*i32,i32",
//     print_triton_ir_only=True,
// )
// print(ret.asm["ttir"])

module {
  tt.func public @test_cumsum_op_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: i32) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %3 = tt.splat %1 : i32 -> tensor<4096xi32>
    %4 = arith.addi %3, %2 : tensor<4096xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    %7 = tt.load %6 : tensor<4096x!tt.ptr<f32>>
    %8 = "tt.scan"(%7) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %12 = arith.addf %arg3, %arg4 : f32
      tt.scan.return %12 : f32
    }) : (tensor<4096xf32>) -> tensor<4096xf32>
    %9 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4096x!tt.ptr<i32>>
    %10 = tt.addptr %9, %4 : tensor<4096x!tt.ptr<i32>>, tensor<4096xi32>
    %11 = arith.fptosi %8 : tensor<4096xf32> to tensor<4096xi32>
    tt.store %10, %11 : tensor<4096x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @test_cumsum_op_012(
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<i32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32) {
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ARG6]], %[[ARG2]] : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4096xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[MULI_0]] : i32) outs(%[[EMPTY_1]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_0]], %[[GENERIC_0]] : tensor<4096xi32>, tensor<4096xi32>) outs(%[[FILL_0]] : tensor<4096xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_1]] : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>) outs(%[[SPLAT_0]] : tensor<4096x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !tt.ptr<f32>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_4]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_2]] : tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<4096xf32>
// CHECK:           %[[CUMSUM_0:.*]] = ttx.cumsum {axis = 0 : ui32, operandSegmentSizes = array<i32: 1, 1>} ins(%[[LOAD_0]] : tensor<4096xf32>) outs(%[[EMPTY_2]] : tensor<4096xf32>) -> tensor<4096xf32>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<4096x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_1]] : tensor<4096x!tt.ptr<i32>>, tensor<4096xi32>) outs(%[[SPLAT_1]] : tensor<4096x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: !tt.ptr<i32>, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_7]], %[[VAL_8]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<i32>
// CHECK:           } -> tensor<4096x!tt.ptr<i32>>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[CUMSUM_0]] : tensor<4096xf32>) outs(%[[EMPTY_3]] : tensor<4096xi32>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: f32, %[[VAL_11:.*]]: i32):
// CHECK:             %[[FPTOSI_0:.*]] = arith.fptosi %[[VAL_10]] : f32 to i32
// CHECK:             linalg.yield %[[FPTOSI_0]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           tt.store %[[GENERIC_3]], %[[GENERIC_4]] : tensor<4096x!tt.ptr<i32>>
// CHECK:           return
// CHECK:         }

