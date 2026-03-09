// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func public @bcast_kernel_01(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
    %6 = tt.splat %1 : i32 -> tensor<2048xi32>
    %7 = arith.addi %6, %5 : tensor<2048xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %9 = tt.addptr %8, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %10 = tt.load %9 : tensor<32x!tt.ptr<f32>>
    %11 = tt.reshape %10 allow_reorder : tensor<32xf32> -> tensor<1x32xf32>
    %12 = tt.broadcast %11 : tensor<1x32xf32> -> tensor<64x32xf32>
    %13 = tt.reshape %12 allow_reorder : tensor<64x32xf32> -> tensor<2048xf32>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    %15 = tt.addptr %14, %7 : tensor<2048x!tt.ptr<f32>>, tensor<2048xi32>
    tt.store %15, %13 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @bcast_kernel_01(
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 32 : i32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ARG5]], %[[CONSTANT_0]] : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<32xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<32xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<32xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<32xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[MULI_0]] : i32) outs(%[[EMPTY_1]] : tensor<32xi32>) -> tensor<32xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_0]], %[[GENERIC_0]] : tensor<32xi32>, tensor<32xi32>) outs(%[[FILL_0]] : tensor<32xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<32xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_2]] : tensor<2048xi32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32):
// CHECK:             %[[INDEX_1:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_1]] : i32
// CHECK:           } -> tensor<2048xi32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<2048xi32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[MULI_0]] : i32) outs(%[[EMPTY_3]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_1]], %[[GENERIC_2]] : tensor<2048xi32>, tensor<2048xi32>) outs(%[[FILL_1]] : tensor<2048xi32>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32):
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:             linalg.yield %[[ADDI_1]] : i32
// CHECK:           } -> tensor<2048xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<32x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_1]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>) outs(%[[SPLAT_0]] : tensor<32x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: !tt.ptr<f32>, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_8]], %[[VAL_9]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } -> tensor<32x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_4]] : tensor<32x!tt.ptr<f32>>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[LOAD_0]] {{\[\[}}0, 1]] output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<64x32xf32>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]] : tensor<1x32xf32>) outs(%[[EMPTY_4]] : tensor<64x32xf32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: f32):
// CHECK:             linalg.yield %[[VAL_11]] : f32
// CHECK:           } -> tensor<64x32xf32>
// CHECK:           %[[COLLAPSE_SHAPE_0:.*]] = tensor.collapse_shape %[[GENERIC_5]] {{\[\[}}0, 1]] : tensor<64x32xf32> into tensor<2048xf32>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<2048x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_3]] : tensor<2048x!tt.ptr<f32>>, tensor<2048xi32>) outs(%[[SPLAT_1]] : tensor<2048x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_13:.*]]: !tt.ptr<f32>, %[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_13]], %[[VAL_14]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<f32>
// CHECK:           } -> tensor<2048x!tt.ptr<f32>>
// CHECK:           tt.store %[[GENERIC_6]], %[[COLLAPSE_SHAPE_0]] : tensor<2048x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
