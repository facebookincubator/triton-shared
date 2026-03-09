// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
    %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<128x1xi32>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<i32>>
    %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %6 = tt.load %5 : tensor<128x!tt.ptr<i32>>
    %7 = tt.join %3, %6 : tensor<128xi32> -> tensor<128x2xi32>
    %8 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %9 = arith.muli %8, %cst : tensor<128x1xi32>
    %10 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>>
    %11 = tt.addptr %10, %9 : tensor<128x1x!tt.ptr<i32>>, tensor<128x1xi32>
    %12 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %14 = tt.broadcast %11 : tensor<128x1x!tt.ptr<i32>> -> tensor<128x2x!tt.ptr<i32>>
    %15 = tt.broadcast %13 : tensor<1x2xi32> -> tensor<128x2xi32>
    %16 = tt.addptr %14, %15 : tensor<128x2x!tt.ptr<i32>>, tensor<128x2xi32>
    tt.store %16, %7 : tensor<128x2x!tt.ptr<i32>>
    tt.return
  }
}


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},                 %[[ARG1:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},                 %[[ARG2:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},                 %[[ARG3:.*]]: i32,                 %[[ARG4:.*]]: i32,                 %[[ARG5:.*]]: i32,                 %[[ARG6:.*]]: i32,                 %[[ARG7:.*]]: i32,                 %[[ARG8:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<128x1xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_0]] : tensor<128x1xi32>) -> tensor<128x1xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_1]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<128x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_0]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>) outs(%[[SPLAT_0]] : tensor<128x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !tt.ptr<i32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_1]], %[[VAL_2]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<i32>
// CHECK:           } -> tensor<128x!tt.ptr<i32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_1]] : tensor<128x!tt.ptr<i32>>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<128x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_0]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>) outs(%[[SPLAT_1]] : tensor<128x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !tt.ptr<i32>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_4]], %[[VAL_5]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<i32>
// CHECK:           } -> tensor<128x!tt.ptr<i32>>
// CHECK:           %[[LOAD_1:.*]] = tt.load %[[GENERIC_2]] : tensor<128x!tt.ptr<i32>>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<128x2xi32>
// CHECK:           %[[INSERT_SLICE_0:.*]] = tensor.insert_slice %[[LOAD_0]] into %[[EMPTY_2]][0, 0] [128, 1] [1, 1] : tensor<128xi32> into tensor<128x2xi32>
// CHECK:           %[[INSERT_SLICE_1:.*]] = tensor.insert_slice %[[LOAD_1]] into %[[INSERT_SLICE_0]][0, 1] [128, 1] [1, 1] : tensor<128xi32> into tensor<128x2xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]], %[[FILL_0]] : tensor<128x1xi32>, tensor<128x1xi32>) outs(%[[EXPAND_SHAPE_0]] : tensor<128x1xi32>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32):
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:             linalg.yield %[[MULI_0]] : i32
// CHECK:           } -> tensor<128x1xi32>
// CHECK:           %[[SPLAT_2:.*]] = tensor.splat %[[ARG2]] : tensor<128x1x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_2]], %[[GENERIC_3]] : tensor<128x1x!tt.ptr<i32>>, tensor<128x1xi32>) outs(%[[SPLAT_2]] : tensor<128x1x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: !tt.ptr<i32>, %[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_2:.*]] = tt.addptr %[[VAL_10]], %[[VAL_11]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_2]] : !tt.ptr<i32>
// CHECK:           } -> tensor<128x1x!tt.ptr<i32>>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<2xi32>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_3]] : tensor<2xi32>) {
// CHECK:           ^bb0(%[[VAL_13:.*]]: i32):
// CHECK:             %[[INDEX_1:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_1]] : i32
// CHECK:           } -> tensor<2xi32>
// CHECK:           %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[GENERIC_5]] {{\[\[}}0, 1]] output_shape [1, 2] : tensor<2xi32> into tensor<1x2xi32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<128x2x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_4]] : tensor<128x1x!tt.ptr<i32>>) outs(%[[EMPTY_4]] : tensor<128x2x!tt.ptr<i32>>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_14:.*]]: !tt.ptr<i32>, %[[VAL_15:.*]]: !tt.ptr<i32>):
// CHECK:             linalg.yield %[[VAL_14]] : !tt.ptr<i32>
// CHECK:           } -> tensor<128x2x!tt.ptr<i32>>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<128x2xi32>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]] : tensor<1x2xi32>) outs(%[[EMPTY_5]] : tensor<128x2xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_16]] : i32
// CHECK:           } -> tensor<128x2xi32>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_6]], %[[GENERIC_7]] : tensor<128x2x!tt.ptr<i32>>, tensor<128x2xi32>) outs(%[[GENERIC_6]] : tensor<128x2x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_18:.*]]: !tt.ptr<i32>, %[[VAL_19:.*]]: i32, %[[VAL_20:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_3:.*]] = tt.addptr %[[VAL_18]], %[[VAL_19]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_3]] : !tt.ptr<i32>
// CHECK:           } -> tensor<128x2x!tt.ptr<i32>>
// CHECK:           tt.store %[[GENERIC_8]], %[[INSERT_SLICE_1]] : tensor<128x2x!tt.ptr<i32>>
// CHECK:           return
// CHECK:         }

