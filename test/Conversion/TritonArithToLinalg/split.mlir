// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %3 = tt.load %2 : tensor<256x!tt.ptr<i32>>
    %4 = tt.reshape %3 allow_reorder : tensor<256xi32> -> tensor<128x2xi32>
    %outLHS, %outRHS = tt.split %4 : tensor<128x2xi32> -> tensor<128xi32>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %7 = tt.addptr %6, %5 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %7, %outLHS : tensor<128x!tt.ptr<i32>>
    %8 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %9 = tt.addptr %8, %5 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %9, %outRHS : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},                 %[[ARG1:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},                 %[[ARG2:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32},                 %[[ARG3:.*]]: i32,                 %[[ARG4:.*]]: i32,                 %[[ARG5:.*]]: i32,                 %[[ARG6:.*]]: i32,                 %[[ARG7:.*]]: i32,                 %[[ARG8:.*]]: i32) {
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<256xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<256xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<256x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_0]] : tensor<256x!tt.ptr<i32>>, tensor<256xi32>) outs(%[[SPLAT_0]] : tensor<256x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !tt.ptr<i32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_1]], %[[VAL_2]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<i32>
// CHECK:           } -> tensor<256x!tt.ptr<i32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_1]] : tensor<256x!tt.ptr<i32>>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[LOAD_0]] {{\[\[}}0, 1]] output_shape [128, 2] : tensor<256xi32> into tensor<128x2xi32>
// CHECK:           %[[EXTRACT_SLICE_0:.*]] = tensor.extract_slice %[[EXPAND_SHAPE_0]][0, 0] [128, 1] [1, 1] : tensor<128x2xi32> to tensor<128xi32>
// CHECK:           %[[EXTRACT_SLICE_1:.*]] = tensor.extract_slice %[[EXPAND_SHAPE_0]][0, 1] [128, 1] [1, 1] : tensor<128x2xi32> to tensor<128xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_1]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32):
// CHECK:             %[[INDEX_1:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_1]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<128x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_2]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>) outs(%[[SPLAT_1]] : tensor<128x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: !tt.ptr<i32>, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_5]], %[[VAL_6]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<i32>
// CHECK:           } -> tensor<128x!tt.ptr<i32>>
// CHECK:           tt.store %[[GENERIC_3]], %[[EXTRACT_SLICE_0]] : tensor<128x!tt.ptr<i32>>
// CHECK:           %[[SPLAT_2:.*]] = tensor.splat %[[ARG2]] : tensor<128x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_2]], %[[GENERIC_2]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>) outs(%[[SPLAT_2]] : tensor<128x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: !tt.ptr<i32>, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_2:.*]] = tt.addptr %[[VAL_8]], %[[VAL_9]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_2]] : !tt.ptr<i32>
// CHECK:           } -> tensor<128x!tt.ptr<i32>>
// CHECK:           tt.store %[[GENERIC_4]], %[[EXTRACT_SLICE_1]] : tensor<128x!tt.ptr<i32>>
// CHECK:           return
// CHECK:         }

