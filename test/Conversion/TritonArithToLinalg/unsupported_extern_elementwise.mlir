// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @rand(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i32>>
    %4 = tt.extern_elementwise %3, %0 {libname = "", libpath = "", pure = true, symbol = "some_symbol"} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @rand(
// CHECK-SAME:                    %[[ARG0:.*]]: !tt.ptr<i32>,               %[[ARG1:.*]]: !tt.ptr<i32>,               %[[ARG2:.*]]: i32,               %[[ARG3:.*]]: i32,               %[[ARG4:.*]]: i32,               %[[ARG5:.*]]: i32,               %[[ARG6:.*]]: i32,               %[[ARG7:.*]]: i32) {
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<8xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<8xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<8xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<8x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_0]] : tensor<8x!tt.ptr<i32>>, tensor<8xi32>) outs(%[[SPLAT_0]] : tensor<8x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !tt.ptr<i32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_1]], %[[VAL_2]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<i32>
// CHECK:           } -> tensor<8x!tt.ptr<i32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_1]] : tensor<8x!tt.ptr<i32>>
// CHECK:           %[[EXTERN_ELEMENTWISE_0:.*]] = tt.extern_elementwise %[[LOAD_0]], %[[GENERIC_0]] {libname = "", libpath = "", pure = true, symbol = "some_symbol"} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<8x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_0]] : tensor<8x!tt.ptr<i32>>, tensor<8xi32>) outs(%[[SPLAT_1]] : tensor<8x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !tt.ptr<i32>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_4]], %[[VAL_5]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<i32>
// CHECK:           } -> tensor<8x!tt.ptr<i32>>
// CHECK:           tt.store %[[GENERIC_2]], %[[EXTERN_ELEMENTWISE_0]] : tensor<8x!tt.ptr<i32>>
// CHECK:           return
// CHECK:         }

