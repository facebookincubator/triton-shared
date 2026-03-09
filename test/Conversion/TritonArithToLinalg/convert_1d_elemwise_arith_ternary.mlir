// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<i1>,
    %b : !tt.ptr<f32>,
    %c : !tt.ptr<f32>,
    %d : tensor<1024x!tt.ptr<f32>>
  ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // a pointer
    %8 = tt.splat %a : !tt.ptr<i1> -> tensor<1024x!tt.ptr<i1>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i1>>, tensor<1024xi32>
    // b pointer
    %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // c pointer
    %28 = tt.splat %c : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %29 = tt.addptr %28, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %am = tt.load %9 : tensor<1024x!tt.ptr<i1>>
    %bm = tt.load %19 : tensor<1024x!tt.ptr<f32>>
    %cm = tt.load %29 : tensor<1024x!tt.ptr<f32>>
    %10 = arith.select %am, %bm, %cm : tensor<1024xi1>, tensor<1024xf32>
    tt.store %d, %10 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<i1>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: !tt.ptr<f32>, %[[ARG3:.*]]: tensor<1024x!tt.ptr<f32>>, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32, %[[ARG9:.*]]: i32) {
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1024xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<1024xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<1024x!tt.ptr<i1>>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_0]] : tensor<1024x!tt.ptr<i1>>, tensor<1024xi32>) outs(%[[SPLAT_0]] : tensor<1024x!tt.ptr<i1>>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !tt.ptr<i1>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !tt.ptr<i1>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_1]], %[[VAL_2]] : !tt.ptr<i1>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<i1>
// CHECK:           } -> tensor<1024x!tt.ptr<i1>>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_0]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs(%[[SPLAT_1]] : tensor<1024x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !tt.ptr<f32>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_4]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<f32>
// CHECK:           } -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[SPLAT_2:.*]] = tensor.splat %[[ARG2]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_2]], %[[GENERIC_0]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs(%[[SPLAT_2]] : tensor<1024x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: !tt.ptr<f32>, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_2:.*]] = tt.addptr %[[VAL_7]], %[[VAL_8]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_2]] : !tt.ptr<f32>
// CHECK:           } -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_1]] : tensor<1024x!tt.ptr<i1>>
// CHECK:           %[[LOAD_1:.*]] = tt.load %[[GENERIC_2]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[LOAD_2:.*]] = tt.load %[[GENERIC_3]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[LOAD_0]], %[[LOAD_1]], %[[LOAD_2]] : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%[[LOAD_1]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: i1, %[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[VAL_10]], %[[VAL_11]], %[[VAL_12]] : f32
// CHECK:             linalg.yield %[[SELECT_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           tt.store %[[ARG3]], %[[GENERIC_4]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
