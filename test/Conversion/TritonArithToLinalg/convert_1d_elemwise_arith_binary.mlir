// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<f32>,
    %b : !tt.ptr<f32>,
    %c : tensor<1024x!tt.ptr<f32>>
  ) -> () {
        %cst = arith.constant dense<true> : tensor<1024xi1>
        // offset calculations
        %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
        // a pointer
        %8 = tt.splat %a : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        // b pointer
        %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        %am = tt.load %9 : tensor<1024x!tt.ptr<f32>>
        %bm = tt.load %19 : tensor<1024x!tt.ptr<f32>>
        %1 = arith.addf %am, %bm : tensor<1024xf32>
        %2 = arith.subf %1, %bm : tensor<1024xf32>
        %3 = arith.mulf %2, %bm : tensor<1024xf32>
        %4 = arith.divf %3, %bm : tensor<1024xf32>
        %5 = arith.cmpf "oeq", %4, %bm : tensor<1024xf32>
        %6 = arith.select %5, %am, %bm : tensor<1024xi1>, tensor<1024xf32>
        tt.store %c, %6 : tensor<1024x!tt.ptr<f32>>
        tt.return
    }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: tensor<1024x!tt.ptr<f32>>, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32) {
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1024xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<1024xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_0]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs(%[[SPLAT_0]] : tensor<1024x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !tt.ptr<f32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_1]], %[[VAL_2]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_0]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs(%[[SPLAT_1]] : tensor<1024x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !tt.ptr<f32>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_4]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<f32>
// CHECK:           } -> tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_1]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[LOAD_1:.*]] = tt.load %[[GENERIC_2]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[LOAD_0]], %[[LOAD_1]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[LOAD_0]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32, %[[VAL_9:.*]]: f32):
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f32
// CHECK:             linalg.yield %[[ADDF_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_3]], %[[LOAD_1]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[GENERIC_3]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: f32, %[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: f32):
// CHECK:             %[[SUBF_0:.*]] = arith.subf %[[VAL_10]], %[[VAL_11]] : f32
// CHECK:             linalg.yield %[[SUBF_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_4]], %[[LOAD_1]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[GENERIC_4]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: f32):
// CHECK:             %[[MULF_0:.*]] = arith.mulf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:             linalg.yield %[[MULF_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_5]], %[[LOAD_1]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[GENERIC_5]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_16:.*]]: f32, %[[VAL_17:.*]]: f32, %[[VAL_18:.*]]: f32):
// CHECK:             %[[DIVF_0:.*]] = arith.divf %[[VAL_16]], %[[VAL_17]] : f32
// CHECK:             linalg.yield %[[DIVF_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<1024xi1>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_6]], %[[LOAD_1]] : tensor<1024xf32>, tensor<1024xf32>) outs(%[[EMPTY_1]] : tensor<1024xi1>) {
// CHECK:           ^bb0(%[[VAL_19:.*]]: f32, %[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: i1):
// CHECK:             %[[CMPF_0:.*]] = arith.cmpf oeq, %[[VAL_19]], %[[VAL_20]] : f32
// CHECK:             linalg.yield %[[CMPF_0]] : i1
// CHECK:           } -> tensor<1024xi1>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_7]], %[[LOAD_0]], %[[LOAD_1]] : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs(%[[LOAD_0]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: i1, %[[VAL_23:.*]]: f32, %[[VAL_24:.*]]: f32, %[[VAL_25:.*]]: f32):
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[VAL_22]], %[[VAL_23]], %[[VAL_24]] : f32
// CHECK:             linalg.yield %[[SELECT_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           tt.store %[[ARG2]], %[[GENERIC_8]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
