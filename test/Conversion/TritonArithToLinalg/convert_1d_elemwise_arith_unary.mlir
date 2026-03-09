// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %f32ptr : !tt.ptr<f32>,
    %intptr : !tt.ptr<i32>,
    %f16ptr : !tt.ptr<f16>,
    %save0 : tensor<1024x!tt.ptr<bf16>>,
    %save1 : tensor<1024x!tt.ptr<f32>>,
    %save2 : tensor<1024x!tt.ptr<f32>>,
    %save3 : tensor<1024x!tt.ptr<f32>>,
    %save4 : tensor<1024x!tt.ptr<f32>>
  ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // f32ptr pointer
    %8 = tt.splat %f32ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // intptr pointer
    %18 = tt.splat %intptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
    // f32ptr pointer
    %28 = tt.splat %f16ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %29 = tt.addptr %28, %0 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %afm = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    %aim = tt.load %19 : tensor<1024x!tt.ptr<i32>>
    %bfm = tt.load %29 : tensor<1024x!tt.ptr<f16>>
    %5 = arith.truncf %afm : tensor<1024xf32> to tensor<1024xbf16>
    %6 = math.exp %afm : tensor<1024xf32>
    %7 = arith.sitofp %aim : tensor<1024xi32> to tensor<1024xf32>
    %10 = arith.extf %bfm : tensor<1024xf16> to tensor<1024xf32>
    %11 = math.sqrt %afm : tensor<1024xf32>
    tt.store %save0, %5 : tensor<1024x!tt.ptr<bf16>>
    tt.store %save1, %6 : tensor<1024x!tt.ptr<f32>>
    tt.store %save2, %7 : tensor<1024x!tt.ptr<f32>>
    tt.store %save3, %10 : tensor<1024x!tt.ptr<f32>>
    tt.store %save4, %11 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<i32>, %[[ARG2:.*]]: !tt.ptr<f16>, %[[ARG3:.*]]: tensor<1024x!tt.ptr<bf16>>, %[[ARG4:.*]]: tensor<1024x!tt.ptr<f32>>, %[[ARG5:.*]]: tensor<1024x!tt.ptr<f32>>, %[[ARG6:.*]]: tensor<1024x!tt.ptr<f32>>, %[[ARG7:.*]]: tensor<1024x!tt.ptr<f32>>, %[[ARG8:.*]]: i32, %[[ARG9:.*]]: i32, %[[ARG10:.*]]: i32, %[[ARG11:.*]]: i32, %[[ARG12:.*]]: i32, %[[ARG13:.*]]: i32) {
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
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<1024x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_0]] : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>) outs(%[[SPLAT_1]] : tensor<1024x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !tt.ptr<i32>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_4]], %[[VAL_5]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<i32>
// CHECK:           } -> tensor<1024x!tt.ptr<i32>>
// CHECK:           %[[SPLAT_2:.*]] = tensor.splat %[[ARG2]] : tensor<1024x!tt.ptr<f16>>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_2]], %[[GENERIC_0]] : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>) outs(%[[SPLAT_2]] : tensor<1024x!tt.ptr<f16>>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: !tt.ptr<f16>, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: !tt.ptr<f16>):
// CHECK:             %[[ADDPTR_2:.*]] = tt.addptr %[[VAL_7]], %[[VAL_8]] : !tt.ptr<f16>, i32
// CHECK:             linalg.yield %[[ADDPTR_2]] : !tt.ptr<f16>
// CHECK:           } -> tensor<1024x!tt.ptr<f16>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_1]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           %[[LOAD_1:.*]] = tt.load %[[GENERIC_2]] : tensor<1024x!tt.ptr<i32>>
// CHECK:           %[[LOAD_2:.*]] = tt.load %[[GENERIC_3]] : tensor<1024x!tt.ptr<f16>>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<1024xbf16>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[LOAD_0]] : tensor<1024xf32>) outs(%[[EMPTY_1]] : tensor<1024xbf16>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: f32, %[[VAL_11:.*]]: bf16):
// CHECK:             %[[TRUNCF_0:.*]] = arith.truncf %[[VAL_10]] : f32 to bf16
// CHECK:             linalg.yield %[[TRUNCF_0]] : bf16
// CHECK:           } -> tensor<1024xbf16>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[LOAD_0]] : tensor<1024xf32>) outs(%[[LOAD_0]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[EXP_0:.*]] = math.exp %[[VAL_12]] : f32
// CHECK:             linalg.yield %[[EXP_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[LOAD_1]] : tensor<1024xi32>) outs(%[[EMPTY_2]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: f32):
// CHECK:             %[[SITOFP_0:.*]] = arith.sitofp %[[VAL_14]] : i32 to f32
// CHECK:             linalg.yield %[[SITOFP_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[LOAD_2]] : tensor<1024xf16>) outs(%[[EMPTY_3]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_16:.*]]: f16, %[[VAL_17:.*]]: f32):
// CHECK:             %[[EXTF_0:.*]] = arith.extf %[[VAL_16]] : f16 to f32
// CHECK:             linalg.yield %[[EXTF_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[LOAD_0]] : tensor<1024xf32>) outs(%[[LOAD_0]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_18:.*]]: f32, %[[VAL_19:.*]]: f32):
// CHECK:             %[[SQRT_0:.*]] = math.sqrt %[[VAL_18]] : f32
// CHECK:             linalg.yield %[[SQRT_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           tt.store %[[ARG3]], %[[GENERIC_4]] : tensor<1024x!tt.ptr<bf16>>
// CHECK:           tt.store %[[ARG4]], %[[GENERIC_5]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.store %[[ARG5]], %[[GENERIC_6]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.store %[[ARG6]], %[[GENERIC_7]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.store %[[ARG7]], %[[GENERIC_8]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }