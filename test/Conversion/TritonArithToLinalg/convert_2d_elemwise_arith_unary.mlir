// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %f32ptr : !tt.ptr<f32>,
    %intptr : !tt.ptr<i32>,
    %f16ptr : !tt.ptr<f16>,
    %save0 : tensor<128x128x!tt.ptr<bf16>>,
    %save1 : tensor<128x128x!tt.ptr<f32>>,
    %save2 : tensor<128x128x!tt.ptr<f32>>,
    %save3 : tensor<128x128x!tt.ptr<f32>>,
    %save4 : tensor<128x128x!tt.ptr<f32>>
  ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %moff = tt.broadcast %1 : tensor<128x1xi32> -> tensor<128x128xi32>
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %koff = tt.broadcast %4 : tensor<1x128xi32> -> tensor<128x128xi32>
    %mkoff = arith.addi %moff, %koff : tensor<128x128xi32>
    // f32ptr pointer
    %8 = tt.splat %f32ptr : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    %9 = tt.addptr %8, %mkoff : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // intptr pointer
    %18 = tt.splat %intptr : !tt.ptr<i32> -> tensor<128x128x!tt.ptr<i32>>
    %19 = tt.addptr %18, %mkoff : tensor<128x128x!tt.ptr<i32>>, tensor<128x128xi32>
    // f16ptr pointer
    %28 = tt.splat %f16ptr : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>>
    %29 = tt.addptr %28, %mkoff : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
    %afm = tt.load %9 : tensor<128x128x!tt.ptr<f32>>
    %aim = tt.load %19 : tensor<128x128x!tt.ptr<i32>>
    %bfm = tt.load %29 : tensor<128x128x!tt.ptr<f16>>
    %5 = arith.truncf %afm : tensor<128x128xf32> to tensor<128x128xbf16>
    %6 = math.exp %afm : tensor<128x128xf32>
    %7 = arith.sitofp %aim : tensor<128x128xi32> to tensor<128x128xf32>
    %10 = arith.extf %bfm : tensor<128x128xf16> to tensor<128x128xf32>
    %11 = math.sqrt %afm : tensor<128x128xf32>
    tt.store %save0, %5 : tensor<128x128x!tt.ptr<bf16>>
    tt.store %save1, %6 : tensor<128x128x!tt.ptr<f32>>
    tt.store %save2, %7 : tensor<128x128x!tt.ptr<f32>>
    tt.store %save3, %10 : tensor<128x128x!tt.ptr<f32>>
    tt.store %save4, %11 : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<i32>, %[[ARG2:.*]]: !tt.ptr<f16>, %[[ARG3:.*]]: tensor<128x128x!tt.ptr<bf16>>, %[[ARG4:.*]]: tensor<128x128x!tt.ptr<f32>>, %[[ARG5:.*]]: tensor<128x128x!tt.ptr<f32>>, %[[ARG6:.*]]: tensor<128x128x!tt.ptr<f32>>, %[[ARG7:.*]]: tensor<128x128x!tt.ptr<f32>>, %[[ARG8:.*]]: i32, %[[ARG9:.*]]: i32, %[[ARG10:.*]]: i32, %[[ARG11:.*]]: i32, %[[ARG12:.*]]: i32, %[[ARG13:.*]]: i32) {
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<128x128xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]] : tensor<128x1xi32>) outs(%[[EMPTY_1]] : tensor<128x128xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_1]] : i32
// CHECK:           } -> tensor<128x128xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_2]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: i32):
// CHECK:             %[[INDEX_1:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_1]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[GENERIC_2]] {{\[\[}}0, 1]] output_shape [1, 128] : tensor<128xi32> into tensor<1x128xi32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<128x128xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]] : tensor<1x128xi32>) outs(%[[EMPTY_3]] : tensor<128x128xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_4]] : i32
// CHECK:           } -> tensor<128x128xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_1]], %[[GENERIC_3]] : tensor<128x128xi32>, tensor<128x128xi32>) outs(%[[GENERIC_1]] : tensor<128x128xi32>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<128x128xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<128x128x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_4]] : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>) outs(%[[SPLAT_0]] : tensor<128x128x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: !tt.ptr<f32>, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_9]], %[[VAL_10]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } -> tensor<128x128x!tt.ptr<f32>>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<128x128x!tt.ptr<i32>>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_4]] : tensor<128x128x!tt.ptr<i32>>, tensor<128x128xi32>) outs(%[[SPLAT_1]] : tensor<128x128x!tt.ptr<i32>>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: !tt.ptr<i32>, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: !tt.ptr<i32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_12]], %[[VAL_13]] : !tt.ptr<i32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<i32>
// CHECK:           } -> tensor<128x128x!tt.ptr<i32>>
// CHECK:           %[[SPLAT_2:.*]] = tensor.splat %[[ARG2]] : tensor<128x128x!tt.ptr<f16>>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_2]], %[[GENERIC_4]] : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>) outs(%[[SPLAT_2]] : tensor<128x128x!tt.ptr<f16>>) {
// CHECK:           ^bb0(%[[VAL_15:.*]]: !tt.ptr<f16>, %[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: !tt.ptr<f16>):
// CHECK:             %[[ADDPTR_2:.*]] = tt.addptr %[[VAL_15]], %[[VAL_16]] : !tt.ptr<f16>, i32
// CHECK:             linalg.yield %[[ADDPTR_2]] : !tt.ptr<f16>
// CHECK:           } -> tensor<128x128x!tt.ptr<f16>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_5]] : tensor<128x128x!tt.ptr<f32>>
// CHECK:           %[[LOAD_1:.*]] = tt.load %[[GENERIC_6]] : tensor<128x128x!tt.ptr<i32>>
// CHECK:           %[[LOAD_2:.*]] = tt.load %[[GENERIC_7]] : tensor<128x128x!tt.ptr<f16>>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<128x128xbf16>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[LOAD_0]] : tensor<128x128xf32>) outs(%[[EMPTY_4]] : tensor<128x128xbf16>) {
// CHECK:           ^bb0(%[[VAL_18:.*]]: f32, %[[VAL_19:.*]]: bf16):
// CHECK:             %[[TRUNCF_0:.*]] = arith.truncf %[[VAL_18]] : f32 to bf16
// CHECK:             linalg.yield %[[TRUNCF_0]] : bf16
// CHECK:           } -> tensor<128x128xbf16>
// CHECK:           %[[GENERIC_9:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[LOAD_0]] : tensor<128x128xf32>) outs(%[[LOAD_0]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_20:.*]]: f32, %[[VAL_21:.*]]: f32):
// CHECK:             %[[EXP_0:.*]] = math.exp %[[VAL_20]] : f32
// CHECK:             linalg.yield %[[EXP_0]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<128x128xf32>
// CHECK:           %[[GENERIC_10:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[LOAD_1]] : tensor<128x128xi32>) outs(%[[EMPTY_5]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: i32, %[[VAL_23:.*]]: f32):
// CHECK:             %[[SITOFP_0:.*]] = arith.sitofp %[[VAL_22]] : i32 to f32
// CHECK:             linalg.yield %[[SITOFP_0]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           %[[EMPTY_6:.*]] = tensor.empty() : tensor<128x128xf32>
// CHECK:           %[[GENERIC_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[LOAD_2]] : tensor<128x128xf16>) outs(%[[EMPTY_6]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_24:.*]]: f16, %[[VAL_25:.*]]: f32):
// CHECK:             %[[EXTF_0:.*]] = arith.extf %[[VAL_24]] : f16 to f32
// CHECK:             linalg.yield %[[EXTF_0]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           %[[GENERIC_12:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[LOAD_0]] : tensor<128x128xf32>) outs(%[[LOAD_0]] : tensor<128x128xf32>) {
// CHECK:           ^bb0(%[[VAL_26:.*]]: f32, %[[VAL_27:.*]]: f32):
// CHECK:             %[[SQRT_0:.*]] = math.sqrt %[[VAL_26]] : f32
// CHECK:             linalg.yield %[[SQRT_0]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           tt.store %[[ARG3]], %[[GENERIC_8]] : tensor<128x128x!tt.ptr<bf16>>
// CHECK:           tt.store %[[ARG4]], %[[GENERIC_9]] : tensor<128x128x!tt.ptr<f32>>
// CHECK:           tt.store %[[ARG5]], %[[GENERIC_10]] : tensor<128x128x!tt.ptr<f32>>
// CHECK:           tt.store %[[ARG6]], %[[GENERIC_11]] : tensor<128x128x!tt.ptr<f32>>
// CHECK:           tt.store %[[ARG7]], %[[GENERIC_12]] : tensor<128x128x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }

