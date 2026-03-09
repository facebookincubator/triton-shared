// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<bf16>,
        %res : !tt.ptr<bf16>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : i32 -> tensor<512xi32>
    %ws = arith.muli %ct256, %0 : tensor<512xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
    %moff = tt.broadcast %1 : tensor<512x1xi32> -> tensor<512x256xi32>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %koff = tt.broadcast %4 : tensor<1x256xi32> -> tensor<512x256xi32>
    %mkoff = arith.addi %moff, %koff : tensor<512x256xi32>
    // afloat pointer
    %8 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<512x256x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %mkoff : tensor<512x256x!tt.ptr<bf16>>, tensor<512x256xi32>
    // res pointer
    %18 = tt.splat %res : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>>
    %19 = tt.addptr %18, %0 : tensor<512x!tt.ptr<bf16>>, tensor<512xi32>
    %afm = tt.load %9 : tensor<512x256x!tt.ptr<bf16>>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 1 : i32} : (tensor<512x256xbf16>) -> tensor<512xbf16>
    tt.store %19, %5 : tensor<512x!tt.ptr<bf16>>
    tt.return
    }
}



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<bf16>,                 %[[ARG1:.*]]: !tt.ptr<bf16>,                 %[[ARG2:.*]]: i32,                 %[[ARG3:.*]]: i32,                 %[[ARG4:.*]]: i32,                 %[[ARG5:.*]]: i32,                 %[[ARG6:.*]]: i32,                 %[[ARG7:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 256 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<512xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_0]] : tensor<512xi32>) -> tensor<512xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<512xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_1]] : tensor<512xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<512xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[FILL_0]] : tensor<512xi32>, tensor<512xi32>) outs(%[[GENERIC_0]] : tensor<512xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[MULI_0]] : i32
// CHECK:           } -> tensor<512xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_1]] {{\[\[}}0, 1]] output_shape [512, 1] : tensor<512xi32> into tensor<512x1xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<512x256xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]] : tensor<512x1xi32>) outs(%[[EMPTY_2]] : tensor<512x256xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_4]] : i32
// CHECK:           } -> tensor<512x256xi32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<256xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_3]] : tensor<256xi32>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32):
// CHECK:             %[[INDEX_1:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_1]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK:           %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[GENERIC_3]] {{\[\[}}0, 1]] output_shape [1, 256] : tensor<256xi32> into tensor<1x256xi32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<512x256xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]] : tensor<1x256xi32>) outs(%[[EMPTY_4]] : tensor<512x256xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_7]] : i32
// CHECK:           } -> tensor<512x256xi32>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_2]], %[[GENERIC_4]] : tensor<512x256xi32>, tensor<512x256xi32>) outs(%[[GENERIC_2]] : tensor<512x256xi32>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<512x256xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<512x256x!tt.ptr<bf16>>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_5]] : tensor<512x256x!tt.ptr<bf16>>, tensor<512x256xi32>) outs(%[[SPLAT_0]] : tensor<512x256x!tt.ptr<bf16>>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: !tt.ptr<bf16>, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: !tt.ptr<bf16>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_12]], %[[VAL_13]] : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<512x256x!tt.ptr<bf16>>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<512x!tt.ptr<bf16>>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_0]] : tensor<512x!tt.ptr<bf16>>, tensor<512xi32>) outs(%[[SPLAT_1]] : tensor<512x!tt.ptr<bf16>>) {
// CHECK:           ^bb0(%[[VAL_15:.*]]: !tt.ptr<bf16>, %[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: !tt.ptr<bf16>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_15]], %[[VAL_16]] : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<512x!tt.ptr<bf16>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_6]] : tensor<512x256x!tt.ptr<bf16>>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<256x512xbf16>
// CHECK:           %[[TRANSPOSE_0:.*]] = linalg.transpose ins(%[[LOAD_0]] : tensor<512x256xbf16>) outs(%[[EMPTY_5]] : tensor<256x512xbf16>) permutation = [1, 0]
// CHECK:           %[[EMPTY_6:.*]] = tensor.empty() : tensor<512xbf16>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_0]] : bf16) outs(%[[EMPTY_6]] : tensor<512xbf16>) -> tensor<512xbf16>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[TRANSPOSE_0]] : tensor<256x512xbf16>) outs(%[[FILL_1]] : tensor<512xbf16>) dimensions = [0]
// CHECK:             (%[[VAL_18:.*]]: bf16, %[[VAL_19:.*]]: bf16) {
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[VAL_18]], %[[VAL_19]] : bf16
// CHECK:               linalg.yield %[[ADDF_0]] : bf16
// CHECK:             }
// CHECK:           tt.store %[[GENERIC_7]], %[[REDUCE_0]] : tensor<512x!tt.ptr<bf16>>
// CHECK:           return
// CHECK:         }

