// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<bf16>,
        %res : tensor<256x16x!tt.ptr<bf16>>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : i32 -> tensor<32xi32>
    %ws = arith.muli %ct256, %0 : tensor<32xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %m2 = tt.broadcast %1 : tensor<32x1xi32> -> tensor<32x256xi32>
    %100 = tt.expand_dims %m2 {axis = 2 : i32} : tensor<32x256xi32> -> tensor<32x256x1xi32>
    %moff = tt.broadcast %100 : tensor<32x256x1xi32> -> tensor<32x256x16xi32>
    %33 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %34 = tt.expand_dims %33 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %k2 = tt.broadcast %34 : tensor<1x256xi32> -> tensor<32x256xi32>
    %200 = tt.expand_dims %k2 {axis = 2 : i32} : tensor<32x256xi32> -> tensor<32x256x1xi32>
    %koff = tt.broadcast %200 : tensor<32x256x1xi32> -> tensor<32x256x16xi32>
    %23 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %n2 = tt.broadcast %24 : tensor<1x16xi32> -> tensor<256x16xi32>
    %300 = tt.expand_dims %n2 {axis = 0 : i32} : tensor<256x16xi32> -> tensor<1x256x16xi32>
    %noff = tt.broadcast %300 : tensor<1x256x16xi32> -> tensor<32x256x16xi32>
    %mkoff = arith.addi %moff, %koff : tensor<32x256x16xi32>
    %mknoff = arith.addi %mkoff, %noff : tensor<32x256x16xi32>
    // afloat pointer
    %8 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<32x256x16x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %mknoff : tensor<32x256x16x!tt.ptr<bf16>>, tensor<32x256x16xi32>
    %afm = tt.load %9 : tensor<32x256x16x!tt.ptr<bf16>>
    %6 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.cmpf ogt, %arg5, %arg6 : bf16
      %22 = arith.select %21, %arg5, %arg6 : bf16
      tt.reduce.return %22 : bf16
    }) {axis = 0 : i32} : (tensor<32x256x16xbf16>) -> tensor<256x16xbf16>
    tt.store %res, %6 : tensor<256x16x!tt.ptr<bf16>>
    tt.return
    }
}




// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1, d2) -> (0, d1, d2)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<bf16>,                 %[[ARG1:.*]]: tensor<256x16x!tt.ptr<bf16>>,                 %[[ARG2:.*]]: i32,                 %[[ARG3:.*]]: i32,                 %[[ARG4:.*]]: i32,                 %[[ARG5:.*]]: i32,                 %[[ARG6:.*]]: i32,                 %[[ARG7:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0xFF80 : bf16
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 256 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<32xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_0]] : tensor<32xi32>) -> tensor<32xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<32xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_1]] : tensor<32xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<32xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[FILL_0]] : tensor<32xi32>, tensor<32xi32>) outs(%[[GENERIC_0]] : tensor<32xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[MULI_0]] : i32
// CHECK:           } -> tensor<32xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_1]] {{\[\[}}0, 1]] output_shape [32, 1] : tensor<32xi32> into tensor<32x1xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<32x256xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]] : tensor<32x1xi32>) outs(%[[EMPTY_2]] : tensor<32x256xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_4]] : i32
// CHECK:           } -> tensor<32x256xi32>
// CHECK:           %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[GENERIC_2]] {{\[\[}}0], [1, 2]] output_shape [32, 256, 1] : tensor<32x256xi32> into tensor<32x256x1xi32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<32x256x16xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_4]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]] : tensor<32x256x1xi32>) outs(%[[EMPTY_3]] : tensor<32x256x16xi32>) attrs =  {broadcastDims = array<i64: 2>} {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_6]] : i32
// CHECK:           } -> tensor<32x256x16xi32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<256xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_4]] : tensor<256xi32>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i32):
// CHECK:             %[[INDEX_1:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_1]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK:           %[[EXPAND_SHAPE_2:.*]] = tensor.expand_shape %[[GENERIC_4]] {{\[\[}}0, 1]] output_shape [1, 256] : tensor<256xi32> into tensor<1x256xi32>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<32x256xi32>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_2]] : tensor<1x256xi32>) outs(%[[EMPTY_5]] : tensor<32x256xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_9]] : i32
// CHECK:           } -> tensor<32x256xi32>
// CHECK:           %[[EXPAND_SHAPE_3:.*]] = tensor.expand_shape %[[GENERIC_5]] {{\[\[}}0], [1, 2]] output_shape [32, 256, 1] : tensor<32x256xi32> into tensor<32x256x1xi32>
// CHECK:           %[[EMPTY_6:.*]] = tensor.empty() : tensor<32x256x16xi32>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_4]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[EXPAND_SHAPE_3]] : tensor<32x256x1xi32>) outs(%[[EMPTY_6]] : tensor<32x256x16xi32>) attrs =  {broadcastDims = array<i64: 2>} {
// CHECK:           ^bb0(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_11]] : i32
// CHECK:           } -> tensor<32x256x16xi32>
// CHECK:           %[[EMPTY_7:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_7]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_13:.*]]: i32):
// CHECK:             %[[INDEX_2:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[INDEX_2]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_2]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[EXPAND_SHAPE_4:.*]] = tensor.expand_shape %[[GENERIC_7]] {{\[\[}}0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
// CHECK:           %[[EMPTY_8:.*]] = tensor.empty() : tensor<256x16xi32>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_4]] : tensor<1x16xi32>) outs(%[[EMPTY_8]] : tensor<256x16xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_14]] : i32
// CHECK:           } -> tensor<256x16xi32>
// CHECK:           %[[EXPAND_SHAPE_5:.*]] = tensor.expand_shape %[[GENERIC_8]] {{\[\[}}0, 1], [2]] output_shape [1, 256, 16] : tensor<256x16xi32> into tensor<1x256x16xi32>
// CHECK:           %[[EMPTY_9:.*]] = tensor.empty() : tensor<32x256x16xi32>
// CHECK:           %[[GENERIC_9:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_6]], #[[$ATTR_4]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[EXPAND_SHAPE_5]] : tensor<1x256x16xi32>) outs(%[[EMPTY_9]] : tensor<32x256x16xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_16]] : i32
// CHECK:           } -> tensor<32x256x16xi32>
// CHECK:           %[[GENERIC_10:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]], #[[$ATTR_4]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[GENERIC_3]], %[[GENERIC_6]] : tensor<32x256x16xi32>, tensor<32x256x16xi32>) outs(%[[GENERIC_3]] : tensor<32x256x16xi32>) {
// CHECK:           ^bb0(%[[VAL_18:.*]]: i32, %[[VAL_19:.*]]: i32, %[[VAL_20:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_18]], %[[VAL_19]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<32x256x16xi32>
// CHECK:           %[[GENERIC_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]], #[[$ATTR_4]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[GENERIC_10]], %[[GENERIC_9]] : tensor<32x256x16xi32>, tensor<32x256x16xi32>) outs(%[[GENERIC_10]] : tensor<32x256x16xi32>) {
// CHECK:           ^bb0(%[[VAL_21:.*]]: i32, %[[VAL_22:.*]]: i32, %[[VAL_23:.*]]: i32):
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_21]], %[[VAL_22]] : i32
// CHECK:             linalg.yield %[[ADDI_1]] : i32
// CHECK:           } -> tensor<32x256x16xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<32x256x16x!tt.ptr<bf16>>
// CHECK:           %[[GENERIC_12:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]], #[[$ATTR_4]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_11]] : tensor<32x256x16x!tt.ptr<bf16>>, tensor<32x256x16xi32>) outs(%[[SPLAT_0]] : tensor<32x256x16x!tt.ptr<bf16>>) {
// CHECK:           ^bb0(%[[VAL_24:.*]]: !tt.ptr<bf16>, %[[VAL_25:.*]]: i32, %[[VAL_26:.*]]: !tt.ptr<bf16>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_24]], %[[VAL_25]] : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<32x256x16x!tt.ptr<bf16>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_12]] : tensor<32x256x16x!tt.ptr<bf16>>
// CHECK:           %[[EMPTY_10:.*]] = tensor.empty() : tensor<256x16xbf16>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_0]] : bf16) outs(%[[EMPTY_10]] : tensor<256x16xbf16>) -> tensor<256x16xbf16>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[LOAD_0]] : tensor<32x256x16xbf16>) outs(%[[FILL_1]] : tensor<256x16xbf16>) dimensions = [0]
// CHECK:             (%[[VAL_27:.*]]: bf16, %[[VAL_28:.*]]: bf16) {
// CHECK:               %[[MAXIMUMF_0:.*]] = arith.maximumf %[[VAL_27]], %[[VAL_28]] : bf16
// CHECK:               linalg.yield %[[MAXIMUMF_0]] : bf16
// CHECK:             }
// CHECK:           tt.store %[[ARG1]], %[[REDUCE_0]] : tensor<256x16x!tt.ptr<bf16>>
// CHECK:           return
// CHECK:         }

