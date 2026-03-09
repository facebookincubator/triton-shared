// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>
  )
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c64 = arith.constant 128 : i32
    %1 = tt.splat %c64 : i32 -> tensor<128xi32>
    %2 = arith.muli %0, %1 : tensor<128xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %4 = tt.broadcast %3 : tensor<128x1xi32> -> tensor<128x64xi32>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %7 = tt.broadcast %6 : tensor<1x64xi32> -> tensor<128x64xi32>
    %8 = arith.addi %4, %7 : tensor<128x64xi32>
    %10 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
    %12 = tt.broadcast %11 : tensor<256x1xi32> -> tensor<256x64xi32>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %c256 = arith.constant 256 : i32
    %14 = tt.splat %c256 : i32 -> tensor<64xi32>
    %15 = arith.muli %13, %14 : tensor<64xi32>
    %16 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %17 = tt.broadcast %16 : tensor<1x64xi32> -> tensor<256x64xi32>
    %18 = arith.addi %12, %17 : tensor<256x64xi32>
    %20 = tt.splat %c256 : i32 -> tensor<128xi32>
    %21 = arith.muli %0, %20 : tensor<128xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %23 = tt.broadcast %22 : tensor<128x1xi32> -> tensor<128x256xi32>
    %24 = tt.expand_dims %10 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %25 = tt.broadcast %24 {axis = 0 : i32} : tensor<1x256xi32> -> tensor<128x256xi32>
    %26 = arith.addi %23, %25 : tensor<128x256xi32>
    %30 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    %31 = tt.addptr %30, %8 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %32 = tt.load %31 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<128x64x!tt.ptr<bf16>>
    %40 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x64x!tt.ptr<bf16>>
    %41 = tt.addptr %40, %18 : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>
    %42 = tt.load %41 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x64x!tt.ptr<bf16>>
    %43 = tt.trans %42 {order = array<i32: 1, 0>} : tensor<256x64xbf16> -> tensor<64x256xbf16>
    %50 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    %51 = tt.addptr %50, %26 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %52 = tt.load %51 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<128x256x!tt.ptr<bf16>>
    %60 = tt.dot %32, %43, %52 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
    tt.store %51, %60 : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<bf16>,                 %[[ARG1:.*]]: !tt.ptr<bf16>,                 %[[ARG2:.*]]: !tt.ptr<bf16>,                 %[[ARG3:.*]]: i32,                 %[[ARG4:.*]]: i32,                 %[[ARG5:.*]]: i32,                 %[[ARG6:.*]]: i32,                 %[[ARG7:.*]]: i32,                 %[[ARG8:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 128 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 256 : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_2]] : i32) outs(%[[EMPTY_0]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<64xi32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_2]] : i32) outs(%[[EMPTY_1]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_2]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_3]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[FILL_2]] : tensor<128xi32>, tensor<128xi32>) outs(%[[GENERIC_0]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[MULI_0]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_1]] {{\[\[}}0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<128x64xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]] : tensor<128x1xi32>) outs(%[[EMPTY_4]] : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_4]] : i32
// CHECK:           } -> tensor<128x64xi32>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<64xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_5]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32):
// CHECK:             %[[INDEX_1:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_1]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[GENERIC_3]] {{\[\[}}0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
// CHECK:           %[[EMPTY_6:.*]] = tensor.empty() : tensor<128x64xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]] : tensor<1x64xi32>) outs(%[[EMPTY_6]] : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_7]] : i32
// CHECK:           } -> tensor<128x64xi32>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_2]], %[[GENERIC_4]] : tensor<128x64xi32>, tensor<128x64xi32>) outs(%[[GENERIC_2]] : tensor<128x64xi32>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<128x64xi32>
// CHECK:           %[[EMPTY_7:.*]] = tensor.empty() : tensor<256xi32>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_7]] : tensor<256xi32>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: i32):
// CHECK:             %[[INDEX_2:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_2:.*]] = arith.index_cast %[[INDEX_2]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_2]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK:           %[[EXPAND_SHAPE_2:.*]] = tensor.expand_shape %[[GENERIC_6]] {{\[\[}}0, 1]] output_shape [256, 1] : tensor<256xi32> into tensor<256x1xi32>
// CHECK:           %[[EMPTY_8:.*]] = tensor.empty() : tensor<256x64xi32>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_2]] : tensor<256x1xi32>) outs(%[[EMPTY_8]] : tensor<256x64xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_13]] : i32
// CHECK:           } -> tensor<256x64xi32>
// CHECK:           %[[EMPTY_9:.*]] = tensor.empty() : tensor<64xi32>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_9]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_15:.*]]: i32):
// CHECK:             %[[INDEX_3:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_3:.*]] = arith.index_cast %[[INDEX_3]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_3]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[GENERIC_9:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_8]], %[[FILL_1]] : tensor<64xi32>, tensor<64xi32>) outs(%[[GENERIC_8]] : tensor<64xi32>) {
// CHECK:           ^bb0(%[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i32):
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_16]], %[[VAL_17]] : i32
// CHECK:             linalg.yield %[[MULI_1]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           %[[EXPAND_SHAPE_3:.*]] = tensor.expand_shape %[[GENERIC_9]] {{\[\[}}0, 1]] output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
// CHECK:           %[[EMPTY_10:.*]] = tensor.empty() : tensor<256x64xi32>
// CHECK:           %[[GENERIC_10:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_3]] : tensor<1x64xi32>) outs(%[[EMPTY_10]] : tensor<256x64xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_19:.*]]: i32, %[[VAL_20:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_19]] : i32
// CHECK:           } -> tensor<256x64xi32>
// CHECK:           %[[GENERIC_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_7]], %[[GENERIC_10]] : tensor<256x64xi32>, tensor<256x64xi32>) outs(%[[GENERIC_7]] : tensor<256x64xi32>) {
// CHECK:           ^bb0(%[[VAL_21:.*]]: i32, %[[VAL_22:.*]]: i32, %[[VAL_23:.*]]: i32):
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_21]], %[[VAL_22]] : i32
// CHECK:             linalg.yield %[[ADDI_1]] : i32
// CHECK:           } -> tensor<256x64xi32>
// CHECK:           %[[GENERIC_12:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_0]], %[[FILL_0]] : tensor<128xi32>, tensor<128xi32>) outs(%[[GENERIC_0]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_24:.*]]: i32, %[[VAL_25:.*]]: i32, %[[VAL_26:.*]]: i32):
// CHECK:             %[[MULI_2:.*]] = arith.muli %[[VAL_24]], %[[VAL_25]] : i32
// CHECK:             linalg.yield %[[MULI_2]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[EXPAND_SHAPE_4:.*]] = tensor.expand_shape %[[GENERIC_12]] {{\[\[}}0, 1]] output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
// CHECK:           %[[EMPTY_11:.*]] = tensor.empty() : tensor<128x256xi32>
// CHECK:           %[[GENERIC_13:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_4]] : tensor<128x1xi32>) outs(%[[EMPTY_11]] : tensor<128x256xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_27:.*]]: i32, %[[VAL_28:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_27]] : i32
// CHECK:           } -> tensor<128x256xi32>
// CHECK:           %[[EXPAND_SHAPE_5:.*]] = tensor.expand_shape %[[GENERIC_6]] {{\[\[}}0, 1]] output_shape [1, 256] : tensor<256xi32> into tensor<1x256xi32>
// CHECK:           %[[EMPTY_12:.*]] = tensor.empty() : tensor<128x256xi32>
// CHECK:           %[[GENERIC_14:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_5]] : tensor<1x256xi32>) outs(%[[EMPTY_12]] : tensor<128x256xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_29:.*]]: i32, %[[VAL_30:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_29]] : i32
// CHECK:           } -> tensor<128x256xi32>
// CHECK:           %[[GENERIC_15:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_13]], %[[GENERIC_14]] : tensor<128x256xi32>, tensor<128x256xi32>) outs(%[[GENERIC_13]] : tensor<128x256xi32>) {
// CHECK:           ^bb0(%[[VAL_31:.*]]: i32, %[[VAL_32:.*]]: i32, %[[VAL_33:.*]]: i32):
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_31]], %[[VAL_32]] : i32
// CHECK:             linalg.yield %[[ADDI_2]] : i32
// CHECK:           } -> tensor<128x256xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<128x64x!tt.ptr<bf16>>
// CHECK:           %[[GENERIC_16:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_5]] : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>) outs(%[[SPLAT_0]] : tensor<128x64x!tt.ptr<bf16>>) {
// CHECK:           ^bb0(%[[VAL_34:.*]]: !tt.ptr<bf16>, %[[VAL_35:.*]]: i32, %[[VAL_36:.*]]: !tt.ptr<bf16>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_34]], %[[VAL_35]] : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<128x64x!tt.ptr<bf16>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_16]] : tensor<128x64x!tt.ptr<bf16>>
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<256x64x!tt.ptr<bf16>>
// CHECK:           %[[GENERIC_17:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_11]] : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>) outs(%[[SPLAT_1]] : tensor<256x64x!tt.ptr<bf16>>) {
// CHECK:           ^bb0(%[[VAL_37:.*]]: !tt.ptr<bf16>, %[[VAL_38:.*]]: i32, %[[VAL_39:.*]]: !tt.ptr<bf16>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_37]], %[[VAL_38]] : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<256x64x!tt.ptr<bf16>>
// CHECK:           %[[LOAD_1:.*]] = tt.load %[[GENERIC_17]] : tensor<256x64x!tt.ptr<bf16>>
// CHECK:           %[[EMPTY_13:.*]] = tensor.empty() : tensor<64x256xbf16>
// CHECK:           %[[TRANSPOSE_0:.*]] = linalg.transpose ins(%[[LOAD_1]] : tensor<256x64xbf16>) outs(%[[EMPTY_13]] : tensor<64x256xbf16>) permutation = [1, 0]
// CHECK:           %[[SPLAT_2:.*]] = tensor.splat %[[ARG2]] : tensor<128x256x!tt.ptr<bf16>>
// CHECK:           %[[GENERIC_18:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_2]], %[[GENERIC_15]] : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>) outs(%[[SPLAT_2]] : tensor<128x256x!tt.ptr<bf16>>) {
// CHECK:           ^bb0(%[[VAL_40:.*]]: !tt.ptr<bf16>, %[[VAL_41:.*]]: i32, %[[VAL_42:.*]]: !tt.ptr<bf16>):
// CHECK:             %[[ADDPTR_2:.*]] = tt.addptr %[[VAL_40]], %[[VAL_41]] : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield %[[ADDPTR_2]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<128x256x!tt.ptr<bf16>>
// CHECK:           %[[LOAD_2:.*]] = tt.load %[[GENERIC_18]] : tensor<128x256x!tt.ptr<bf16>>
// CHECK:           %[[EMPTY_14:.*]] = tensor.empty() : tensor<128x256xbf16>
// CHECK:           %[[FILL_3:.*]] = linalg.fill ins(%[[CONSTANT_0]] : bf16) outs(%[[EMPTY_14]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           %[[MATMUL_0:.*]] = linalg.matmul ins(%[[LOAD_0]], %[[TRANSPOSE_0]] : tensor<128x64xbf16>, tensor<64x256xbf16>) outs(%[[FILL_3]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           %[[GENERIC_19:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[LOAD_2]], %[[MATMUL_0]] : tensor<128x256xbf16>, tensor<128x256xbf16>) outs(%[[LOAD_2]] : tensor<128x256xbf16>) {
// CHECK:           ^bb0(%[[VAL_43:.*]]: bf16, %[[VAL_44:.*]]: bf16, %[[VAL_45:.*]]: bf16):
// CHECK:             %[[ADDF_0:.*]] = arith.addf %[[VAL_43]], %[[VAL_44]] : bf16
// CHECK:             linalg.yield %[[ADDF_0]] : bf16
// CHECK:           } -> tensor<128x256xbf16>
// CHECK:           tt.store %[[GENERIC_18]], %[[GENERIC_19]] : tensor<128x256x!tt.ptr<bf16>>
// CHECK:           return
// CHECK:         }

