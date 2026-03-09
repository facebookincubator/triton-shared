// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

// @triton.jit
// def test(
//     a_ptr, c_ptr, stride_am, stride_an
// ):
//     offs_am = tl.arange(0, 4)
//     offs_an = tl.arange(0, 4)
//     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)
//     a = tl.load(a_ptrs)
//     m = tl.argmax(a, axis=1)
//     tl.store(c_ptr + tl.arange(0, 4), m)
//
// ret = triton.compiler.compile(
//     test,
//     signature=" *fp32,*fp32,i32,i32",
//     print_triton_ir_only=True,
// )

module {
  tt.func public @test_argmax(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<4x1xi32>
    %3 = arith.muli %1, %2 : tensor<4x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x4xi32>
    %6 = arith.muli %4, %5 : tensor<1x4xi32>
    %7 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
    %8 = tt.broadcast %6 : tensor<1x4xi32> -> tensor<4x4xi32>
    %9 = arith.addi %7, %8 : tensor<4x4xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %12 = tt.load %11 : tensor<4x4x!tt.ptr<f32>>
    %13 = tt.broadcast %4 : tensor<1x4xi32> -> tensor<4x4xi32>
    %14:2 = "tt.reduce"(%12, %13) <{axis = 1 : i32}> ({
    ^bb0(%arg4: f32, %arg5: i32, %arg6: f32, %arg7: i32):
      %18 = arith.cmpf oeq, %arg4, %arg6 : f32
      %19 = arith.cmpi slt, %arg5, %arg7 : i32
      %20 = arith.andi %18, %19 : i1
      %21 = arith.cmpf ogt, %arg4, %arg6 : f32
      %22 = arith.ori %21, %20 : i1
      %23 = arith.select %22, %arg4, %arg6 : f32
      %24 = arith.select %22, %arg5, %arg7 : i32
      tt.reduce.return %23, %24 : f32, i32
    }) : (tensor<4x4xf32>, tensor<4x4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
    %15 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %16 = tt.addptr %15, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %17 = arith.sitofp %14#1 : tensor<4xi32> to tensor<4xf32>
    tt.store %16, %17 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}


// -----

// @triton.jit
// def test(
//     a_ptr, c_ptr, stride_am, stride_an
// ):
//     offs_am = tl.arange(0, 4)
//     offs_an = tl.arange(0, 4)
//     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)
//     a = tl.load(a_ptrs)
//     m = tl.argmin(a, axis=1)
//     tl.store(c_ptr + tl.arange(0, 4), m)
//
// ret = triton.compiler.compile(
//     test,
//     signature=" *fp32,*fp32,i32,i32",
//     print_triton_ir_only=True,
// )

module {
  tt.func public @test_argmin(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<4x1xi32>
    %3 = arith.muli %1, %2 : tensor<4x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x4xi32>
    %6 = arith.muli %4, %5 : tensor<1x4xi32>
    %7 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
    %8 = tt.broadcast %6 : tensor<1x4xi32> -> tensor<4x4xi32>
    %9 = arith.addi %7, %8 : tensor<4x4xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x4x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>
    %12 = tt.load %11 : tensor<4x4x!tt.ptr<f32>>
    %13 = tt.broadcast %4 : tensor<1x4xi32> -> tensor<4x4xi32>
    %14:2 = "tt.reduce"(%12, %13) <{axis = 1 : i32}> ({
    ^bb0(%arg4: f32, %arg5: i32, %arg6: f32, %arg7: i32):
      %18 = arith.cmpf oeq, %arg4, %arg6 : f32
      %19 = arith.cmpi slt, %arg5, %arg7 : i32
      %20 = arith.andi %18, %19 : i1
      %21 = arith.cmpf olt, %arg4, %arg6 : f32
      %22 = arith.ori %21, %20 : i1
      %23 = arith.select %22, %arg4, %arg6 : f32
      %24 = arith.select %22, %arg5, %arg7 : i32
      tt.reduce.return %23, %24 : f32, i32
    }) : (tensor<4x4xf32>, tensor<4x4xi32>) -> (tensor<4xf32>, tensor<4xi32>)
    %15 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %16 = tt.addptr %15, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %17 = arith.sitofp %14#1 : tensor<4xi32> to tensor<4xf32>
    tt.store %16, %17 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}




// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:   func.func @test_argmax(
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32, %[[ARG9:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0xFF800000 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [4, 1] : tensor<4xi32> into tensor<4x1xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4x1xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[ARG2]] : i32) outs(%[[EMPTY_1]] : tensor<4x1xi32>) -> tensor<4x1xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]], %[[FILL_0]] : tensor<4x1xi32>, tensor<4x1xi32>) outs(%[[EXPAND_SHAPE_0]] : tensor<4x1xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[MULI_0]] : i32
// CHECK:           } -> tensor<4x1xi32>
// CHECK:           %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<1x4xi32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[ARG3]] : i32) outs(%[[EMPTY_2]] : tensor<1x4xi32>) -> tensor<1x4xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]], %[[FILL_1]] : tensor<1x4xi32>, tensor<1x4xi32>) outs(%[[EXPAND_SHAPE_1]] : tensor<1x4xi32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:             linalg.yield %[[MULI_1]] : i32
// CHECK:           } -> tensor<1x4xi32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_1]] : tensor<4x1xi32>) outs(%[[EMPTY_3]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_7]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_2]] : tensor<1x4xi32>) outs(%[[EMPTY_4]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_9]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_3]], %[[GENERIC_4]] : tensor<4x4xi32>, tensor<4x4xi32>) outs(%[[GENERIC_3]] : tensor<4x4xi32>) {
// CHECK:           ^bb0(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<4x4x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_5]] : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>) outs(%[[SPLAT_0]] : tensor<4x4x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_14:.*]]: !tt.ptr<f32>, %[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_14]], %[[VAL_15]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4x4x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_6]] : tensor<4x4x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]] : tensor<1x4xi32>) outs(%[[EMPTY_5]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_17]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[EMPTY_6:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_6]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[EMPTY_7:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[FILL_3:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_7]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[REDUCE_0:.*]]:2 = linalg.reduce ins(%[[LOAD_0]], %[[GENERIC_7]] : tensor<4x4xf32>, tensor<4x4xi32>) outs(%[[FILL_2]], %[[FILL_3]] : tensor<4xf32>, tensor<4xi32>) dimensions = [1]
// CHECK:             (%[[VAL_19:.*]]: f32, %[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: f32, %[[VAL_22:.*]]: i32) {
// CHECK:               %[[CMPF_0:.*]] = arith.cmpf oeq, %[[VAL_19]], %[[VAL_21]] : f32
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_22]] : i32
// CHECK:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_0]], %[[CMPI_0]] : i1
// CHECK:               %[[CMPF_1:.*]] = arith.cmpf ogt, %[[VAL_19]], %[[VAL_21]] : f32
// CHECK:               %[[ORI_0:.*]] = arith.ori %[[CMPF_1]], %[[ANDI_0]] : i1
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[VAL_19]], %[[VAL_21]] : f32
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[ORI_0]], %[[VAL_20]], %[[VAL_22]] : i32
// CHECK:               linalg.yield %[[SELECT_0]], %[[SELECT_1]] : f32, i32
// CHECK:             }
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<4x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>) outs(%[[SPLAT_1]] : tensor<4x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_23:.*]]: !tt.ptr<f32>, %[[VAL_24:.*]]: i32, %[[VAL_25:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_23]], %[[VAL_24]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_8:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[GENERIC_9:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[REDUCE_0]]#1 : tensor<4xi32>) outs(%[[EMPTY_8]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_26:.*]]: i32, %[[VAL_27:.*]]: f32):
// CHECK:             %[[SITOFP_0:.*]] = arith.sitofp %[[VAL_26]] : i32 to f32
// CHECK:             linalg.yield %[[SITOFP_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           tt.store %[[GENERIC_8]], %[[GENERIC_9]] : tensor<4x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }

// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK: #[[$ATTR_7:.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:   func.func @test_argmin(
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32, %[[ARG9:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0x7F800000 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4xi32>
// CHECK:           %[[EXPAND_SHAPE_0:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [4, 1] : tensor<4xi32> into tensor<4x1xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4x1xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[ARG2]] : i32) outs(%[[EMPTY_1]] : tensor<4x1xi32>) -> tensor<4x1xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_5]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_0]], %[[FILL_0]] : tensor<4x1xi32>, tensor<4x1xi32>) outs(%[[EXPAND_SHAPE_0]] : tensor<4x1xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[MULI_0]] : i32
// CHECK:           } -> tensor<4x1xi32>
// CHECK:           %[[EXPAND_SHAPE_1:.*]] = tensor.expand_shape %[[GENERIC_0]] {{\[\[}}0, 1]] output_shape [1, 4] : tensor<4xi32> into tensor<1x4xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<1x4xi32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[ARG3]] : i32) outs(%[[EMPTY_2]] : tensor<1x4xi32>) -> tensor<1x4xi32>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_5]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]], %[[FILL_1]] : tensor<1x4xi32>, tensor<1x4xi32>) outs(%[[EXPAND_SHAPE_1]] : tensor<1x4xi32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:             linalg.yield %[[MULI_1]] : i32
// CHECK:           } -> tensor<1x4xi32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_6]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_1]] : tensor<4x1xi32>) outs(%[[EMPTY_3]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_7]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_7]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_2]] : tensor<1x4xi32>) outs(%[[EMPTY_4]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_9]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_5]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[GENERIC_3]], %[[GENERIC_4]] : tensor<4x4xi32>, tensor<4x4xi32>) outs(%[[GENERIC_3]] : tensor<4x4xi32>) {
// CHECK:           ^bb0(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<4x4x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_5]], #[[$ATTR_5]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_5]] : tensor<4x4x!tt.ptr<f32>>, tensor<4x4xi32>) outs(%[[SPLAT_0]] : tensor<4x4x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_14:.*]]: !tt.ptr<f32>, %[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_14]], %[[VAL_15]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4x4x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_6]] : tensor<4x4x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<4x4xi32>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_7]], #[[$ATTR_5]]], iterator_types = ["parallel", "parallel"]} ins(%[[EXPAND_SHAPE_1]] : tensor<1x4xi32>) outs(%[[EMPTY_5]] : tensor<4x4xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0(%[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i32):
// CHECK:             linalg.yield %[[VAL_17]] : i32
// CHECK:           } -> tensor<4x4xi32>
// CHECK:           %[[EMPTY_6:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_6]] : tensor<4xf32>) -> tensor<4xf32>
// CHECK:           %[[EMPTY_7:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[FILL_3:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_7]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[REDUCE_0:.*]]:2 = linalg.reduce ins(%[[LOAD_0]], %[[GENERIC_7]] : tensor<4x4xf32>, tensor<4x4xi32>) outs(%[[FILL_2]], %[[FILL_3]] : tensor<4xf32>, tensor<4xi32>) dimensions = [1]
// CHECK:             (%[[VAL_19:.*]]: f32, %[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: f32, %[[VAL_22:.*]]: i32) {
// CHECK:               %[[CMPF_0:.*]] = arith.cmpf oeq, %[[VAL_19]], %[[VAL_21]] : f32
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_22]] : i32
// CHECK:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_0]], %[[CMPI_0]] : i1
// CHECK:               %[[CMPF_1:.*]] = arith.cmpf olt, %[[VAL_19]], %[[VAL_21]] : f32
// CHECK:               %[[ORI_0:.*]] = arith.ori %[[CMPF_1]], %[[ANDI_0]] : i1
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[VAL_19]], %[[VAL_21]] : f32
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[ORI_0]], %[[VAL_20]], %[[VAL_22]] : i32
// CHECK:               linalg.yield %[[SELECT_0]], %[[SELECT_1]] : f32, i32
// CHECK:             }
// CHECK:           %[[SPLAT_1:.*]] = tensor.splat %[[ARG1]] : tensor<4x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]], #[[$ATTR_4]]], iterator_types = ["parallel"]} ins(%[[SPLAT_1]], %[[GENERIC_0]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>) outs(%[[SPLAT_1]] : tensor<4x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_23:.*]]: !tt.ptr<f32>, %[[VAL_24:.*]]: i32, %[[VAL_25:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_1:.*]] = tt.addptr %[[VAL_23]], %[[VAL_24]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_1]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_8:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:           %[[GENERIC_9:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_4]], #[[$ATTR_4]]], iterator_types = ["parallel"]} ins(%[[REDUCE_0]]#1 : tensor<4xi32>) outs(%[[EMPTY_8]] : tensor<4xf32>) {
// CHECK:           ^bb0(%[[VAL_26:.*]]: i32, %[[VAL_27:.*]]: f32):
// CHECK:             %[[SITOFP_0:.*]] = arith.sitofp %[[VAL_26]] : i32 to f32
// CHECK:             linalg.yield %[[SITOFP_0]] : f32
// CHECK:           } -> tensor<4xf32>
// CHECK:           tt.store %[[GENERIC_8]], %[[GENERIC_9]] : tensor<4x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }

