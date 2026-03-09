// RUN: triton-shared-opt --triton-arith-to-linalg="tensor-ptr-to-linalg" --triton-to-ptr --cse --canonicalize %s | FileCheck %s
// Original triton program:
//    @triton.jit
//    def ptr_cat(in_ptr0, out_ptr0, mask_ptr):
//        offsets = tl.arange(0, 16)
//        ptr_0 = in_ptr0 + tl.arange(0, 8)
//        ptr_1 = out_ptr0 + tl.arange(0, 8)
//        ptr = tl.cat(ptr_0, ptr_1, can_reorder=True)
//        ptr_true = ptr + 4 * tl.load(offsets + ptr)
//        ptr_false = ptr + 5 * tl.load(offsets + ptr)
//        masks = tl.load(mask_ptr + offsets)
//        ptr_load = tl.where(masks, ptr_true, ptr_false)
//        a = tl.load(ptr_load + offsets, mask=masks)
//        tl.store(out_ptr0 + offsets, a, mask=masks)

module {
  tt.func public @ptr_cat(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i1>) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<16xi8>
    %cst_0 = arith.constant dense<5> : tensor<16xi32>
    %cst_1 = arith.constant dense<4> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %3 = tt.addptr %2, %1 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %5 = tt.addptr %4, %1 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %6 = tt.cat %3, %5 : tensor<8x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i32>>
    %7 = tt.addptr %6, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %8 = tt.load %7 : tensor<16x!tt.ptr<i32>>
    %9 = arith.muli %8, %cst_1 : tensor<16xi32>
    %10 = tt.addptr %6, %9 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %11 = arith.muli %8, %cst_0 : tensor<16xi32>
    %12 = tt.addptr %6, %11 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %13 = tt.splat %arg2 : !tt.ptr<i1> -> tensor<16x!tt.ptr<i1>>
    %14 = tt.addptr %13, %0 : tensor<16x!tt.ptr<i1>>, tensor<16xi32>
    %15 = tt.bitcast %14 : tensor<16x!tt.ptr<i1>> -> tensor<16x!tt.ptr<i8>>
    %16 = tt.load %15 : tensor<16x!tt.ptr<i8>>
    %17 = arith.cmpi ne, %16, %cst : tensor<16xi8>
    %18 = arith.select %17, %10, %12 : tensor<16xi1>, tensor<16x!tt.ptr<i32>>
    %19 = tt.addptr %18, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %20 = tt.load %19, %17 : tensor<16x!tt.ptr<i32>>
    %21 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %22 = tt.addptr %21, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    tt.store %22, %20, %17 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @ptr_cat(
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<i32>, %[[ARG1:.*]]: !tt.ptr<i32>, %[[ARG2:.*]]: !tt.ptr<i1>, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i8
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 5 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 4 : i32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG2]] : !tt.ptr<i1> to !ptr.ptr<#tptr.default_memory_space>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<i32> to !ptr.ptr<#tptr.default_memory_space>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to !ptr.ptr<#tptr.default_memory_space>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<16xi8>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_2]] : i8) outs(%[[EMPTY_0]] : tensor<16xi8>) -> tensor<16xi8>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_3]] : i32) outs(%[[EMPTY_1]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_4]] : i32) outs(%[[EMPTY_1]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_1]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<8xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_2]] : tensor<8xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32):
// CHECK:             %[[INDEX_1:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[INDEX_1]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_1]] : i32
// CHECK:           } -> tensor<8xi32>
// CHECK:           %[[FILL_3:.*]] = tensor.splat %[[UNREALIZED_CONVERSION_CAST_2]] : tensor<8x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_3]], %[[GENERIC_1]] : tensor<8x!ptr.ptr<#tptr.default_memory_space>>, tensor<8xi32>) outs(%[[FILL_3]] : tensor<8x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_2:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[TYPE_OFFSET_0:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_3]], %[[TYPE_OFFSET_0]] : i32
// CHECK:             %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[VAL_2]], %[[MULI_0]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_0]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<8x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[FILL_4:.*]] = tensor.splat %[[UNREALIZED_CONVERSION_CAST_1]] : tensor<8x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_4]], %[[GENERIC_1]] : tensor<8x!ptr.ptr<#tptr.default_memory_space>>, tensor<8xi32>) outs(%[[FILL_4]] : tensor<8x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[TYPE_OFFSET_1:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_6]], %[[TYPE_OFFSET_1]] : i32
// CHECK:             %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[VAL_5]], %[[MULI_1]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_1]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<8x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[EMPTY_4:.*]] = tensor.empty() : tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[INSERT_SLICE_0:.*]] = tensor.insert_slice %[[GENERIC_2]] into %[[EMPTY_4]][0] [8] [1] : tensor<8x!ptr.ptr<#tptr.default_memory_space>> into tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[INSERT_SLICE_1:.*]] = tensor.insert_slice %[[GENERIC_3]] into %[[INSERT_SLICE_0]][8] [8] [1] : tensor<8x!ptr.ptr<#tptr.default_memory_space>> into tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[INSERT_SLICE_1]], %[[GENERIC_0]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>) outs(%[[INSERT_SLICE_1]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[TYPE_OFFSET_2:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_2:.*]] = arith.muli %[[VAL_9]], %[[TYPE_OFFSET_2]] : i32
// CHECK:             %[[PTR_ADD_2:.*]] = ptr.ptr_add %[[VAL_8]], %[[MULI_2]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_2]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_4]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) outs(%[[EMPTY_1]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_11:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_12:.*]]: i32):
// CHECK:             %[[FROM_PTR_0:.*]] = ptr.from_ptr %[[VAL_11]] : <#tptr.default_memory_space> -> memref<1xi32, #tptr.default_memory_space>
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[FROM_PTR_0]]{{\[}}%[[CONSTANT_1]]] : memref<1xi32, #tptr.default_memory_space>
// CHECK:             linalg.yield %[[LOAD_0]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_5]], %[[FILL_2]] : tensor<16xi32>, tensor<16xi32>) outs(%[[GENERIC_5]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: i32):
// CHECK:             %[[MULI_3:.*]] = arith.muli %[[VAL_13]], %[[VAL_14]] : i32
// CHECK:             linalg.yield %[[MULI_3]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[INSERT_SLICE_1]], %[[GENERIC_6]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>) outs(%[[INSERT_SLICE_1]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_16:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[TYPE_OFFSET_3:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_4:.*]] = arith.muli %[[VAL_17]], %[[TYPE_OFFSET_3]] : i32
// CHECK:             %[[PTR_ADD_3:.*]] = ptr.ptr_add %[[VAL_16]], %[[MULI_4]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_3]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_5]], %[[FILL_1]] : tensor<16xi32>, tensor<16xi32>) outs(%[[GENERIC_5]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_19:.*]]: i32, %[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: i32):
// CHECK:             %[[MULI_5:.*]] = arith.muli %[[VAL_19]], %[[VAL_20]] : i32
// CHECK:             linalg.yield %[[MULI_5]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[GENERIC_9:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[INSERT_SLICE_1]], %[[GENERIC_8]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>) outs(%[[INSERT_SLICE_1]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_23:.*]]: i32, %[[VAL_24:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[TYPE_OFFSET_4:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_6:.*]] = arith.muli %[[VAL_23]], %[[TYPE_OFFSET_4]] : i32
// CHECK:             %[[PTR_ADD_4:.*]] = ptr.ptr_add %[[VAL_22]], %[[MULI_6]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_4]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[FILL_5:.*]] = tensor.splat %[[UNREALIZED_CONVERSION_CAST_0]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_10:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_5]], %[[GENERIC_0]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>) outs(%[[FILL_5]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_26:.*]]: i32, %[[VAL_27:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[TYPE_OFFSET_5:.*]] = ptr.type_offset i1 : i32
// CHECK:             %[[MULI_7:.*]] = arith.muli %[[VAL_26]], %[[TYPE_OFFSET_5]] : i32
// CHECK:             %[[PTR_ADD_5:.*]] = ptr.ptr_add %[[VAL_25]], %[[MULI_7]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_5]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_10]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) outs(%[[EMPTY_0]] : tensor<16xi8>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_29:.*]]: i8):
// CHECK:             %[[FROM_PTR_1:.*]] = ptr.from_ptr %[[VAL_28]] : <#tptr.default_memory_space> -> memref<1xi8, #tptr.default_memory_space>
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[FROM_PTR_1]]{{\[}}%[[CONSTANT_1]]] : memref<1xi8, #tptr.default_memory_space>
// CHECK:             linalg.yield %[[LOAD_1]] : i8
// CHECK:           } -> tensor<16xi8>
// CHECK:           %[[EMPTY_5:.*]] = tensor.empty() : tensor<16xi1>
// CHECK:           %[[GENERIC_12:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_11]], %[[FILL_0]] : tensor<16xi8>, tensor<16xi8>) outs(%[[EMPTY_5]] : tensor<16xi1>) {
// CHECK:           ^bb0(%[[VAL_30:.*]]: i8, %[[VAL_31:.*]]: i8, %[[VAL_32:.*]]: i1):
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_31]] : i8
// CHECK:             linalg.yield %[[CMPI_0]] : i1
// CHECK:           } -> tensor<16xi1>
// CHECK:           %[[GENERIC_13:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_12]], %[[GENERIC_7]], %[[GENERIC_9]] : tensor<16xi1>, tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16x!ptr.ptr<#tptr.default_memory_space>>) outs(%[[GENERIC_7]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_33:.*]]: i1, %[[VAL_34:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_35:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_36:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[SELECT_0:.*]] = arith.select %[[VAL_33]], %[[VAL_34]], %[[VAL_35]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:             linalg.yield %[[SELECT_0]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_14:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_13]], %[[GENERIC_0]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>) outs(%[[GENERIC_13]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_37:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_38:.*]]: i32, %[[VAL_39:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[TYPE_OFFSET_6:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_8:.*]] = arith.muli %[[VAL_38]], %[[TYPE_OFFSET_6]] : i32
// CHECK:             %[[PTR_ADD_6:.*]] = ptr.ptr_add %[[VAL_37]], %[[MULI_8]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_6]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_15:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_14]], %[[GENERIC_12]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi1>) outs(%[[EMPTY_1]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_40:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_41:.*]]: i1, %[[VAL_42:.*]]: i32):
// CHECK:             %[[FROM_PTR_2:.*]] = ptr.from_ptr %[[VAL_40]] : <#tptr.default_memory_space> -> memref<1xi32, #tptr.default_memory_space>
// CHECK:             %[[IF_0:.*]] = scf.if %[[VAL_41]] -> (i32) {
// CHECK:               %[[LOAD_2:.*]] = memref.load %[[FROM_PTR_2]]{{\[}}%[[CONSTANT_1]]] : memref<1xi32, #tptr.default_memory_space>
// CHECK:               scf.yield %[[LOAD_2]] : i32
// CHECK:             } else {
// CHECK:               scf.yield %[[CONSTANT_0]] : i32
// CHECK:             }
// CHECK:             linalg.yield %[[IF_0]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[FILL_6:.*]] = tensor.splat %[[UNREALIZED_CONVERSION_CAST_1]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_16:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_6]], %[[GENERIC_0]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>) outs(%[[FILL_6]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_43:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_44:.*]]: i32, %[[VAL_45:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[TYPE_OFFSET_7:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_9:.*]] = arith.muli %[[VAL_44]], %[[TYPE_OFFSET_7]] : i32
// CHECK:             %[[PTR_ADD_7:.*]] = ptr.ptr_add %[[VAL_43]], %[[MULI_9]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_7]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_16]], %[[GENERIC_15]], %[[GENERIC_12]] : tensor<16x!ptr.ptr<#tptr.default_memory_space>>, tensor<16xi32>, tensor<16xi1>) {
// CHECK:           ^bb0(%[[VAL_46:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_47:.*]]: i32, %[[VAL_48:.*]]: i1):
// CHECK:             scf.if %[[VAL_48]] {
// CHECK:               %[[FROM_PTR_3:.*]] = ptr.from_ptr %[[VAL_46]] : <#tptr.default_memory_space> -> memref<1xi32, #tptr.default_memory_space>
// CHECK:               memref.store %[[VAL_47]], %[[FROM_PTR_3]]{{\[}}%[[CONSTANT_1]]] : memref<1xi32, #tptr.default_memory_space>
// CHECK:             }
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }