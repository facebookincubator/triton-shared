// Test all triton ops on tensor of pointers are converted to linalg.generic
// and all triton ops on pointers are converted to ops in the pointer dialect.
// Original triton program:
//    @triton.jit
//    def tensor_ptr(in_ptr0, out_ptr0):
//        ints = tl.load(in_ptr0 + tl.arange(0, 16)).to(tl.int64)
//        ptrs = ints.to(tl.pointer_type(tl.int32))
//        vals = tl.load(ptrs)
//        out_ptrs = out_ptr0 + tl.arange(0, 16)
//        ints_2 = out_ptrs.to(tl.int64) + vals
//        out_ptrs = out_ptr0 + tl.arange(0, 16)
//        out_ptrs_i64 = out_ptrs.to(tl.pointer_type(tl.int64))
//        out_ptrs_i64 += 2
//        out_ptrs_i32 = out_ptrs_i64.to(tl.pointer_type(tl.int32))
//        tl.store(out_ptrs_i32, ints_2.to(tl.int32))

// RUN: triton-shared-opt  --triton-arith-to-linalg="tensor-ptr-to-linalg"  --triton-to-ptr --canonicalize --cse %s | FileCheck %s

module {
  tt.func public @tensor_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
    %4 = arith.extsi %3 : tensor<16xi32> to tensor<16xi64>
    %5 = tt.int_to_ptr %4 : tensor<16xi64> -> tensor<16x!tt.ptr<i32>>
    %6 = tt.load %5 : tensor<16x!tt.ptr<i32>>
    %7 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %8 = tt.addptr %7, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %9 = tt.ptr_to_int %8 : tensor<16x!tt.ptr<i32>> -> tensor<16xi64>
    %10 = arith.extsi %6 : tensor<16xi32> to tensor<16xi64>
    %11 = arith.addi %9, %10 : tensor<16xi64>
    %12 = tt.bitcast %8 : tensor<16x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i64>>
    %13 = tt.addptr %12, %cst : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %14 = tt.bitcast %13 : tensor<16x!tt.ptr<i64>> -> tensor<16x!tt.ptr<i32>>
    %15 = arith.trunci %11 : tensor<16xi64> to tensor<16xi32>
    tt.store %14, %15 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @tensor_ptr(
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<i32>, %[[ARG1:.*]]: !tt.ptr<i32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to !ptr.ptr<#ptr.generic_space>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_1]] : i32) outs(%[[EMPTY_0]] : tensor<16xi32>) -> tensor<16xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[FILL_1:.*]] = tensor.splat %[[UNREALIZED_CONVERSION_CAST_1]] : tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_1]], %[[GENERIC_0]] : tensor<16x!ptr.ptr<#ptr.generic_space>>, tensor<16xi32>) outs(%[[FILL_1]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[TYPE_OFFSET_0:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_2]], %[[TYPE_OFFSET_0]] : i32
// CHECK:             %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[VAL_1]], %[[MULI_0]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_1]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) outs(%[[EMPTY_0]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_5:.*]]: i32):
// CHECK:             %[[FROM_PTR_0:.*]] = tptr.from_ptr %[[VAL_4]] : <#ptr.generic_space> -> memref<1xi32, #ptr.generic_space>
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[FROM_PTR_0]]{{\[}}%[[CONSTANT_0]]] : memref<1xi32, #ptr.generic_space>
// CHECK:             linalg.yield %[[LOAD_0]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<16xi64>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_2]] : tensor<16xi32>) outs(%[[EMPTY_2]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i64):
// CHECK:             %[[EXTSI_0:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
// CHECK:             linalg.yield %[[EXTSI_0]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_3]] : tensor<16xi64>) outs(%[[EMPTY_1]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i64, %[[VAL_9:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[INTTOPTR_0:.*]] = tptr.int_to_ptr %[[VAL_8]] : i64 to <#ptr.generic_space>
// CHECK:             linalg.yield %[[INTTOPTR_0]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_5:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_4]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) outs(%[[EMPTY_0]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_11:.*]]: i32):
// CHECK:             %[[FROM_PTR_1:.*]] = tptr.from_ptr %[[VAL_10]] : <#ptr.generic_space> -> memref<1xi32, #ptr.generic_space>
// CHECK:             %[[LOAD_1:.*]] = memref.load %[[FROM_PTR_1]]{{\[}}%[[CONSTANT_0]]] : memref<1xi32, #ptr.generic_space>
// CHECK:             linalg.yield %[[LOAD_1]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[FILL_2:.*]] = tensor.splat %[[UNREALIZED_CONVERSION_CAST_0]] : tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_6:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_2]], %[[GENERIC_0]] : tensor<16x!ptr.ptr<#ptr.generic_space>>, tensor<16xi32>) outs(%[[FILL_2]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_12:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[TYPE_OFFSET_1:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_13]], %[[TYPE_OFFSET_1]] : i32
// CHECK:             %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[VAL_12]], %[[MULI_1]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_1]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_7:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_6]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) outs(%[[EMPTY_2]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_15:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_16:.*]]: i64):
// CHECK:             %[[PTRTOINT_0:.*]] = tptr.ptr_to_int %[[VAL_15]] : <#ptr.generic_space> to i64
// CHECK:             linalg.yield %[[PTRTOINT_0]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[GENERIC_8:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_5]] : tensor<16xi32>) outs(%[[EMPTY_2]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i64):
// CHECK:             %[[EXTSI_1:.*]] = arith.extsi %[[VAL_17]] : i32 to i64
// CHECK:             linalg.yield %[[EXTSI_1]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[GENERIC_9:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_7]], %[[GENERIC_8]] : tensor<16xi64>, tensor<16xi64>) outs(%[[GENERIC_7]] : tensor<16xi64>) {
// CHECK:           ^bb0(%[[VAL_19:.*]]: i64, %[[VAL_20:.*]]: i64, %[[VAL_21:.*]]: i64):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_19]], %[[VAL_20]] : i64
// CHECK:             linalg.yield %[[ADDI_0]] : i64
// CHECK:           } -> tensor<16xi64>
// CHECK:           %[[GENERIC_10:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_6]], %[[FILL_0]] : tensor<16x!ptr.ptr<#ptr.generic_space>>, tensor<16xi32>) outs(%[[GENERIC_6]] : tensor<16x!ptr.ptr<#ptr.generic_space>>) {
// CHECK:           ^bb0(%[[VAL_22:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_23:.*]]: i32, %[[VAL_24:.*]]: !ptr.ptr<#ptr.generic_space>):
// CHECK:             %[[TYPE_OFFSET_2:.*]] = ptr.type_offset i64 : i32
// CHECK:             %[[MULI_2:.*]] = arith.muli %[[VAL_23]], %[[TYPE_OFFSET_2]] : i32
// CHECK:             %[[PTR_ADD_2:.*]] = ptr.ptr_add %[[VAL_22]], %[[MULI_2]] : !ptr.ptr<#ptr.generic_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_2]] : !ptr.ptr<#ptr.generic_space>
// CHECK:           } -> tensor<16x!ptr.ptr<#ptr.generic_space>>
// CHECK:           %[[GENERIC_11:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_9]] : tensor<16xi64>) outs(%[[EMPTY_0]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: i64, %[[VAL_26:.*]]: i32):
// CHECK:             %[[TRUNCI_0:.*]] = arith.trunci %[[VAL_25]] : i64 to i32
// CHECK:             linalg.yield %[[TRUNCI_0]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_10]], %[[GENERIC_11]] : tensor<16x!ptr.ptr<#ptr.generic_space>>, tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_27:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_28:.*]]: i32):
// CHECK:             %[[FROM_PTR_2:.*]] = tptr.from_ptr %[[VAL_27]] : <#ptr.generic_space> -> memref<1xi32, #ptr.generic_space>
// CHECK:             memref.store %[[VAL_28]], %[[FROM_PTR_2]]{{\[}}%[[CONSTANT_0]]] : memref<1xi32, #ptr.generic_space>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }