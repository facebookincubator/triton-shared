// RUN: triton-shared-opt --triton-arith-to-linalg="tensor-ptr-to-linalg"  --triton-to-ptr --canonicalize --cse %s | FileCheck %s

module {
  tt.func @kernel(%a : !tt.ptr<i32>, %b : !tt.ptr<f32>) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>

    // a pointer
    %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>

    // b pointer
    %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>

    %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>

    %am_bitcast = tt.bitcast %am : tensor<1024xi32> -> tensor<1024xf32>
    tt.store %19, %am_bitcast : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<i32>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to !ptr.ptr<#tptr.default_memory_space>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<i32> to !ptr.ptr<#tptr.default_memory_space>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<1024xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<1024xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<1024x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[UNREALIZED_CONVERSION_CAST_1]] : !ptr.ptr<#tptr.default_memory_space>) outs(%[[EMPTY_1]] : tensor<1024x!ptr.ptr<#tptr.default_memory_space>>) -> tensor<1024x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_0]], %[[GENERIC_0]] : tensor<1024x!ptr.ptr<#tptr.default_memory_space>>, tensor<1024xi32>) outs(%[[FILL_0]] : tensor<1024x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[TYPE_OFFSET_0:.*]] = ptr.type_offset i32 : i32
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[VAL_2]], %[[TYPE_OFFSET_0]] : i32
// CHECK:             %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[VAL_1]], %[[MULI_0]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_0]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<1024x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[UNREALIZED_CONVERSION_CAST_0]] : !ptr.ptr<#tptr.default_memory_space>) outs(%[[EMPTY_1]] : tensor<1024x!ptr.ptr<#tptr.default_memory_space>>) -> tensor<1024x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_1]], %[[GENERIC_0]] : tensor<1024x!ptr.ptr<#tptr.default_memory_space>>, tensor<1024xi32>) outs(%[[FILL_1]] : tensor<1024x!ptr.ptr<#tptr.default_memory_space>>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: !ptr.ptr<#tptr.default_memory_space>):
// CHECK:             %[[TYPE_OFFSET_1:.*]] = ptr.type_offset f32 : i32
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[VAL_5]], %[[TYPE_OFFSET_1]] : i32
// CHECK:             %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[VAL_4]], %[[MULI_1]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             linalg.yield %[[PTR_ADD_1]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } -> tensor<1024x!ptr.ptr<#tptr.default_memory_space>>
// CHECK:           %[[GENERIC_3:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_1]] : tensor<1024x!ptr.ptr<#tptr.default_memory_space>>) outs(%[[EMPTY_0]] : tensor<1024xi32>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_8:.*]]: i32):
// CHECK:             %[[FROM_PTR_0:.*]] = ptr.from_ptr %[[VAL_7]] : <#tptr.default_memory_space> -> memref<1xi32, #tptr.default_memory_space>
// CHECK:             %[[LOAD_0:.*]] = memref.load %[[FROM_PTR_0]]{{\[}}%[[CONSTANT_0]]] : memref<1xi32, #tptr.default_memory_space>
// CHECK:             linalg.yield %[[LOAD_0]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[GENERIC_4:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_3]] : tensor<1024xi32>) outs(%[[EMPTY_2]] : tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: f32):
// CHECK:             %[[BITCAST_0:.*]] = arith.bitcast %[[VAL_9]] : i32 to f32
// CHECK:             linalg.yield %[[BITCAST_0]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[GENERIC_2]], %[[GENERIC_4]] : tensor<1024x!ptr.ptr<#tptr.default_memory_space>>, tensor<1024xf32>) {
// CHECK:           ^bb0(%[[VAL_11:.*]]: !ptr.ptr<#tptr.default_memory_space>, %[[VAL_12:.*]]: f32):
// CHECK:             %[[FROM_PTR_1:.*]] = ptr.from_ptr %[[VAL_11]] : <#tptr.default_memory_space> -> memref<1xf32, #tptr.default_memory_space>
// CHECK:             memref.store %[[VAL_12]], %[[FROM_PTR_1]]{{\[}}%[[CONSTANT_0]]] : memref<1xf32, #tptr.default_memory_space>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }