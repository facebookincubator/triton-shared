// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @simple_cf_into_structured_load(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = arith.cmpi eq, %arg2, %c1_i32 : i32
    %1 = scf.if %0 -> (!tt.ptr<f32>) {
      %9 = arith.muli %arg2, %c2_i32 : i32
      %10 = tt.addptr %arg0, %9 : !tt.ptr<f32>, i32
      scf.yield %10 : !tt.ptr<f32>
    } else {
      %9 = tt.addptr %arg0, %arg2 : !tt.ptr<f32>, i32
      scf.yield %9 : !tt.ptr<f32>
    }
    %2 = tt.addptr %1, %c6_i32 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %6 = tt.load %5 : tensor<4x!tt.ptr<f32>>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %8 = tt.addptr %7, %3 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %8, %6 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @simple_cf_into_structured_load(
// CHECK-SAME:      %[[ARG0:.*]]: memref<*xf32>, %[[ARG1:.*]]: memref<*xf32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CAST_0:.*]] = memref.cast %[[ARG0]] : memref<*xf32> to memref<1xf32>
// CHECK:           %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CAST_0]] : memref<1xf32> to memref<1xf32, #tptr.default_memory_space>
// CHECK:           %[[TO_PTR_0:.*]] = ptr.to_ptr %[[MEMORY_SPACE_CAST_0]] : memref<1xf32, #tptr.default_memory_space> -> <#tptr.default_memory_space>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi eq, %[[ARG2]], %[[CONSTANT_1]] : i32
// CHECK:           %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (!ptr.ptr<#tptr.default_memory_space>) {
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[ARG2]], %[[CONSTANT_0]] : i32
// CHECK:             %[[TYPE_OFFSET_0:.*]] = ptr.type_offset f32 : i32
// CHECK:             %[[MULI_1:.*]] = arith.muli %[[MULI_0]], %[[TYPE_OFFSET_0]] : i32
// CHECK:             %[[PTR_ADD_0:.*]] = ptr.ptr_add %[[TO_PTR_0]], %[[MULI_1]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             scf.yield %[[PTR_ADD_0]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           } else {
// CHECK:             %[[TYPE_OFFSET_1:.*]] = ptr.type_offset f32 : i32
// CHECK:             %[[MULI_2:.*]] = arith.muli %[[ARG2]], %[[TYPE_OFFSET_1]] : i32
// CHECK:             %[[PTR_ADD_1:.*]] = ptr.ptr_add %[[TO_PTR_0]], %[[MULI_2]] : !ptr.ptr<#tptr.default_memory_space>, i32
// CHECK:             scf.yield %[[PTR_ADD_1]] : !ptr.ptr<#tptr.default_memory_space>
// CHECK:           }
// CHECK:           %[[FROM_PTR_0:.*]] = ptr.from_ptr %[[IF_0]] : <#tptr.default_memory_space> -> memref<1xf32, #tptr.default_memory_space>
// CHECK:           %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[FROM_PTR_0]] : memref<1xf32, #tptr.default_memory_space> to memref<1xf32>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [6], sizes: [4], strides: [1] : memref<1xf32> to memref<4xf32, strided<[1], offset: 6>>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<4xf32, strided<[1], offset: 6>> to memref<4xf32>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[TO_TENSOR_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<4xf32>, memref<4xf32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
