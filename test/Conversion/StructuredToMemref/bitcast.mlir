// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

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

    // cast result before doing float add
    %am_bitcast = tt.bitcast %am : tensor<1024xi32> -> tensor<1024xf32>


    tt.store %19, %am_bitcast : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[PARAM_1_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[PARAM_0__TTPTR:%.+]] = tptr.from_ptr [[PARAM_0_]] : <#ptr.generic_space> -> memref<1xi32, #ptr.generic_space>
// CHECK-DAG:       [[PARAM_0__MEMREF:%.+]] = memref.memory_space_cast [[PARAM_0__TTPTR]] : memref<1xi32, #ptr.generic_space> to memref<1xi32>
// CHECK-DAG:       [[PARAM_1__TTPTR:%.+]] = tptr.from_ptr [[PARAM_1_]] : <#ptr.generic_space> -> memref<1xf32, #ptr.generic_space>
// CHECK-DAG:       [[PARAM_1__MEMREF:%.+]] = memref.memory_space_cast [[PARAM_1__TTPTR]] : memref<1xf32, #ptr.generic_space> to memref<1xf32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0__MEMREF]] to offset: [0], sizes: [1024], strides: [1] : memref<1xi32> to memref<1024xi32, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1__MEMREF]] to offset: [0], sizes: [1024], strides: [1] : memref<1xf32> to memref<1024xf32, strided<[1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1024xi32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<1024xi32, strided<[1]>> to memref<1024xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]] : tensor<1024xi32>) outs([[VAR_1_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_3_:%.+]] = arith.bitcast [[IN_0_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_3_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           bufferization.materialize_in_destination [[VAR_2_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<1024xf32>, memref<1024xf32, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
