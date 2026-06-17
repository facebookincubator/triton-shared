// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
    %f32ptr : !tt.ptr<f32>,
    %intptr : !tt.ptr<i32>,
    %f16ptr : !tt.ptr<f16>,
    %save_ptr0 : !tt.ptr<bf16>,
    %save_ptr1 : !tt.ptr<f32>,
    %save_ptr2 : !tt.ptr<f32>,
    %save_ptr3 : !tt.ptr<f32>,
    %save_ptr4 : !tt.ptr<f32>
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
    // save pointers, intentionally splat the base pointer for brevity
    %save0 = tt.splat %save_ptr0 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>>
    %save1 = tt.splat %save_ptr1 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    %save2 = tt.splat %save_ptr2 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    %save3 = tt.splat %save_ptr3 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    %save4 = tt.splat %save_ptr4 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    tt.store %save0, %5 : tensor<128x128x!tt.ptr<bf16>>
    tt.store %save1, %6 : tensor<128x128x!tt.ptr<f32>>
    tt.store %save2, %7 : tensor<128x128x!tt.ptr<f32>>
    tt.store %save3, %10 : tensor<128x128x!tt.ptr<f32>>
    tt.store %save4, %11 : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[PARAM_1_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[PARAM_2_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[PARAM_3_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[PARAM_4_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[PARAM_5_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[PARAM_6_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[PARAM_7_:%.+]]: !ptr.ptr<#ptr.generic_space>, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32) {
// CHECK-DAG:       [[PARAM_0__TTPTR:%.+]] = tptr.from_ptr [[PARAM_0_]] : <#ptr.generic_space> -> memref<1xf32, #ptr.generic_space>
// CHECK-DAG:       [[PARAM_0__MEMREF:%.+]] = memref.memory_space_cast [[PARAM_0__TTPTR]] : memref<1xf32, #ptr.generic_space> to memref<1xf32>
// CHECK-DAG:       [[PARAM_1__TTPTR:%.+]] = tptr.from_ptr [[PARAM_1_]] : <#ptr.generic_space> -> memref<1xi32, #ptr.generic_space>
// CHECK-DAG:       [[PARAM_1__MEMREF:%.+]] = memref.memory_space_cast [[PARAM_1__TTPTR]] : memref<1xi32, #ptr.generic_space> to memref<1xi32>
// CHECK-DAG:       [[PARAM_2__TTPTR:%.+]] = tptr.from_ptr [[PARAM_2_]] : <#ptr.generic_space> -> memref<1xf16, #ptr.generic_space>
// CHECK-DAG:       [[PARAM_2__MEMREF:%.+]] = memref.memory_space_cast [[PARAM_2__TTPTR]] : memref<1xf16, #ptr.generic_space> to memref<1xf16>
// CHECK-DAG:       [[PARAM_3__TTPTR:%.+]] = tptr.from_ptr [[PARAM_3_]] : <#ptr.generic_space> -> memref<1xbf16, #ptr.generic_space>
// CHECK-DAG:       [[PARAM_3__MEMREF:%.+]] = memref.memory_space_cast [[PARAM_3__TTPTR]] : memref<1xbf16, #ptr.generic_space> to memref<1xbf16>
// CHECK-DAG:       [[PARAM_4__TTPTR:%.+]] = tptr.from_ptr [[PARAM_4_]] : <#ptr.generic_space> -> memref<1xf32, #ptr.generic_space>
// CHECK-DAG:       [[PARAM_4__MEMREF:%.+]] = memref.memory_space_cast [[PARAM_4__TTPTR]] : memref<1xf32, #ptr.generic_space> to memref<1xf32>
// CHECK-DAG:       [[PARAM_5__TTPTR:%.+]] = tptr.from_ptr [[PARAM_5_]] : <#ptr.generic_space> -> memref<1xf32, #ptr.generic_space>
// CHECK-DAG:       [[PARAM_5__MEMREF:%.+]] = memref.memory_space_cast [[PARAM_5__TTPTR]] : memref<1xf32, #ptr.generic_space> to memref<1xf32>
// CHECK-DAG:       [[PARAM_6__TTPTR:%.+]] = tptr.from_ptr [[PARAM_6_]] : <#ptr.generic_space> -> memref<1xf32, #ptr.generic_space>
// CHECK-DAG:       [[PARAM_6__MEMREF:%.+]] = memref.memory_space_cast [[PARAM_6__TTPTR]] : memref<1xf32, #ptr.generic_space> to memref<1xf32>
// CHECK-DAG:       [[PARAM_7__TTPTR:%.+]] = tptr.from_ptr [[PARAM_7_]] : <#ptr.generic_space> -> memref<1xf32, #ptr.generic_space>
// CHECK-DAG:       [[PARAM_7__MEMREF:%.+]] = memref.memory_space_cast [[PARAM_7__TTPTR]] : memref<1xf32, #ptr.generic_space> to memref<1xf32>
// CHECK-DAG:           [[VAR_cast_3_:%.+]] = memref.reinterpret_cast [[PARAM_3__MEMREF]] to offset: [0], sizes: [1], strides: [1] : memref<1xbf16> to memref<?xbf16>
// CHECK-DAG:           [[VAR_cast_4_:%.+]] = memref.reinterpret_cast [[PARAM_4__MEMREF]] to offset: [0], sizes: [1], strides: [1] : memref<1xf32> to memref<?xf32>
// CHECK-DAG:           [[VAR_cast_5_:%.+]] = memref.reinterpret_cast [[PARAM_5__MEMREF]] to offset: [0], sizes: [1], strides: [1] : memref<1xf32> to memref<?xf32>
// CHECK-DAG:           [[VAR_cast_6_:%.+]] = memref.reinterpret_cast [[PARAM_6__MEMREF]] to offset: [0], sizes: [1], strides: [1] : memref<1xf32> to memref<?xf32>
// CHECK-DAG:           [[VAR_cast_7_:%.+]] = memref.reinterpret_cast [[PARAM_7__MEMREF]] to offset: [0], sizes: [1], strides: [1] : memref<1xf32> to memref<?xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_empty_offsets_:%.+]] = tensor.empty() : tensor<128x128xi32>
// CHECK-DAG:       [[VAR_zero_offsets_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_empty_offsets_]] : tensor<128x128xi32>) -> tensor<128x128xi32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0__MEMREF]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<1xf32> to memref<128x128xf32, strided<[1, 1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1__MEMREF]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<1xi32> to memref<128x128xi32, strided<[1, 1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_2__MEMREF]] to offset: [0], sizes: [128, 128], strides: [1, 1] : memref<1xf16> to memref<128x128xf16, strided<[1, 1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<128x128xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<128x128xf32, strided<[1, 1]>> to memref<128x128xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128x128xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<128x128xi32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_0_]], [[RES_1_]] : memref<128x128xi32, strided<[1, 1]>> to memref<128x128xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<128x128xi32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<128x128xf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_1_]], [[RES_2_]] : memref<128x128xf16, strided<[1, 1]>> to memref<128x128xf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_2_]] restrict writable : memref<128x128xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = tensor.empty() : tensor<128x128xbf16>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_0_]] : tensor<128x128xf32>) outs([[VAR_3_]] : tensor<128x128xbf16>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: bf16):
// CHECK:             [[VAR_10_:%.+]] = arith.truncf [[IN_0_]] : f32 to bf16
// CHECK:             linalg.yield [[VAR_10_]] : bf16
// CHECK:           } -> tensor<128x128xbf16>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_0_]] : tensor<128x128xf32>) outs([[VAR_0_]] : tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_2_:%.+]]: f32, [[IN_3_:%.+]]: f32):
// CHECK:             [[VAR_10_1_:%.+]] = math.exp [[IN_2_]] : f32
// CHECK:             linalg.yield [[VAR_10_1_]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           [[VAR_6_:%.+]] = tensor.empty() : tensor<128x128xf32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_1_]] : tensor<128x128xi32>) outs([[VAR_6_]] : tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_4_:%.+]]: i32, [[IN_5_:%.+]]: f32):
// CHECK:             [[VAR_10_2_:%.+]] = arith.sitofp [[IN_4_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_10_2_]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_2_]] : tensor<128x128xf16>) outs([[VAR_6_]] : tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_6_:%.+]]: f16, [[IN_7_:%.+]]: f32):
// CHECK:             [[VAR_10_3_:%.+]] = arith.extf [[IN_6_]] : f16 to f32
// CHECK:             linalg.yield [[VAR_10_3_]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_0_]] : tensor<128x128xf32>) outs([[VAR_0_]] : tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_8_:%.+]]: f32, [[IN_9_:%.+]]: f32):
// CHECK:             [[VAR_10_4_:%.+]] = math.sqrt [[IN_8_]] : f32
// CHECK:             linalg.yield [[VAR_10_4_]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_zero_offsets_]], [[VAR_4_]] : tensor<128x128xi32>, tensor<128x128xbf16>) {
// CHECK:           ^bb0([[IN_11_:%.+]]: i32, [[IN_12_:%.+]]: bf16):
// CHECK:             [[VAR_11_:%.+]] = arith.index_cast [[IN_11_]] : i32 to index
// CHECK:             memref.store [[IN_12_]], [[VAR_cast_3_]]{{.}}[[VAR_11_]]{{.}} : memref<?xbf16>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_zero_offsets_]], [[VAR_5_]] : tensor<128x128xi32>, tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_13_:%.+]]: i32, [[IN_14_:%.+]]: f32):
// CHECK:             [[VAR_12_:%.+]] = arith.index_cast [[IN_13_]] : i32 to index
// CHECK:             memref.store [[IN_14_]], [[VAR_cast_4_]]{{.}}[[VAR_12_]]{{.}} : memref<?xf32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_zero_offsets_]], [[VAR_7_]] : tensor<128x128xi32>, tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_15_:%.+]]: i32, [[IN_16_:%.+]]: f32):
// CHECK:             [[VAR_13_:%.+]] = arith.index_cast [[IN_15_]] : i32 to index
// CHECK:             memref.store [[IN_16_]], [[VAR_cast_5_]]{{.}}[[VAR_13_]]{{.}} : memref<?xf32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_zero_offsets_]], [[VAR_8_]] : tensor<128x128xi32>, tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_17_:%.+]]: i32, [[IN_18_:%.+]]: f32):
// CHECK:             [[VAR_14_:%.+]] = arith.index_cast [[IN_17_]] : i32 to index
// CHECK:             memref.store [[IN_18_]], [[VAR_cast_6_]]{{.}}[[VAR_14_]]{{.}} : memref<?xf32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_zero_offsets_]], [[VAR_9_]] : tensor<128x128xi32>, tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_19_:%.+]]: i32, [[IN_20_:%.+]]: f32):
// CHECK:             [[VAR_15_:%.+]] = arith.index_cast [[IN_19_]] : i32 to index
// CHECK:             memref.store [[IN_20_]], [[VAR_cast_7_]]{{.}}[[VAR_15_]]{{.}} : memref<?xf32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
