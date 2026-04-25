// RUN: triton-shared-opt --triton-to-ptr %s | FileCheck %s

module {
  func.func @descriptor_ops(%src: memref<*xi32>, %dst: memref<*xi32>,
                            %m: i32, %n: i32) {
    %src_ptr = builtin.unrealized_conversion_cast %src : memref<*xi32> to !tt.ptr<i32>
    %dst_ptr = builtin.unrealized_conversion_cast %dst : memref<*xi32> to !tt.ptr<i32>
    %c0_i32 = arith.constant 0 : i32
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64

    %src_desc = tt.make_tensor_descriptor %src_ptr, [%m, %n], [%c4_i64, %c1_i64] : <i32>, <2x4xi32>
    %dst_desc = tt.make_tensor_descriptor %dst_ptr, [%m, %n], [%c4_i64, %c1_i64] : <i32>, <2x4xi32>
    %val = tt.descriptor_load %src_desc[%c0_i32, %c0_i32] : !tt.tensordesc<2x4xi32> -> tensor<2x4xi32>
    tt.descriptor_store %dst_desc[%c0_i32, %c0_i32], %val : !tt.tensordesc<2x4xi32>, tensor<2x4xi32>
    return
  }
}

// CHECK-LABEL: func.func @descriptor_ops(
// CHECK-SAME: %[[SRC:.*]]: memref<*xi32>, %[[DST:.*]]: memref<*xi32>, %[[M:.*]]: i32, %[[N:.*]]: i32
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[M_IDX:.*]] = arith.index_cast %[[M]] : i32 to index
// CHECK: %[[N_IDX:.*]] = arith.index_cast %[[N]] : i32 to index
// CHECK: %[[STRIDE0:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK: %[[STRIDE1:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK-DAG: %[[SRC_DESC:.*]] = memref.reinterpret_cast %[[SRC]]
// CHECK-SAME: {tt.descriptor_padding = 1 : i32}
// CHECK-DAG: %[[DST_DESC:.*]] = memref.reinterpret_cast %[[DST]]
// CHECK-SAME: {tt.descriptor_padding = 1 : i32}
// CHECK: %[[LOAD_ALLOC:.*]] = memref.alloc() : memref<2x4xi32>
// CHECK: scf.if %{{.*}} {
// CHECK:   linalg.fill ins(%{{.*}} : i32) outs(%[[LOAD_ALLOC]] : memref<2x4xi32>)
// CHECK: }
// CHECK: %[[SRC_SUBVIEW:.*]] = memref.subview %[[SRC_DESC]]
// CHECK: %[[LOAD_SUBVIEW:.*]] = memref.subview %[[LOAD_ALLOC]]
// CHECK: memref.copy %[[SRC_SUBVIEW]], %[[LOAD_SUBVIEW]]
// CHECK: %[[VAL:.*]] = bufferization.to_tensor %[[LOAD_ALLOC]] restrict writable : memref<2x4xi32> to tensor<2x4xi32>
// CHECK: %[[STORE_SLICE:.*]] = tensor.extract_slice %[[VAL]]
// CHECK: %[[DST_SUBVIEW:.*]] = memref.subview %[[DST_DESC]]
// CHECK: bufferization.materialize_in_destination %[[STORE_SLICE]] in writable %[[DST_SUBVIEW]]
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK-NOT: tt.descriptor_store
// CHECK: return
