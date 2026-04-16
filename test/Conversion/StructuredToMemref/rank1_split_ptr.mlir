// RUN: triton-shared-opt --structured-to-memref --canonicalize %s | FileCheck %s

// A rank-1 tts.make_tptr with a non-zero shape field is a "split pointer"
// (isSplitPtr() == true).  For rank-1 tensors the wrap-around is handled by
// treating the pointer as a regular structured pointer, emitting a single
// memref.reinterpret_cast rather than two split chunks.

// -----

module {
  tt.func public @rank1_split_ptr_static_offset(%arg0: !tt.ptr<f32>) {
    %c16 = arith.constant 16 : index
    %ptr = tts.make_tptr %arg0 to sizes: [128], strides: [1], offsets: [%c16],
               shape: [256], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
    %load = "tts.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>,
               static_mask_dims = array<i64>}> :
               (tensor<128x!tt.ptr<f32>>) -> tensor<128xf32>
    "tts.store"(%ptr, %load) <{static_mask_dims = array<i64>}> :
               (tensor<128x!tt.ptr<f32>>, tensor<128xf32>) -> ()
    tt.return
  }
}

// CHECK-LABEL:  tt.func public @rank1_split_ptr_static_offset
// CHECK:        [[RC:%.+]] = memref.reinterpret_cast {{%.+}} to offset: [16], sizes: [128], strides: [1]
// CHECK:        [[ALLOC:%.+]] = memref.alloc() : memref<128xf32>
// CHECK:        memref.copy [[RC]], [[ALLOC]]
// CHECK:        [[T:%.+]] = bufferization.to_tensor [[ALLOC]] restrict writable
// CHECK:        bufferization.materialize_in_destination [[T]] in writable
// CHECK:        tt.return

// -----

module {
  tt.func public @rank1_split_ptr_dynamic_offset(%arg0: !tt.ptr<f32>,
                                                  %arg1: index) {
    %ptr = tts.make_tptr %arg0 to sizes: [128], strides: [1], offsets: [%arg1],
               shape: [256], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
    %load = "tts.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>,
               static_mask_dims = array<i64>}> :
               (tensor<128x!tt.ptr<f32>>) -> tensor<128xf32>
    "tts.store"(%ptr, %load) <{static_mask_dims = array<i64>}> :
               (tensor<128x!tt.ptr<f32>>, tensor<128xf32>) -> ()
    tt.return
  }
}

// CHECK-LABEL:  tt.func public @rank1_split_ptr_dynamic_offset
// CHECK:        [[RC:%.+]] = memref.reinterpret_cast {{%.+}} to offset: [{{%.+}}], sizes: [128], strides: [1]
// CHECK:        [[ALLOC:%.+]] = memref.alloc() : memref<128xf32>
// CHECK:        memref.copy [[RC]], [[ALLOC]]
// CHECK:        [[T:%.+]] = bufferization.to_tensor [[ALLOC]] restrict writable
// CHECK:        bufferization.materialize_in_destination [[T]] in writable [[RC]]
// CHECK:        tt.return
