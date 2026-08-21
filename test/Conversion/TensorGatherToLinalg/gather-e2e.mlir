// RUN: triton-shared-opt --triton-arith-to-linalg --tensor-gather-to-linalg %s | FileCheck %s

// Composes TritonArithToLinalg's tt.gather -> tensor.gather lowering with
// this pass's tensor.gather -> linalg.generic reverse lowering, and checks
// the result is bit-for-bit identical to the pre-retarget, hand-rolled
// GatherConverter output (see commit 2f10c15,
// ConversionPatterns.hpp:1033-1092) -- i.e. round-tripping through
// tensor.gather is lossless.
module {
  tt.func public @gather_axis1(%src: tensor<8x16xf32>, %indices: tensor<8x16xi32>) -> tensor<8x16xf32> {
    %0 = tt.gather %src[%indices] {axis = 1 : i32} : (tensor<8x16xf32>, tensor<8x16xi32>) -> tensor<8x16xf32>
    tt.return %0 : tensor<8x16xf32>
  }
}

// CHECK-LABEL:   func.func @gather_axis1(
// CHECK-SAME:                            %[[SRC:.*]]: tensor<8x16xf32>,
// CHECK-SAME:                            %[[INDICES:.*]]: tensor<8x16xi32>) -> tensor<8x16xf32> {
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<8x16xf32>
// CHECK:           %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel"]} ins(%[[INDICES]] : tensor<8x16xi32>) outs(%[[EMPTY]] : tensor<8x16xf32>) {
// CHECK:           ^bb0(%[[IDX:.*]]: i32, %{{.*}}: f32):
// CHECK:             %[[IDX_CAST:.*]] = arith.index_cast %[[IDX]] : i32 to index
// CHECK:             %[[ROW:.*]] = linalg.index 0 : index
// CHECK:             %[[EXTRACTED:.*]] = tensor.extract %[[SRC]]{{\[}}%[[ROW]], %[[IDX_CAST]]] : tensor<8x16xf32>
// CHECK:             linalg.yield %[[EXTRACTED]] : f32
// CHECK:           } -> tensor<8x16xf32>
// CHECK:           return %[[GENERIC]] : tensor<8x16xf32>
// CHECK:         }
// CHECK-NOT:       tensor.gather

// -----

module {
  tt.func public @gather_axis0(%src: tensor<8x16xf32>, %indices: tensor<8x16xi32>) -> tensor<8x16xf32> {
    %0 = tt.gather %src[%indices] {axis = 0 : i32} : (tensor<8x16xf32>, tensor<8x16xi32>) -> tensor<8x16xf32>
    tt.return %0 : tensor<8x16xf32>
  }
}

// CHECK-LABEL:   func.func @gather_axis0(
// CHECK-SAME:                            %[[SRC:.*]]: tensor<8x16xf32>,
// CHECK-SAME:                            %[[INDICES:.*]]: tensor<8x16xi32>) -> tensor<8x16xf32> {
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<8x16xf32>
// CHECK:           %[[GENERIC:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel"]} ins(%[[INDICES]] : tensor<8x16xi32>) outs(%[[EMPTY]] : tensor<8x16xf32>) {
// CHECK:           ^bb0(%[[IDX:.*]]: i32, %{{.*}}: f32):
// CHECK:             %[[IDX_CAST:.*]] = arith.index_cast %[[IDX]] : i32 to index
// CHECK:             %[[COL:.*]] = linalg.index 1 : index
// CHECK:             %[[EXTRACTED:.*]] = tensor.extract %[[SRC]]{{\[}}%[[IDX_CAST]], %[[COL]]] : tensor<8x16xf32>
// CHECK:             linalg.yield %[[EXTRACTED]] : f32
// CHECK:           } -> tensor<8x16xf32>
// CHECK:           return %[[GENERIC]] : tensor<8x16xf32>
// CHECK:         }
// CHECK-NOT:       tensor.gather
