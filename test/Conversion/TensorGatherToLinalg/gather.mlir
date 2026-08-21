// RUN: triton-shared-opt --tensor-gather-to-linalg %s | FileCheck %s

// A tensor.gather carrying the tt.gather provenance marker attribute, built
// in the exact shape GatherConverter (TritonArithToLinalg) is guaranteed to
// produce: `coords` is a chain of two tensor.insert_slice ops over an empty
// index tensor, one per source dim, with the axis=1 slot carrying the real
// tt.gather indices -- scalarized via a linalg.generic wrapping
// arith.index_cast (the form linalg::populateElementwiseToLinalgConversionPatterns
// legalizes GatherConverter's bare tensor-wide arith.index_cast into, within
// the same TritonArithToLinalg conversion, since arith ops on tensors are
// dynamically illegal there) -- and the axis=0 slot carrying a per-row iota.
// This should be rewritten back into the single-axis linalg.generic +
// tensor.extract form.
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @gather_axis1(%src: tensor<8x16xf32>, %indices: tensor<8x16xi32>) -> tensor<8x16xf32> {
    %coords_init = tensor.empty() : tensor<8x16x2xindex>
    %cast_init = tensor.empty() : tensor<8x16xindex>
    %idx_cast = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%indices : tensor<8x16xi32>) outs(%cast_init : tensor<8x16xindex>) {
    ^bb0(%in: i32, %out: index):
      %cast = arith.index_cast %in : i32 to index
      linalg.yield %cast : index
    } -> tensor<8x16xindex>
    %iota_init = tensor.empty() : tensor<8x16xindex>
    %iota = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%iota_init : tensor<8x16xindex>) {
    ^bb0(%out: index):
      %row = linalg.index 0 : index
      linalg.yield %row : index
    } -> tensor<8x16xindex>
    %coords0 = tensor.insert_slice %iota into %coords_init[0, 0, 0] [8, 16, 1] [1, 1, 1] : tensor<8x16xindex> into tensor<8x16x2xindex>
    %coords1 = tensor.insert_slice %idx_cast into %coords0[0, 0, 1] [8, 16, 1] [1, 1, 1] : tensor<8x16xindex> into tensor<8x16x2xindex>
    %gather = tensor.gather %src[%coords1] gather_dims([0, 1]) {triton_shared.tt_gather_axis = 1 : i64} : (tensor<8x16xf32>, tensor<8x16x2xindex>) -> tensor<8x16xf32>
    return %gather : tensor<8x16xf32>
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

// A tensor.gather without the marker attribute must be left untouched: the
// pattern must not fire on arbitrary tensor.gather ops, only ones known to
// have come from tt.gather.
module {
  func.func @unmarked_gather(%src: tensor<8x16xf32>, %coords: tensor<8x16x2xindex>) -> tensor<8x16xf32> {
    %gather = tensor.gather %src[%coords] gather_dims([0, 1]) : (tensor<8x16xf32>, tensor<8x16x2xindex>) -> tensor<8x16xf32>
    return %gather : tensor<8x16xf32>
  }
}

// CHECK-LABEL:   func.func @unmarked_gather(
// CHECK-SAME:                               %[[SRC:.*]]: tensor<8x16xf32>,
// CHECK-SAME:                               %[[COORDS:.*]]: tensor<8x16x2xindex>) -> tensor<8x16xf32> {
// CHECK:           %[[GATHER:.*]] = tensor.gather %[[SRC]]{{\[}}%[[COORDS]]] gather_dims([0, 1]) : (tensor<8x16xf32>, tensor<8x16x2xindex>) -> tensor<8x16xf32>
// CHECK:           return %[[GATHER]] : tensor<8x16xf32>
// CHECK:         }
