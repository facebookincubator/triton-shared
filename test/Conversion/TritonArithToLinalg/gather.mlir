// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @gather_axis1(%src: tensor<8x16xf32>, %indices: tensor<8x16xi32>) -> tensor<8x16xf32> {
    %0 = tt.gather %src[%indices] {axis = 1 : i32} : (tensor<8x16xf32>, tensor<8x16xi32>) -> tensor<8x16xf32>
    tt.return %0 : tensor<8x16xf32>
  }
}

// CHECK-LABEL:   func.func @gather_axis1(
// CHECK-SAME:                            %[[SRC:.*]]: tensor<8x16xf32>,
// CHECK-SAME:                            %[[INDICES:.*]]: tensor<8x16xi32>) -> tensor<8x16xf32> {
// CHECK:           %[[COORDS_INIT:.*]] = tensor.empty() : tensor<8x16x2xindex>
// CHECK:           %[[CAST_INIT:.*]] = tensor.empty() : tensor<8x16xindex>
// CHECK:           %[[IDX_CAST:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel"]} ins(%[[INDICES]] : tensor<8x16xi32>) outs(%[[CAST_INIT]] : tensor<8x16xindex>) {
// CHECK:           ^bb0(%[[IN:.*]]: i32, %{{.*}}: index):
// CHECK:             %[[CAST:.*]] = arith.index_cast %[[IN]] : i32 to index
// CHECK:             linalg.yield %[[CAST]] : index
// CHECK:           } -> tensor<8x16xindex>
// CHECK:           %[[IOTA_INIT0:.*]] = tensor.empty() : tensor<8x16xindex>
// CHECK:           %[[IOTA0:.*]] = linalg.generic {indexing_maps = [#{{.*}}], iterator_types = ["parallel", "parallel"]} outs(%[[IOTA_INIT0]] : tensor<8x16xindex>) {
// CHECK:           ^bb0(%{{.*}}: index):
// CHECK:             %[[ROW:.*]] = linalg.index 0 : index
// CHECK:             linalg.yield %[[ROW]] : index
// CHECK:           } -> tensor<8x16xindex>
// CHECK:           %[[COORDS0:.*]] = tensor.insert_slice %[[IOTA0]] into %[[COORDS_INIT]][0, 0, 0] [8, 16, 1] [1, 1, 1] : tensor<8x16xindex> into tensor<8x16x2xindex>
// CHECK:           %[[COORDS1:.*]] = tensor.insert_slice %[[IDX_CAST]] into %[[COORDS0]][0, 0, 1] [8, 16, 1] [1, 1, 1] : tensor<8x16xindex> into tensor<8x16x2xindex>
// CHECK:           %[[GATHER:.*]] = tensor.gather %[[SRC]]{{\[}}%[[COORDS1]]] gather_dims([0, 1]) {triton_shared.tt_gather_axis = 1 : i64} : (tensor<8x16xf32>, tensor<8x16x2xindex>) -> tensor<8x16xf32>
// CHECK:           return %[[GATHER]] : tensor<8x16xf32>
// CHECK:         }

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
// CHECK:           %[[COORDS_INIT:.*]] = tensor.empty() : tensor<8x16x2xindex>
// CHECK:           %[[CAST_INIT:.*]] = tensor.empty() : tensor<8x16xindex>
// CHECK:           %[[IDX_CAST:.*]] = linalg.generic {indexing_maps = [#{{.*}}, #{{.*}}], iterator_types = ["parallel", "parallel"]} ins(%[[INDICES]] : tensor<8x16xi32>) outs(%[[CAST_INIT]] : tensor<8x16xindex>) {
// CHECK:           ^bb0(%[[IN:.*]]: i32, %{{.*}}: index):
// CHECK:             %[[CAST:.*]] = arith.index_cast %[[IN]] : i32 to index
// CHECK:             linalg.yield %[[CAST]] : index
// CHECK:           } -> tensor<8x16xindex>
// CHECK:           %[[COORDS0:.*]] = tensor.insert_slice %[[IDX_CAST]] into %[[COORDS_INIT]][0, 0, 0] [8, 16, 1] [1, 1, 1] : tensor<8x16xindex> into tensor<8x16x2xindex>
// CHECK:           %[[IOTA_INIT1:.*]] = tensor.empty() : tensor<8x16xindex>
// CHECK:           %[[IOTA1:.*]] = linalg.generic {indexing_maps = [#{{.*}}], iterator_types = ["parallel", "parallel"]} outs(%[[IOTA_INIT1]] : tensor<8x16xindex>) {
// CHECK:           ^bb0(%{{.*}}: index):
// CHECK:             %[[COL:.*]] = linalg.index 1 : index
// CHECK:             linalg.yield %[[COL]] : index
// CHECK:           } -> tensor<8x16xindex>
// CHECK:           %[[COORDS1:.*]] = tensor.insert_slice %[[IOTA1]] into %[[COORDS0]][0, 0, 1] [8, 16, 1] [1, 1, 1] : tensor<8x16xindex> into tensor<8x16x2xindex>
// CHECK:           %[[GATHER:.*]] = tensor.gather %[[SRC]]{{\[}}%[[COORDS1]]] gather_dims([0, 1]) {triton_shared.tt_gather_axis = 0 : i64} : (tensor<8x16xf32>, tensor<8x16x2xindex>) -> tensor<8x16xf32>
// CHECK:           return %[[GATHER]] : tensor<8x16xf32>
// CHECK:         }
