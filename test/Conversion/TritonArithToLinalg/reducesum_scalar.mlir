// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(%afloat : !tt.ptr<bf16>, %res : !tt.ptr<bf16>)
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %afm = tt.load %2 : tensor<128x!tt.ptr<bf16>>
    %3 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 0 : i32} : (tensor<128xbf16>) -> bf16
    tt.store %res, %3 : !tt.ptr<bf16>
    tt.return
  }
}



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:.*]]: !tt.ptr<bf16>,                 %[[ARG1:.*]]: !tt.ptr<bf16>,                 %[[ARG2:.*]]: i32,                 %[[ARG3:.*]]: i32,                 %[[ARG4:.*]]: i32,                 %[[ARG5:.*]]: i32,                 %[[ARG6:.*]]: i32,                 %[[ARG7:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<128x!tt.ptr<bf16>>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_0]] : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>) outs(%[[SPLAT_0]] : tensor<128x!tt.ptr<bf16>>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: !tt.ptr<bf16>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: !tt.ptr<bf16>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_1]], %[[VAL_2]] : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<128x!tt.ptr<bf16>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_1]] : tensor<128x!tt.ptr<bf16>>
// CHECK:           %[[ALLOC_TENSOR_0:.*]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           %[[INSERT_0:.*]] = tensor.insert %[[CONSTANT_0]] into %[[ALLOC_TENSOR_0]][] : tensor<f32>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[LOAD_0]] : tensor<128xbf16>) outs(%[[INSERT_0]] : tensor<f32>) dimensions = [0]
// CHECK:             (%[[VAL_4:.*]]: bf16, %[[VAL_5:.*]]: f32) {
// CHECK:               %[[EXTF_0:.*]] = arith.extf %[[VAL_4]] : bf16 to f32
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[EXTF_0]], %[[VAL_5]] : f32
// CHECK:               linalg.yield %[[ADDF_0]] : f32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<f32>
// CHECK:           %[[TRUNCF_0:.*]] = arith.truncf %[[EXTRACT_0]] : f32 to bf16
// CHECK:           tt.store %[[ARG1]], %[[TRUNCF_0]] : !tt.ptr<bf16>
// CHECK:           return
// CHECK:         }

