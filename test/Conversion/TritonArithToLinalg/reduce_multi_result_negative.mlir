// Negative cases for ReduceConverter::convertToLinalgReduce: bodies that must
// be rejected and left as `tt.reduce` (unconverted) rather than lowered to
// linalg.reduce, because they don't fit the "each result independently
// combines its own (input_i, accumulator_i) pair" shape the converter
// requires.

// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

// An unsupported binary op (subf isn't associative/commutative in a way the
// converter recognizes as a reduction) must not be converted.
module {
  tt.func public @unsupported_op(%x: tensor<32x512xf32>) -> tensor<32xf32> {
    %red = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a0: f32, %b0: f32):
      %s0 = arith.subf %a0, %b0 : f32
      tt.reduce.return %s0 : f32
    }) : (tensor<32x512xf32>) -> tensor<32xf32>
    tt.return %red : tensor<32xf32>
  }
}

// CHECK-LABEL:   func.func @unsupported_op(
// CHECK:           %[[RED:.*]] = "tt.reduce"(%[[ARG0:.*]])
// CHECK:           ^bb0(%[[A0:.*]]: f32, %[[B0:.*]]: f32):
// CHECK:             %[[S0:.*]] = arith.subf %[[A0]], %[[B0]] : f32
// CHECK:             tt.reduce.return %[[S0]] : f32
// CHECK:           return %[[RED]] : tensor<32xf32>

// -----

// Cross-result data flow: result 0 combines (input_1, acc_0) instead of its
// own (input_0, acc_0) pair. This shape is what argmin/argmax tie-breaking
// bodies also have, and must be left for the dedicated ArgMinMax converters
// (which don't match this particular body either, so it stays unconverted).
module {
  tt.func public @cross_result_dataflow(%x: tensor<32x512xf32>, %y: tensor<32x512xf32>) -> (tensor<32xf32>, tensor<32xf32>) {
    %red:2 = "tt.reduce"(%x, %y) <{axis = 1 : i32}> ({
    ^bb0(%a0: f32, %a1: f32, %b0: f32, %b1: f32):
      %s0 = arith.addf %a1, %b0 : f32
      %s1 = arith.addf %a0, %b1 : f32
      tt.reduce.return %s0, %s1 : f32, f32
    }) : (tensor<32x512xf32>, tensor<32x512xf32>) -> (tensor<32xf32>, tensor<32xf32>)
    tt.return %red#0, %red#1 : tensor<32xf32>, tensor<32xf32>
  }
}

// CHECK-LABEL:   func.func @cross_result_dataflow(
// CHECK:           %[[RED:.*]]:2 = "tt.reduce"(%[[ARG0:.*]], %[[ARG1:.*]])
// CHECK:           ^bb0(%[[A0:.*]]: f32, %[[A1:.*]]: f32, %[[B0:.*]]: f32, %[[B1:.*]]: f32):
// CHECK:             %[[S0:.*]] = arith.addf %[[A1]], %[[B0]] : f32
// CHECK:             %[[S1:.*]] = arith.addf %[[A0]], %[[B1]] : f32
// CHECK:             tt.reduce.return %[[S0]], %[[S1]] : f32, f32
// CHECK:           return %[[RED]]#0, %[[RED]]#1 : tensor<32xf32>, tensor<32xf32>

// -----

// Multi-result reduce over a rank-1 (vector) input: the manual affine
// lowering used for vector reduces is single-result only, so this must be
// deferred rather than lowered.
module {
  tt.func public @vector_multi_result(%x: tensor<512xf32>, %y: tensor<512xf32>) -> (f32, f32) {
    %red:2 = "tt.reduce"(%x, %y) <{axis = 0 : i32}> ({
    ^bb0(%a0: f32, %a1: f32, %b0: f32, %b1: f32):
      %s0 = arith.addf %a0, %b0 : f32
      %s1 = arith.addf %a1, %b1 : f32
      tt.reduce.return %s0, %s1 : f32, f32
    }) : (tensor<512xf32>, tensor<512xf32>) -> (f32, f32)
    tt.return %red#0, %red#1 : f32, f32
  }
}

// CHECK-LABEL:   func.func @vector_multi_result(
// CHECK:           %[[RED:.*]]:2 = "tt.reduce"(%[[ARG0:.*]], %[[ARG1:.*]])
// CHECK:           ^bb0(%[[A0:.*]]: f32, %[[A1:.*]]: f32, %[[B0:.*]]: f32, %[[B1:.*]]: f32):
// CHECK:             %[[S0:.*]] = arith.addf %[[A0]], %[[B0]] : f32
// CHECK:             %[[S1:.*]] = arith.addf %[[A1]], %[[B1]] : f32
// CHECK:             tt.reduce.return %[[S0]], %[[S1]] : f32, f32
// CHECK:           return %[[RED]]#0, %[[RED]]#1 : f32, f32
