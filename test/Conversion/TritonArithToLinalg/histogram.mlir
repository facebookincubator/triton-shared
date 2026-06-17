// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @histogram(%arg0: tensor<8xi32>) -> tensor<4xi32> {
    %0 = tt.histogram %arg0 : tensor<8xi32> -> tensor<4xi32>
    tt.return %0 : tensor<4xi32>
  }
}

// CHECK-LABEL: func.func @histogram(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8xi32>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:     %[[I0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[I1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[I8:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[I4:.*]] = arith.constant 4 : index
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:         %[[HIST:.*]] = scf.for %[[BIN:.*]] = %[[I0]] to %[[I4]] step %[[I1]] iter_args(%[[OUT:.*]] = %[[FILL]]) -> (tensor<4xi32>) {
// CHECK:           %[[BIN_VALUE:.*]] = arith.index_cast %[[BIN]] : index to i32
// CHECK:           %[[COUNT:.*]] = scf.for %[[SRC_INDEX:.*]] = %[[I0]] to %[[I8]] step %[[I1]] iter_args(%[[ACC:.*]] = %[[C0]]) -> (i32) {
// CHECK:             %[[SRC_VALUE:.*]] = tensor.extract %[[ARG0]][%[[SRC_INDEX]]] : tensor<8xi32>
// CHECK:             %[[IS_BIN:.*]] = arith.cmpi eq, %[[SRC_VALUE]], %[[BIN_VALUE]] : i32
// CHECK:             %[[INC:.*]] = arith.addi %[[ACC]], %[[C1]] : i32
// CHECK:             %[[NEXT:.*]] = arith.select %[[IS_BIN]], %[[INC]], %[[ACC]] : i32
// CHECK:             scf.yield %[[NEXT]] : i32
// CHECK:           }
// CHECK:           %[[NEXT_HIST:.*]] = tensor.insert %[[COUNT]] into %[[OUT]][%[[BIN]]] : tensor<4xi32>
// CHECK:           scf.yield %[[NEXT_HIST]] : tensor<4xi32>
// CHECK:         }
// CHECK-NOT:     tt.histogram
// CHECK:         return %[[HIST]] : tensor<4xi32>

// -----

module {
  tt.func public @masked_histogram(%arg0: tensor<8xi32>,
                                   %arg1: tensor<8xi1>) -> tensor<4xi32> {
    %0 = tt.histogram %arg0, %arg1 : tensor<8xi32> -> tensor<4xi32>
    tt.return %0 : tensor<4xi32>
  }
}

// CHECK-LABEL: func.func @masked_histogram(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<8xi32>, %[[ARG1:.*]]: tensor<8xi1>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : i32
// CHECK-DAG:     %[[I0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[I1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[I8:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[I4:.*]] = arith.constant 4 : index
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:         %[[HIST:.*]] = scf.for %[[BIN:.*]] = %[[I0]] to %[[I4]] step %[[I1]] iter_args(%[[OUT:.*]] = %[[FILL]]) -> (tensor<4xi32>) {
// CHECK:           %[[BIN_VALUE:.*]] = arith.index_cast %[[BIN]] : index to i32
// CHECK:           %[[COUNT:.*]] = scf.for %[[SRC_INDEX:.*]] = %[[I0]] to %[[I8]] step %[[I1]] iter_args(%[[ACC:.*]] = %[[C0]]) -> (i32) {
// CHECK:             %[[SRC_VALUE:.*]] = tensor.extract %[[ARG0]][%[[SRC_INDEX]]] : tensor<8xi32>
// CHECK:             %[[IS_BIN:.*]] = arith.cmpi eq, %[[SRC_VALUE]], %[[BIN_VALUE]] : i32
// CHECK:             %[[MASK:.*]] = tensor.extract %[[ARG1]][%[[SRC_INDEX]]] : tensor<8xi1>
// CHECK:             %[[PRED:.*]] = arith.andi %[[IS_BIN]], %[[MASK]] : i1
// CHECK:             %[[INC:.*]] = arith.addi %[[ACC]], %[[C1]] : i32
// CHECK:             %[[NEXT:.*]] = arith.select %[[PRED]], %[[INC]], %[[ACC]] : i32
// CHECK:             scf.yield %[[NEXT]] : i32
// CHECK:           }
// CHECK:           %[[NEXT_HIST:.*]] = tensor.insert %[[COUNT]] into %[[OUT]][%[[BIN]]] : tensor<4xi32>
// CHECK:           scf.yield %[[NEXT_HIST]] : tensor<4xi32>
// CHECK:         }
// CHECK-NOT:     tt.histogram
// CHECK:         return %[[HIST]] : tensor<4xi32>
