// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @test_argmin_i8(%input: tensor<32xi8>, %indices: tensor<32xi32>) -> (i8, i32) attributes {noinline = false} {
    %0:2 = "tt.reduce"(%input, %indices) <{axis = 0 : i32}> ({
    ^bb0(%arg0: i8, %arg1: i32, %arg2: i8, %arg3: i32):
      %tie = arith.cmpi eq, %arg0, %arg2 : i8
      %tie_idx = arith.cmpi slt, %arg1, %arg3 : i32
      %tie_break = arith.andi %tie, %tie_idx : i1
      %lt = arith.cmpi slt, %arg0, %arg2 : i8
      %should_update = arith.ori %lt, %tie_break : i1
      %value_ret = arith.select %should_update, %arg0, %arg2 : i8
      %index_ret = arith.select %should_update, %arg1, %arg3 : i32
      tt.reduce.return %value_ret, %index_ret : i8, i32
    }) : (tensor<32xi8>, tensor<32xi32>) -> (i8, i32)
    tt.return %0#0, %0#1 : i8, i32
  }
}

// CHECK-LABEL: func.func @test_argmin_i8
// CHECK-DAG:   %[[VALUE_INIT:.*]] = arith.constant 127 : i8
// CHECK-DAG:   %[[INDEX_INIT:.*]] = arith.constant -1 : i32
// CHECK:       %[[VALUE_TENSOR:.*]] = tensor.empty() : tensor<i8>
// CHECK:       %[[VALUE_FILL:.*]] = linalg.fill ins(%[[VALUE_INIT]] : i8) outs(%[[VALUE_TENSOR]] : tensor<i8>) -> tensor<i8>
// CHECK:       %[[INDEX_TENSOR:.*]] = tensor.empty() : tensor<i32>
// CHECK:       %[[INDEX_FILL:.*]] = linalg.fill ins(%[[INDEX_INIT]] : i32) outs(%[[INDEX_TENSOR]] : tensor<i32>) -> tensor<i32>
// CHECK:       %[[REDUCE:.*]]:2 = linalg.reduce
// CHECK-SAME:    ins(%{{.*}}, %{{.*}} : tensor<32xi8>, tensor<32xi32>)
// CHECK-SAME:    outs(%[[VALUE_FILL]], %[[INDEX_FILL]] : tensor<i8>, tensor<i32>)
// CHECK-SAME:    dimensions = [0]
// CHECK:         arith.cmpi eq
// CHECK:         arith.cmpi slt
// CHECK:         arith.andi
// CHECK:         arith.cmpi slt
// CHECK:         arith.ori
// CHECK:         linalg.yield
// CHECK-NOT:   tt.reduce
