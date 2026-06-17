// Test that triton load and store with mask are lowered correctly
// (scf.if guarding the load and store)
// RUN: triton-shared-opt --triton-arith-to-linalg="tensor-ptr-to-linalg" --triton-to-ptr --cse --canonicalize %s | FileCheck %s

module {
  tt.func public @tensor_ptr(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16xi32>
    %cst_0 = arith.constant dense<16> : tensor<16xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %3 = tt.load %2 : tensor<16x!tt.ptr<i32>>
    %4 = arith.extsi %3 : tensor<16xi32> to tensor<16xi64>
    %5 = tt.int_to_ptr %4 : tensor<16xi64> -> tensor<16x!tt.ptr<i32>>
    %6 = arith.cmpi slt, %0, %cst_0 : tensor<16xi32>
    %7 = tt.load %5, %6 : tensor<16x!tt.ptr<i32>>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
    %10 = tt.ptr_to_int %9 : tensor<16x!tt.ptr<i32>> -> tensor<16xi64>
    %11 = arith.extsi %7 : tensor<16xi32> to tensor<16xi64>
    %12 = arith.addi %10, %11 : tensor<16xi64>
    %13 = tt.bitcast %9 : tensor<16x!tt.ptr<i32>> -> tensor<16x!tt.ptr<i64>>
    %14 = tt.addptr %13, %cst : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %15 = tt.bitcast %14 : tensor<16x!tt.ptr<i64>> -> tensor<16x!tt.ptr<i32>>
    %16 = arith.trunci %12 : tensor<16xi64> to tensor<16xi32>
    tt.store %15, %16, %6 : tensor<16x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:  linalg.generic
// CHECK:           ^bb0(%[[VAL_13:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_14:.*]]: i1, %[[VAL_15:.*]]: i32):
// CHECK:             %[[FROM_PTR_1:.*]] = tptr.from_ptr %[[VAL_13]] : <#ptr.generic_space> -> memref<1xi32, #ptr.generic_space>
// CHECK:             %[[IF_0:.*]] = scf.if %[[VAL_14]] -> (i32) {
// CHECK:               %[[LOAD_1:.*]] = memref.load %[[FROM_PTR_1]]{{\[}}%[[CONSTANT_1]]] : memref<1xi32, #ptr.generic_space>
// CHECK:               scf.yield %[[LOAD_1]] : i32
// CHECK:             } else {
// CHECK:               scf.yield %[[CONSTANT_0]] : i32
// CHECK:             }
// CHECK:             linalg.yield %[[IF_0]] : i32
// CHECK:           } -> tensor<16xi32>

// CHECK:  linalg.generic
// CHECK:           ^bb0(%[[VAL_31:.*]]: !ptr.ptr<#ptr.generic_space>, %[[VAL_32:.*]]: i32, %[[VAL_33:.*]]: i1):
// CHECK:             scf.if %[[VAL_33]] {
// CHECK:               %[[FROM_PTR_2:.*]] = tptr.from_ptr %[[VAL_31]] : <#ptr.generic_space> -> memref<1xi32, #ptr.generic_space>
// CHECK:               memref.store %[[VAL_32]], %[[FROM_PTR_2]]{{\[}}%[[CONSTANT_1]]] : memref<1xi32, #ptr.generic_space>
// CHECK:             }
// CHECK:             linalg.yield
// CHECK:           }
