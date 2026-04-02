// RUN: triton-shared-opt --split-input-file --linalg-to-vector -cse -canonicalize %s | FileCheck %s


// CHECK: arith.constant dense
// CHECK: vector.transfer_write
// CHECK-NEXT: scf.for
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: arith.maxsi
// CHECK-NEXT: vector.transfer_write
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: vector.multi_reduction
func.func @reduce_max(%arg0: tensor<1024xi32>, %arg1: tensor<i32>) -> tensor<i32>{
  %0 = linalg.reduce ins(%arg0 : tensor<1024xi32>) outs(%arg1 : tensor<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %0 = arith.maxsi %in, %init : i32
      linalg.yield %0 : i32
    }
  return %0 : tensor<i32>
}

// -----
// CHECK: arith.constant dense
// CHECK: tensor.dim
// CHECK: vector.transfer_write
// CHECK-NEXT: scf.for
// CHECK-NEXT: affine.min
// CHECK-NEXT: tensor.extract_slice
// CHECK-NEXT: tensor.extract_slice
// CHECK-NEXT: vector.create_mask
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: arith.maxsi
// CHECK-NEXT: vector.transfer_write
// CHECK-NEXT: tensor.insert_slice
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: vector.multi_reduction
func.func @reduce_max(%arg0: tensor<?xi32>, %arg1: tensor<i32>) -> tensor<i32> {
  %0 = linalg.reduce ins(%arg0 : tensor<?xi32>) outs(%arg1 : tensor<i32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %0 = arith.maxsi %in, %init : i32
      linalg.yield %0 : i32
    }
  return %0 : tensor<i32>
}

// -----
// CHECK: tensor.dim
// CHECK-NEXT: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: affine.min
// CHECK-NEXT: tensor.extract_slice
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK: arith.maxsi
// CHECK: arith.select
// CHECK: vector.transfer_write
// CHECK-NEXT: tensor.insert_slice
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @reduce_max_2d(%arg0: tensor<2x?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {
  %0 = linalg.reduce ins(%arg0 : tensor<2x?xi32>) outs(%arg1 : tensor<?xi32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %0 = arith.maxsi %in, %init : i32
      linalg.yield %0 : i32
    }
  return %0 : tensor<?xi32>
}

// -----
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: arith.maxsi
// CHECK-NEXT: vector.transfer_write
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: scf.yield
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @reduce_max_2d(%arg0: tensor<2x1024xi32>, %arg1: tensor<1024xi32>)  -> tensor<1024xi32> {
  %0 = linalg.reduce ins(%arg0 : tensor<2x1024xi32>) outs(%arg1 : tensor<1024xi32>) dimensions = [0]
    (%in: i32, %init: i32) {
      %0 = arith.maxsi %in, %init : i32
      linalg.yield %0 : i32
    }
  return %0 : tensor<1024xi32>
}
