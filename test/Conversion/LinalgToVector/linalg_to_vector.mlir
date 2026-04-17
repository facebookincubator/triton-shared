// RUN: triton-shared-opt --split-input-file --linalg-to-vector -cse -canonicalize %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @truncf(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf16>) {
    // CHECK: scf.for
    // CHECK-NEXT: memref.subview
    // CHECK-NEXT: memref.subview
    // CHECK-NEXT: memref.subview
    // CHECK-NEXT: vector.transfer_read
    // CHECK-NEXT: arith.truncf
    // CHECK-NEXT: memref.subview
    // CHECK-NEXT: vector.transfer_write
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<128x128xf32>) outs(%arg1 : memref<128x128xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return
  }
  func.func @truncf2(%arg0: memref<128x?xf32>, %arg1: memref<128x?xf16>) {
    // CHECK: memref.dim
    // CHECK-NEXT: scf.for
    // CHECK-NEXT: scf.for
    // CHECK-NEXT: affine.min
    // CHECK: vector.create_mask
    // CHECK: vector.transfer_read
    // CHECK-NEXT: arith.truncf
    // CHECK: vector.transfer_write
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : memref<128x?xf32>) outs(%arg1 : memref<128x?xf16>) {
    ^bb0(%in: f32, %out: f16):
      %0 = arith.truncf %in : f32 to f16
      linalg.yield %0 : f16
    }
    return
  }
}

// -----
#map = affine_map<(d0) -> (d0)>
module {
  func.func @sigmoid(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
    // CHECK: vector.transfer_read
    // CHECK-NEXT: arith.negf
    // CHECK-NEXT: math.exp
    // CHECK-NEXT: arith.addf
    // CHECK-NEXT: arith.divf
    // CHECK-NEXT: vector.transfer_write
    %cst = arith.constant 1.000000e+00 : f32
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<128xf32>) outs(%arg1 : memref<128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.negf %in : f32
      %1 = math.exp %0 : f32
      %2 = arith.addf %1, %cst : f32
      %3 = arith.divf %cst, %2 : f32
      linalg.yield %3 : f32
    }
    return
  }
  func.func @sigmoid2(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    // CHECK: memref.dim
    // CHECK-NEXT: scf.for
    // CHECK-NEXT: affine.min
    // CHECK: vector.create_mask
    // CHECK: vector.transfer_read
    // CHECK-NEXT: arith.negf
    // CHECK-NEXT: math.exp
    // CHECK-NEXT: arith.addf
    // CHECK-NEXT: arith.divf
    // CHECK: vector.transfer_write
    %cst = arith.constant 1.000000e+00 : f32
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<?xf32>) outs(%arg1 : memref<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.negf %in : f32
      %1 = math.exp %0 : f32
      %2 = arith.addf %1, %cst : f32
      %3 = arith.divf %cst, %2 : f32
      linalg.yield %3 : f32
    }
    return
  }
}

// -----
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK: arith.select
// CHECK: vector.transfer_write
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4, d5, d6, d7, d8)>
module {
  func.func @select() -> memref<2x2x2x2x2x2x2x2x2xi32> {
    %alloc = memref.alloc() : memref<2x2x2x2x2x2x2x2x2xi1>
    %alloc_0 = memref.alloc() : memref<2x2x2x2x2x2x2x2x2xi32>
    %alloc_1 = memref.alloc() : memref<2x2x2x2x2x2x2x2x2xi32>
    linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%alloc, %alloc_0, %alloc_1 : memref<2x2x2x2x2x2x2x2x2xi1>, memref<2x2x2x2x2x2x2x2x2xi32>, memref<2x2x2x2x2x2x2x2x2xi32>) outs(%alloc_0 : memref<2x2x2x2x2x2x2x2x2xi32>) {
    ^bb0(%in: i1, %in_2: i32, %in_3: i32, %out: i32):
      %0 = arith.select %in, %in_2, %in_3 : i32
      linalg.yield %0 : i32
    }
    return %alloc_0 : memref<2x2x2x2x2x2x2x2x2xi32>
  }
}



// -----

// CHECK: scf.for
// CHECK-NEXT: memref.subview
// CHECK-NEXT: memref.subview
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: memref.load
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: arith.addi
// CHECK-NEXT: vector.transfer_write
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (0)>
module {
  func.func @broadcast(%arg0: memref<1024xi32>, %arg1: memref<1xi32>) -> memref<1024xi32> {
    %0 = memref.alloc() : memref<1024xi32>
    linalg.generic {
        indexing_maps = [#map, #map1, #map],
        iterator_types = ["parallel"]
    } ins(%arg0, %arg1 : memref<1024xi32>, memref<1xi32>)
      outs(%0 : memref<1024xi32>) {
      ^bb0(%in_0: i32, %in_1: i32, %out: i32):
        %1 = arith.addi %in_0, %in_1 : i32
        linalg.yield %1 : i32
    }
    return %0 : memref<1024xi32>
  }
}


// -----

// CHECK: arith.constant dense<10> : vector<128xi32>
// CHECK: scf.for
// CHECK-NEXT: memref.subview
// CHECK-NEXT: memref.subview
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: arith.addi
// CHECK-NEXT: vector.transfer_write
#map = affine_map<(d0) -> (d0)>
module {
  func.func @broadcast(%arg0: memref<1024xi32>) -> memref<1024xi32> {
    %0 = memref.alloc() : memref<1024xi32>
    %c0 = arith.constant 10 : i32
    linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel"]
    } ins(%arg0 : memref<1024xi32>)
      outs(%0 : memref<1024xi32>) {
      ^bb0(%in: i32, %out: i32):
        %1 = arith.addi %in, %c0 : i32
        linalg.yield %1 : i32
    }
    return %0 : memref<1024xi32>
  }
}


// -----

// CHECK: scf.for
// CHECK-NEXT: memref.subview
// CHECK-NEXT: vector.step
// CHECK-NEXT: vector.broadcast
// CHECK-NEXT: arith.addi
// CHECK-NEXT: arith.index_cast
// CHECK-NEXT: vector.transfer_write
#map = affine_map<(d0) -> (d0)>
module {
  func.func @make_range() {
    %11 = memref.alloc() : memref<1024xi32>
    linalg.generic {
      indexing_maps = [#map],
      iterator_types = ["parallel"]
    } outs(%11 : memref<1024xi32>) {
      ^bb0(%out: i32):
        %21 = linalg.index 0 : index
        %22 = arith.index_cast %21 : index to i32
        linalg.yield %22 : i32
    }
    return
  }
}
// -----
module {
  func.func @add() {
    %l1_lhs = memref.alloc() : memref<2048xf32>
    %l1_rhs = memref.alloc() : memref<2048xf32>
    %l1_out = memref.alloc() : memref<2048xf32>
    // CHECK: scf.for
    // CHECK: vector.transfer_read
    // CHECK-NEXT: vector.transfer_read
    // CHECK-NEXT: arith.addf
    // CHECK-NEXT: vector.transfer_write
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%l1_lhs, %l1_rhs : memref<2048xf32>, memref<2048xf32>)
      outs(%l1_out : memref<2048xf32>) {
      ^bb0(%arg0: f32, %arg1: f32, %arg2: f32):
        %out = arith.addf %arg0, %arg1 : f32
        linalg.yield %out : f32
    }
    return
  }
}

// -----
module {
  func.func @exp() {
    %l1_lhs = memref.alloc() : memref<2048xf32>
    %l1_out = memref.alloc() : memref<2048xf32>
    // CHECK: scf.for
    // CHECK: vector.transfer_read
    // CHECK-NEXT: math.exp
    // CHECK-NEXT: vector.transfer_write
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%l1_lhs : memref<2048xf32>)
      outs(%l1_out : memref<2048xf32>) {
      ^bb0(%arg0: f32, %arg2: f32):
        %out = math.exp %arg0 : f32
        linalg.yield %out : f32
    }
    return
  }
}

// -----
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK: arith.index_cast
// CHECK: vector.broadcast
// CHECK-NEXT: vector.transfer_write
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @make_range_2D() {
    %11 = memref.alloc() : memref<64x1024xi32>
    linalg.generic {
      indexing_maps = [#map],
      iterator_types = ["parallel", "parallel"]
    } outs(%11 : memref<64x1024xi32>) {
      ^bb0(%out: i32):
        %21 = linalg.index 0 : index
        %22 = arith.index_cast %21 : index to i32
        linalg.yield %22 : i32
    }

    return
  }
}


// -----
// CHECK: scf.for
// CHECK: memref.copy
// CHECK: memref.copy
// CHECK-NEXT: scf.for
// CHECK: vector.transfer_read
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: arith.addf
// CHECK-NEXT: vector.transfer_write
// CHECK: memref.copy
#map = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @elementwise_split(%arg0: memref<2x12x384x384xf16>, %arg1: memref<2x12x384x384xf16>, %arg2: memref<2x12x384x384xf16>) {
    %alloc = memref.alloc(): memref<65536xf16>
    %alloc_0 = memref.alloc(): memref<65536xf16>
    %alloc_1 = memref.alloc(): memref<65536xf16>
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3]] : memref<2x12x384x384xf16> into memref<3538944xf16>
    %collapse_shape_2 = memref.collapse_shape %arg1 [[0, 1, 2, 3]] : memref<2x12x384x384xf16> into memref<3538944xf16>
    %collapse_shape_3 = memref.collapse_shape %arg2 [[0, 1, 2, 3]] : memref<2x12x384x384xf16> into memref<3538944xf16>
    %c3538944 = arith.constant 3538944 : index
    %c0 = arith.constant 0 : index
    %c65536 = arith.constant 65536 : index
    scf.for %arg3 = %c0 to %c3538944 step %c65536 {
      %subview = memref.subview %collapse_shape[%arg3] [65536] [1] : memref<3538944xf16> to memref<65536xf16, #map>
      memref.copy %subview, %alloc_1 : memref<65536xf16, #map> to memref<65536xf16>
      %subview_4 = memref.subview %collapse_shape_2[%arg3] [65536] [1] : memref<3538944xf16> to memref<65536xf16, #map>
      memref.copy %subview_4, %alloc_0 : memref<65536xf16, #map> to memref<65536xf16>
      linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%alloc_1, %alloc_0 : memref<65536xf16>, memref<65536xf16>) outs(%alloc : memref<65536xf16>) {
      ^bb0(%in: f16, %in_6: f16, %out: f16):
        %0 = arith.addf %in, %in_6 : f16
        linalg.yield %0 : f16
      }
      %subview_5 = memref.subview %collapse_shape_3[%arg3] [65536] [1] : memref<3538944xf16> to memref<65536xf16, #map>
      memref.copy %subview_5, %alloc : memref<65536xf16, #map> to memref<65536xf16>
    } {iterator_type = "parallel"}
    return
  }
}

// -----
// CHECK: scf.for
// CHECK-NEXT: affine.min
// CHECK: vector.create_mask
// CHECK: vector.transfer_read
// CHECK: vector.transfer_read
// CHECK-NEXT: arith.addf
// CHECK: vector.transfer_write
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @add_2D() {
    %alloc = memref.alloc(): memref<31x17xf32>
    %alloc_0 = memref.alloc(): memref<31x17xf32>
    %alloc_1 = memref.alloc(): memref<31x17xf32>
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_1, %alloc_0 : memref<31x17xf32>, memref<31x17xf32>) outs(%alloc : memref<31x17xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %0 = arith.addf %in, %in_2 : f32
      linalg.yield %0 : f32
    }
    return
  }
}

// -----

// CHECK: scf.for
// CHECK-NEXT: vector.transfer_read
// CHECK-NEXT: arith.index_cast
// CHECK-NEXT: vector.gather
// CHECK-NEXT: vector.transfer_write
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @gather(%arg0: tensor<?xf32>, %arg1: tensor<32x32xi32>, %arg2: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<32x32xi32>) outs(%arg2 : tensor<32x32xf32>) {
    ^bb0(%in: i32, %out: f32):
      %1 = arith.index_cast %in : i32 to index
      %extracted = tensor.extract %arg0[%1] : tensor<?xf32>
      linalg.yield %extracted : f32
    } -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}