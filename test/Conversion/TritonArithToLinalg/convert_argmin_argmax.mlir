// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @argmax_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %3 = tt.splat %1 : i32 -> tensor<4096xi32>
    %4 = arith.addi %3, %2 : tensor<4096xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    %7 = tt.load %6 : tensor<4096x!tt.ptr<f32>>
    %8:2 = "tt.reduce"(%7, %2) <{axis = 0 : i32}> ({
    ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
      %11 = arith.cmpf oeq, %arg9, %arg11 : f32
      %12 = arith.cmpi slt, %arg10, %arg12 : i32
      %13 = arith.andi %11, %12 : i1
      %14 = arith.cmpf ogt, %arg9, %arg11 : f32
      %15 = arith.ori %14, %13 : i1
      %16 = arith.select %15, %arg9, %arg11 : f32
      %17 = arith.select %15, %arg10, %arg12 : i32
      tt.reduce.return %16, %17 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
    %9 = tt.addptr %arg1, %0 : !tt.ptr<i32>, i32
    tt.store %9, %8#1 : !tt.ptr<i32>
    tt.return
  }
}


// -----

module {
  tt.func public @argmin_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %3 = tt.splat %1 : i32 -> tensor<4096xi32>
    %4 = arith.addi %3, %2 : tensor<4096xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    %7 = tt.load %6 : tensor<4096x!tt.ptr<f32>>
    %8:2 = "tt.reduce"(%7, %2) <{axis = 0 : i32}> ({
    ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
      %11 = arith.cmpf oeq, %arg9, %arg11 : f32
      %12 = arith.cmpi slt, %arg10, %arg12 : i32
      %13 = arith.andi %11, %12 : i1
      %14 = arith.cmpf olt, %arg9, %arg11 : f32
      %15 = arith.ori %14, %13 : i1
      %16 = arith.select %15, %arg9, %arg11 : f32
      %17 = arith.select %15, %arg10, %arg12 : i32
      tt.reduce.return %16, %17 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
    %9 = tt.addptr %arg1, %0 : !tt.ptr<i32>, i32
    tt.store %9, %8#1 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @argmax_012(
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<i32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0xFF800000 : f32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ARG6]], %[[ARG2]] : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4096xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[MULI_0]] : i32) outs(%[[EMPTY_1]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[FILL_0]], %[[GENERIC_0]] : tensor<4096xi32>, tensor<4096xi32>) outs(%[[FILL_0]] : tensor<4096xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_1]] : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>) outs(%[[SPLAT_0]] : tensor<4096x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !tt.ptr<f32>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_4]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_2]] : tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_2]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_3]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[REDUCE_0:.*]]:2 = linalg.reduce ins(%[[LOAD_0]], %[[GENERIC_0]] : tensor<4096xf32>, tensor<4096xi32>) outs(%[[FILL_1]], %[[FILL_2]] : tensor<f32>, tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: f32, %[[VAL_10:.*]]: i32) {
// CHECK:               %[[CMPF_0:.*]] = arith.cmpf oeq, %[[VAL_7]], %[[VAL_9]] : f32
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_10]] : i32
// CHECK:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_0]], %[[CMPI_0]] : i1
// CHECK:               %[[CMPF_1:.*]] = arith.cmpf ogt, %[[VAL_7]], %[[VAL_9]] : f32
// CHECK:               %[[ORI_0:.*]] = arith.ori %[[CMPF_1]], %[[ANDI_0]] : i1
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[VAL_7]], %[[VAL_9]] : f32
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[ORI_0]], %[[VAL_8]], %[[VAL_10]] : i32
// CHECK:               linalg.yield %[[SELECT_0]], %[[SELECT_1]] : f32, i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]]#1[] : tensor<i32>
// CHECK:           %[[ADDPTR_1:.*]] = tt.addptr %[[ARG1]], %[[ARG6]] : !tt.ptr<i32>, i32
// CHECK:           tt.store %[[ADDPTR_1]], %[[EXTRACT_0]] : !tt.ptr<i32>
// CHECK:           return
// CHECK:         }

// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @argmin_012(
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<i32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32, %[[ARG6:.*]]: i32, %[[ARG7:.*]]: i32, %[[ARG8:.*]]: i32) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0x7F800000 : f32
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ARG6]], %[[ARG2]] : i32
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]]], iterator_types = ["parallel"]} outs(%[[EMPTY_0]] : tensor<4096xi32>) {
// CHECK:           ^bb0(%[[VAL_0:.*]]: i32):
// CHECK:             %[[INDEX_0:.*]] = linalg.index 0 : index
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[INDEX_0]] : index to i32
// CHECK:             linalg.yield %[[INDEX_CAST_0]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           %[[EMPTY_1:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[MULI_0]] : i32) outs(%[[EMPTY_1]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel"]} ins(%[[FILL_0]], %[[GENERIC_0]] : tensor<4096xi32>, tensor<4096xi32>) outs(%[[FILL_0]] : tensor<4096xi32>) {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:             linalg.yield %[[ADDI_0]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           %[[SPLAT_0:.*]] = tensor.splat %[[ARG0]] : tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel"]} ins(%[[SPLAT_0]], %[[GENERIC_1]] : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>) outs(%[[SPLAT_0]] : tensor<4096x!tt.ptr<f32>>) {
// CHECK:           ^bb0(%[[VAL_4:.*]]: !tt.ptr<f32>, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: !tt.ptr<f32>):
// CHECK:             %[[ADDPTR_0:.*]] = tt.addptr %[[VAL_4]], %[[VAL_5]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield %[[ADDPTR_0]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[LOAD_0:.*]] = tt.load %[[GENERIC_2]] : tensor<4096x!tt.ptr<f32>>
// CHECK:           %[[EMPTY_2:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[FILL_1:.*]] = linalg.fill ins(%[[CONSTANT_1]] : f32) outs(%[[EMPTY_2]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[EMPTY_3:.*]] = tensor.empty() : tensor<i32>
// CHECK:           %[[FILL_2:.*]] = linalg.fill ins(%[[CONSTANT_0]] : i32) outs(%[[EMPTY_3]] : tensor<i32>) -> tensor<i32>
// CHECK:           %[[REDUCE_0:.*]]:2 = linalg.reduce ins(%[[LOAD_0]], %[[GENERIC_0]] : tensor<4096xf32>, tensor<4096xi32>) outs(%[[FILL_1]], %[[FILL_2]] : tensor<f32>, tensor<i32>) dimensions = [0]
// CHECK:             (%[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: f32, %[[VAL_10:.*]]: i32) {
// CHECK:               %[[CMPF_0:.*]] = arith.cmpf oeq, %[[VAL_7]], %[[VAL_9]] : f32
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_10]] : i32
// CHECK:               %[[ANDI_0:.*]] = arith.andi %[[CMPF_0]], %[[CMPI_0]] : i1
// CHECK:               %[[CMPF_1:.*]] = arith.cmpf olt, %[[VAL_7]], %[[VAL_9]] : f32
// CHECK:               %[[ORI_0:.*]] = arith.ori %[[CMPF_1]], %[[ANDI_0]] : i1
// CHECK:               %[[SELECT_0:.*]] = arith.select %[[ORI_0]], %[[VAL_7]], %[[VAL_9]] : f32
// CHECK:               %[[SELECT_1:.*]] = arith.select %[[ORI_0]], %[[VAL_8]], %[[VAL_10]] : i32
// CHECK:               linalg.yield %[[SELECT_0]], %[[SELECT_1]] : f32, i32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]]#1[] : tensor<i32>
// CHECK:           %[[ADDPTR_1:.*]] = tt.addptr %[[ARG1]], %[[ARG6]] : !tt.ptr<i32>, i32
// CHECK:           tt.store %[[ADDPTR_1]], %[[EXTRACT_0]] : !tt.ptr<i32>
// CHECK:           return
// CHECK:         }

