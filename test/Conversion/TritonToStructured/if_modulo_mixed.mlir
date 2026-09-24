// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

// Test that pointer state with modulo inside scf.if is skipped.

module {
  tt.func @kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %cond: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %c16 = arith.constant 16 : i32
    %c256 = arith.constant 256 : i32

    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %splat_c16 = tt.splat %c16 : i32 -> tensor<256xi32>
    %splat_c256 = tt.splat %c256 : i32 -> tensor<256xi32>

    %if_result:2 = scf.if %cond -> (tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>) {
      %splat_structured = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      %ptr_structured = tt.addptr %splat_structured, %range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

      %splat_unstructured = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      %mod_range = arith.remsi %range, %splat_c16 : tensor<256xi32>
      %ptr_unstructured = tt.addptr %splat_unstructured, %mod_range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

      scf.yield %ptr_structured, %ptr_unstructured : tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
    } else {
      %splat_structured = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      %offset = arith.addi %range, %splat_c256 : tensor<256xi32>
      %ptr_structured = tt.addptr %splat_structured, %offset : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

      %splat_unstructured = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
      %mod_range = arith.remsi %range, %splat_c16 : tensor<256xi32>
      %ptr_unstructured = tt.addptr %splat_unstructured, %mod_range : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

      scf.yield %ptr_structured, %ptr_unstructured : tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
    }

    %output_ptr = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %output_offset = arith.muli %range, %splat_c16 : tensor<256xi32>
    %output = tt.addptr %output_ptr, %output_offset : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

    %loop_result:2 = scf.for %i = %c0 to %c10 step %c1
      iter_args(%ptr_structured_iter = %if_result#0, %ptr_unstructured_iter = %if_result#1)
      -> (tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>) {

      // Load from both pointers
      %data1 = tt.load %ptr_structured_iter : tensor<256x!tt.ptr<f32>>
      %data2 = tt.load %ptr_unstructured_iter : tensor<256x!tt.ptr<f32>>

      // Store to prevent elimination
      %sum = arith.addf %data1, %data2 : tensor<256xf32>
      tt.store %output, %sum : tensor<256x!tt.ptr<f32>>

      %c256_offset = tt.splat %c256 : i32 -> tensor<256xi32>
      %ptr_structured_next = tt.addptr %ptr_structured_iter, %c256_offset : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

      %c16_offset = tt.splat %c16 : i32 -> tensor<256xi32>
      %ptr_unstructured_next = tt.addptr %ptr_unstructured_iter, %c16_offset : tensor<256x!tt.ptr<f32>>, tensor<256xi32>

      scf.yield %ptr_structured_next, %ptr_unstructured_next : tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
    }

    %final_data1 = tt.load %loop_result#0 : tensor<256x!tt.ptr<f32>>
    %final_data2 = tt.load %loop_result#1 : tensor<256x!tt.ptr<f32>>
    %final_sum = arith.addf %final_data1, %final_data2 : tensor<256xf32>
    tt.store %output, %final_sum : tensor<256x!tt.ptr<f32>>

    tt.return
  }
}

// CHECK-LABEL: tt.func @kernel
// CHECK-DAG: %[[CST_256:.*]] = arith.constant dense<256> : tensor<256xi32>
// CHECK-DAG: %[[CST_16:.*]] = arith.constant dense<16> : tensor<256xi32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C10:.*]] = arith.constant 10 : index
// CHECK: %[[RANGE:.*]] = tt.make_range

// Verify the scf.if exists and returns two pointers (both as tensor<256x!tt.ptr<f32>>)
// CHECK: %[[IF_RESULT:.*]]:2 = scf.if
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
// CHECK: } else {
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
// CHECK: }

// CHECK-NOT: tts.make_tptr %{{.*}} to sizes: [256], strides: [%{{.*}}], offsets: [%[[IF_RESULT]]

// CHECK: %[[OUTPUT_PTR:.*]] = tts.make_tptr {{.*}} to sizes: [256], strides: [{{%c16|16}}], offsets: [{{%c0|0}}]

// CHECK: %[[LOOP_RESULT:.*]]:2 = scf.for {{.*}} iter_args(%[[ARG5:.*]] = %[[IF_RESULT]]#0, %[[ARG6:.*]] = %[[IF_RESULT]]#1)

// CHECK: tt.load %[[ARG5]] : tensor<256x!tt.ptr<f32>>
// CHECK: tt.load %[[ARG6]] : tensor<256x!tt.ptr<f32>>

// CHECK: "tts.store"(%[[OUTPUT_PTR]],

// CHECK: tt.addptr %[[ARG5]], %[[CST_256]]
// CHECK: tt.addptr %[[ARG6]], %[[CST_16]]
// CHECK: scf.yield {{.*}}, {{.*}} : tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>

// CHECK: tt.load %[[LOOP_RESULT]]#0 : tensor<256x!tt.ptr<f32>>
// CHECK: tt.load %[[LOOP_RESULT]]#1 : tensor<256x!tt.ptr<f32>>
// CHECK: "tts.store"
// CHECK: tt.return
