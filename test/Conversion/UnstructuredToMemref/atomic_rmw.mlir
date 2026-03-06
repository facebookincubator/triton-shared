// RUN: triton-shared-opt --triton-to-unstructured --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s
//

module {

  // -----------------------------------------------------------------------
  // Test 1: fadd without mask.
  // Expected: linalg.generic with memref.atomic_rmw addf, no scf.if.
  // -----------------------------------------------------------------------
  // CHECK-LABEL: tt.func public @atomic_fadd_no_mask
  // CHECK-NOT:   tt.atomic_rmw
  // CHECK-NOT:   tts.atomic_rmw
  // CHECK:       [[CAST:%.+]] = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
  // CHECK:       [[BASE:%.+]] = memref.cast [[CAST]] : memref<*xf32> to memref<?xf32>
  // CHECK:       linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]], iterator_types = ["parallel"]} ins(%arg2, %arg1 : tensor<64xi32>, tensor<64xf32>)
  // CHECK:       ^bb0([[IDX:%.+]]: i32, [[VAL:%.+]]: f32, {{.*}}: f32):
  // CHECK:         [[I:%.+]] = arith.index_cast [[IDX]] : i32 to index
  // CHECK:         [[OLD:%.+]] = memref.atomic_rmw addf [[VAL]], [[BASE]]{{\[}}[[I]]{{\]}} : (f32, memref<?xf32>) -> f32
  // CHECK:         linalg.yield [[OLD]] : f32
  tt.func public @atomic_fadd_no_mask(
      %out_ptr: !tt.ptr<f32>,
      %values: tensor<64xf32>,
      %offsets: tensor<64xi32>
  ) -> tensor<64xf32> {
    %splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %ptr   = tt.addptr %splat, %offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %old   = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %values
                 : (tensor<64x!tt.ptr<f32>>, tensor<64xf32>) -> tensor<64xf32>
    tt.return %old : tensor<64xf32>
  }

  // -----------------------------------------------------------------------
  // Test 2: fadd with mask.
  // Expected: linalg.generic with scf.if(mask) { memref.atomic_rmw addf }.
  // -----------------------------------------------------------------------
  // CHECK-LABEL: tt.func public @atomic_fadd_with_mask
  // CHECK-NOT:   tt.atomic_rmw
  // CHECK-NOT:   tts.atomic_rmw
  // CHECK:       [[CAST2:%.+]] = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
  // CHECK:       [[BASE2:%.+]] = memref.cast [[CAST2]] : memref<*xf32> to memref<?xf32>
  // CHECK:       linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]], #[[MAP]]], iterator_types = ["parallel"]} ins(%arg2, %arg1, %arg3 : tensor<64xi32>, tensor<64xf32>, tensor<64xi1>)
  // CHECK:       ^bb0([[IDX2:%.+]]: i32, [[VAL2:%.+]]: f32, [[MASK2:%.+]]: i1, [[OUT2:%.+]]: f32):
  // CHECK:         [[RES2:%.+]] = scf.if [[MASK2]] -> (f32) {
  // CHECK:           [[I2:%.+]] = arith.index_cast [[IDX2]] : i32 to index
  // CHECK:           [[OLD2:%.+]] = memref.atomic_rmw addf [[VAL2]], [[BASE2]]{{\[}}[[I2]]{{\]}} : (f32, memref<?xf32>) -> f32
  // CHECK:           scf.yield [[OLD2]] : f32
  // CHECK:         } else {
  // CHECK:           scf.yield [[OUT2]] : f32
  // CHECK:         }
  // CHECK:         linalg.yield [[RES2]] : f32
  tt.func public @atomic_fadd_with_mask(
      %out_ptr: !tt.ptr<f32>,
      %values: tensor<64xf32>,
      %offsets: tensor<64xi32>,
      %mask: tensor<64xi1>
  ) -> tensor<64xf32> {
    %splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %ptr   = tt.addptr %splat, %offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %old   = tt.atomic_rmw fadd, acq_rel, gpu, %ptr, %values, %mask
                 : (tensor<64x!tt.ptr<f32>>, tensor<64xf32>, tensor<64xi1>)
                 -> tensor<64xf32>
    tt.return %old : tensor<64xf32>
  }

  // -----------------------------------------------------------------------
  // Test 3: integer addi without mask.
  // Expected: linalg.generic with memref.atomic_rmw addi, no scf.if.
  // -----------------------------------------------------------------------
  // CHECK-LABEL: tt.func public @atomic_addi_no_mask
  // CHECK-NOT:   tt.atomic_rmw
  // CHECK-NOT:   tts.atomic_rmw
  // CHECK:       memref.atomic_rmw addi {{.*}}, {{.*}}[{{.*}}] : (i32, memref<?xi32>) -> i32
  tt.func public @atomic_addi_no_mask(
      %out_ptr: !tt.ptr<i32>,
      %values: tensor<64xi32>,
      %offsets: tensor<64xi32>
  ) -> tensor<64xi32> {
    %splat = tt.splat %out_ptr : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
    %ptr   = tt.addptr %splat, %offsets : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
    %old   = tt.atomic_rmw add, acq_rel, gpu, %ptr, %values
                 : (tensor<64x!tt.ptr<i32>>, tensor<64xi32>) -> tensor<64xi32>
    tt.return %old : tensor<64xi32>
  }

}
