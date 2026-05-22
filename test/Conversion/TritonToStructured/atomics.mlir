// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

// Positive test: successfully lower to tts.atomic_rmw and tts.make_tptr
tt.func public @atomic_add_1d(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) attributes {noinline = false} {
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<true> : tensor<16xi1>
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<16xf32>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
    %3 = tt.atomic_rmw fadd, acq_rel, gpu, %2, %cst_0, %cst : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
  }
  tt.return
}

// CHECK-LABEL:   tt.func public @atomic_add_1d(
// CHECK-SAME:      {{.*}}: !tt.ptr<f32> {tt.divisibility = 32 : i32}) attributes {noinline = false} {
// CHECK:           %[[TPTR:.*]] = tts.make_tptr {{.*}} to sizes: [16], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<16x!tt.ptr<f32>>
// CHECK:           scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}}  : i32 {
// CHECK:             {{.*}} = tts.atomic_rmw fadd, acq_rel, gpu, %[[TPTR]], {{.*}} {static_mask_dims = array<i64>} : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>) -> tensor<16xf32>
// CHECK:           }


// Test different op (atomic xchg) to validate we lower the "op" field of the RMW correctly
tt.func public @atomic_xchg_1d(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) attributes {noinline = false} {
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<true> : tensor<16xi1>
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<16xf32>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
    %3 = tt.atomic_rmw exch, acq_rel, gpu, %2, %cst_0, %cst : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
  }
  tt.return
}

// CHECK-LABEL:   tt.func public @atomic_xchg_1d(
// CHECK-SAME:      {{.*}}: !tt.ptr<f32> {tt.divisibility = 32 : i32}) attributes {noinline = false} {
// CHECK:           %[[TPTR:.*]] = tts.make_tptr {{.*}} to sizes: [16], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<16x!tt.ptr<f32>>
// CHECK:           scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}}  : i32 {
// CHECK:             {{.*}} = tts.atomic_rmw exch, acq_rel, gpu, %[[TPTR]], {{.*}} {static_mask_dims = array<i64>} : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>) -> tensor<16xf32>
// CHECK:           }


// 2D case
tt.func public @atomic_add_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) attributes {noinline = false} {
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<true> : tensor<8x16xi1>
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<8x16xf32>
  %cst_1 = arith.constant dense<16> : tensor<8x1xi32>
  %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %3 = arith.muli %2, %cst_1 : tensor<8x1xi32>
  %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x1x!tt.ptr<f32>>
  %5 = tt.addptr %4, %3 : tensor<8x1x!tt.ptr<f32>>, tensor<8x1xi32>
  %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %7 = tt.broadcast %5 : tensor<8x1x!tt.ptr<f32>> -> tensor<8x16x!tt.ptr<f32>>
  %8 = tt.broadcast %6 : tensor<1x16xi32> -> tensor<8x16xi32>
  %9 = tt.addptr %7, %8 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
  scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
    %10 = tt.atomic_rmw fadd, acq_rel, gpu, %9, %cst_0, %cst : (tensor<8x16x!tt.ptr<f32>>, tensor<8x16xf32>, tensor<8x16xi1>) -> tensor<8x16xf32>
  }
  tt.return
}

// CHECK-LABEL:   tt.func public @atomic_add_2d(
// CHECK-SAME:      {{.*}}: !tt.ptr<f32> {tt.divisibility = 32 : i32}) attributes {noinline = false} {
// CHECK:           %[[TPTR:.*]] = tts.make_tptr {{.*}} to sizes: [8, 16], strides: {{\[}}{{.*}}, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<8x16x!tt.ptr<f32>>
// CHECK:           scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}}  : i32 {
// CHECK:             {{.*}} = tts.atomic_rmw fadd, acq_rel, gpu, %[[TPTR]], {{.*}} {static_mask_dims = array<i64>} : (tensor<8x16x!tt.ptr<f32>>, tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK:           }


// Negative test - can't prove the region is rectangular so shouldn't lower to tts.atomic_rmw
tt.func public @atomic_add_1d_no_lower(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}, %arg1: tensor<16xi32>) attributes {noinline = false} {
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<true> : tensor<16xi1>
  %cst_0 = arith.constant dense<1.000000e+00> : tensor<16xf32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %2 = tt.addptr %1, %arg1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  scf.for %arg2 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
    %3 = tt.atomic_rmw fadd, acq_rel, gpu, %2, %cst_0, %cst : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
  }
  tt.return
}

// CHECK-LABEL:   tt.func public @atomic_add_1d_no_lower(
// CHECK-SAME:      {{.*}}: !tt.ptr<f32> {tt.divisibility = 32 : i32},
// CHECK-SAME:      {{.*}}: tensor<16xi32>) attributes {noinline = false} {
// CHECK:           %[[PTR:.*]] = tt.addptr {{.*}}, {{.*}} : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
// CHECK:           scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}}  : i32 {
// CHECK:             {{.*}} = tt.atomic_rmw fadd, acq_rel, gpu, %[[PTR]], {{.*}}, {{.*}} : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
// CHECK:           }


// Back-to-back atomic add - an atomic returns the value that was in memory before the op runs
// Use the return value in another atomic to make sure that lowering works
tt.func public @atomic_add_1d_back_to_back(%arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) attributes {noinline = false} {
  %c1_i32 = arith.constant 1 : i32
  %c10_i32 = arith.constant 10 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<true> : tensor<16xi1>
  %cst_0 = arith.constant dense<1.000000e-01> : tensor<16xf32>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
  scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
    %3 = tt.atomic_rmw fadd, acq_rel, gpu, %2, %cst_0, %cst : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
    %4 = tt.atomic_rmw fadd, acq_rel, gpu, %2, %3, %cst : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
  }
  tt.return
}

// CHECK-LABEL:   tt.func public @atomic_add_1d_back_to_back(
// CHECK-SAME:      {{.*}}: !tt.ptr<f32> {tt.divisibility = 32 : i32}) attributes {noinline = false} {
// CHECK:           %[[TPTR:.*]] = tts.make_tptr {{.*}} to sizes: [16], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<16x!tt.ptr<f32>>
// CHECK:           scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}}  : i32 {
// CHECK:             %[[RETVAL:.*]] = tts.atomic_rmw fadd, acq_rel, gpu, %[[TPTR]], {{.*}} {static_mask_dims = array<i64>} : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>) -> tensor<16xf32>
// CHECK:             {{.*}} = tts.atomic_rmw fadd, acq_rel, gpu, %[[TPTR]], %[[RETVAL]] {static_mask_dims = array<i64>} : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>) -> tensor<16xf32>
// CHECK:           }


// Atomic CAS - uses a different op than RMW
tt.func public @atomic_cas_1d(%arg0: !tt.ptr<i32> {tt.divisibility = 32 : i32}) attributes {noinline = false} {
  %cst_0 = arith.constant dense<0> : tensor<16xi32>
  %cst_1 = arith.constant dense<1> : tensor<16xi32>
  %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
  %7 = tt.atomic_cas relaxed, gpu, %2, %cst_0, %cst_1 : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
  tt.return
}

// CHECK-LABEL:   tt.func public @atomic_cas_1d(
// CHECK-SAME:      {{.*}}: !tt.ptr<i32> {tt.divisibility = 32 : i32}) attributes {noinline = false} {
// CHECK:           %[[TPTR:.*]] = tts.make_tptr {{.*}} to sizes: [16], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<16x!tt.ptr<i32>>
// CHECK:           {{.*}} = tts.atomic_cas relaxed, gpu, %[[TPTR]], {{.*}}, {{.*}} : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
// CHECK:         }
