// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize --split-input-file %s | FileCheck %s

// Tests that arith.minsi and arith.maxsi used for bounds clamping in pointer
// arithmetic are handled correctly by PtrAnalysis.
//
// The typical Triton pattern is:
//   indices = tl.arange(0, BLOCK_SIZE)
//   clamped = tl.minimum(indices, max_val)   # or tl.maximum(indices, min_val)
//   ptr = base_ptr + clamped
//
// When one operand of min/max is a scalar bound and the other is a tensor of
// indices, PtrAnalysis should propagate the tensor's PtrState (offset=0,
// stride=1) and ignore the scalar bound, producing a structured tts.make_tptr.

// Test 1: arith.minsi with scalar bound on RHS.
module {
  tt.func @minsi_scalar_rhs(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<128xf32> {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %bound = tt.splat %arg1 : i32 -> tensor<128xi32>
    %clamped = arith.minsi %range, %bound : tensor<128xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %clamped : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %data = tt.load %ptr : tensor<128x!tt.ptr<f32>>
    tt.return %data : tensor<128xf32>
  }
}

// CHECK-LABEL: tt.func @minsi_scalar_rhs
// CHECK:         [[PTR:%.+]] = tts.make_tptr %arg0 to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
// CHECK:         "tts.load"([[PTR]])

// -----

// Test 2: arith.maxsi with scalar bound on LHS.
module {
  tt.func @maxsi_scalar_lhs(%arg0: !tt.ptr<f32>, %arg1: i32) -> tensor<128xf32> {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %bound = tt.splat %arg1 : i32 -> tensor<128xi32>
    // scalar bound on lhs, range on rhs
    %clamped = arith.maxsi %bound, %range : tensor<128xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %clamped : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %data = tt.load %ptr : tensor<128x!tt.ptr<f32>>
    tt.return %data : tensor<128xf32>
  }
}

// CHECK-LABEL: tt.func @maxsi_scalar_lhs
// CHECK:         [[PTR:%.+]] = tts.make_tptr %arg0 to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
// CHECK:         "tts.load"([[PTR]])

// -----

// Test 3: Combined clamping — maxsi(minsi(range, hi), lo).
// This is the typical clamp(x, lo, hi) pattern used in Triton kernels to
// guard against out-of-bounds accesses.
module {
  tt.func @combined_clamp(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> tensor<128xf32> {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %hi = tt.splat %arg1 : i32 -> tensor<128xi32>
    %lo = tt.splat %arg2 : i32 -> tensor<128xi32>
    // clamp to upper bound
    %clamp_hi = arith.minsi %range, %hi : tensor<128xi32>
    // clamp to lower bound
    %clamp_lo = arith.maxsi %clamp_hi, %lo : tensor<128xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %clamp_lo : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %data = tt.load %ptr : tensor<128x!tt.ptr<f32>>
    tt.return %data : tensor<128xf32>
  }
}

// CHECK-LABEL: tt.func @combined_clamp
// CHECK:         [[PTR:%.+]] = tts.make_tptr %arg0 to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
// CHECK:         "tts.load"([[PTR]])

// -----

// Test 4: arith.minsi with a constant splat (arith.constant dense<N>) on RHS.
// This exercises the path where the scalar bound comes from a constant splat
// rather than a tt.splat of a kernel argument.
module {
  tt.func @minsi_const_splat_rhs(%arg0: !tt.ptr<f32>) -> tensor<128xf32> {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    // constant splat: all elements are 64
    %bound = arith.constant dense<64> : tensor<128xi32>
    %clamped = arith.minsi %range, %bound : tensor<128xi32>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %clamped : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %data = tt.load %ptr : tensor<128x!tt.ptr<f32>>
    tt.return %data : tensor<128xf32>
  }
}

// CHECK-LABEL: tt.func @minsi_const_splat_rhs
// CHECK:         [[PTR:%.+]] = tts.make_tptr %arg0 to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
// CHECK:         "tts.load"([[PTR]])
