// RUN: triton-shared-opt --verify-triton-checkpoint --split-input-file --verify-diagnostics %s

module {
  func.func @clean_kernel(%arg0: f32, %arg1: f32) -> f32 {
    %0 = arith.addf %arg0, %arg1 : f32
    return %0 : f32
  }
}

// -----

module {
  func.func @kernel_with_residual_triton_op(%arg0: f32) {
    // expected-error @+1 {{'tt.splat' op unexpected Triton-related op remaining after lowering}}
    %0 = tt.splat %arg0 : f32 -> tensor<256xf32>
    return
  }
}
