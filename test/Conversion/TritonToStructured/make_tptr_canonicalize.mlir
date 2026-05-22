// RUN: triton-shared-opt --split-input-file --canonicalize %s | FileCheck %s

// Test 1: Single dimension with size=1 and stride=0
// The stride should be replaced with 1 since there are no lower dimensions.
module {
  func.func @test_1d_size1_stride0(%arg0: !tt.ptr<f32>) -> tensor<1x!tt.ptr<f32>> {
    %0 = tts.make_tptr %arg0 to sizes: [1], strides: [0], offsets: [0], shape: [0], order: [] : <f32> to tensor<1x!tt.ptr<f32>>
    return %0 : tensor<1x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: func.func @test_1d_size1_stride0
// CHECK:         tts.make_tptr %arg0 to sizes: [1], strides: [1], offsets: [0], shape: [0], order: []
// CHECK-SAME:    : <f32> to tensor<1x!tt.ptr<f32>>

// -----

// Test 2: 2D tensor with outer dim size=1 and stride=0
// sizes: [1, 256], strides: [0, 1]
// For dim 0: size=1, stride=0 -> replace with product of lower dims (256)
// For dim 1: size=256, stride=1 -> keep as is
module {
  func.func @test_2d_outer_size1_stride0(%arg0: !tt.ptr<bf16>) -> tensor<1x256x!tt.ptr<bf16>> {
    %0 = tts.make_tptr %arg0 to sizes: [1, 256], strides: [0, 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16> to tensor<1x256x!tt.ptr<bf16>>
    return %0 : tensor<1x256x!tt.ptr<bf16>>
  }
}

// CHECK-LABEL: func.func @test_2d_outer_size1_stride0
// CHECK:         tts.make_tptr %arg0 to sizes: [1, 256], strides: [256, 1], offsets: [0, 0], shape: [0, 0], order: []
// CHECK-SAME:    : <bf16> to tensor<1x256x!tt.ptr<bf16>>

// -----

// Test 3: 2D tensor with inner dim size=1 and stride=0
// sizes: [128, 1], strides: [1, 0]
// For dim 0: size=128, stride=1 -> keep as is
// For dim 1: size=1, stride=0 -> replace with product of lower dims (1)
module {
  func.func @test_2d_inner_size1_stride0(%arg0: !tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>> {
    %0 = tts.make_tptr %arg0 to sizes: [128, 1], strides: [1, 0], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<128x1x!tt.ptr<f32>>
    return %0 : tensor<128x1x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: func.func @test_2d_inner_size1_stride0
// CHECK:         tts.make_tptr %arg0 to sizes: [128, 1], strides: [1, 1], offsets: [0, 0], shape: [0, 0], order: []
// CHECK-SAME:    : <f32> to tensor<128x1x!tt.ptr<f32>>

// -----

// Test 4: 3D tensor with multiple size=1 dimensions with stride=0
// sizes: [1, 64, 1], strides: [0, 1, 0]
// For dim 0: size=1, stride=0 -> replace with 64*1 = 64
// For dim 1: size=64, stride=1 -> keep as is
// For dim 2: size=1, stride=0 -> replace with 1
module {
  func.func @test_3d_multiple_size1_stride0(%arg0: !tt.ptr<f16>) -> tensor<1x64x1x!tt.ptr<f16>> {
    %0 = tts.make_tptr %arg0 to sizes: [1, 64, 1], strides: [0, 1, 0], offsets: [0, 0, 0], shape: [0, 0, 0], order: [] : <f16> to tensor<1x64x1x!tt.ptr<f16>>
    return %0 : tensor<1x64x1x!tt.ptr<f16>>
  }
}

// CHECK-LABEL: func.func @test_3d_multiple_size1_stride0
// CHECK:         tts.make_tptr %arg0 to sizes: [1, 64, 1], strides: [64, 1, 1], offsets: [0, 0, 0], shape: [0, 0, 0], order: []
// CHECK-SAME:    : <f16> to tensor<1x64x1x!tt.ptr<f16>>

// -----

// Test 5: No canonicalization needed - stride=0 but size!=1
// sizes: [4, 256], strides: [0, 1]
// For dim 0: size=4, stride=0 -> keep as is (size != 1, no canonicalization)
// For dim 1: size=256, stride=1 -> keep as is
module {
  func.func @test_no_canonicalize_size_not_1(%arg0: !tt.ptr<f32>) -> tensor<4x256x!tt.ptr<f32>> {
    %0 = tts.make_tptr %arg0 to sizes: [4, 256], strides: [0, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<4x256x!tt.ptr<f32>>
    return %0 : tensor<4x256x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: func.func @test_no_canonicalize_size_not_1
// CHECK:         tts.make_tptr %arg0 to sizes: [4, 256], strides: [0, 1], offsets: [0, 0], shape: [0, 0], order: []
// CHECK-SAME:    : <f32> to tensor<4x256x!tt.ptr<f32>>

// -----

// Test 6: No canonicalization needed - size=1 but stride!=0
// sizes: [1, 256], strides: [256, 1]
// For dim 0: size=1, stride=256 -> keep as is (stride != 0, no canonicalization)
// For dim 1: size=256, stride=1 -> keep as is
module {
  func.func @test_no_canonicalize_stride_not_0(%arg0: !tt.ptr<f32>) -> tensor<1x256x!tt.ptr<f32>> {
    %0 = tts.make_tptr %arg0 to sizes: [1, 256], strides: [256, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<1x256x!tt.ptr<f32>>
    return %0 : tensor<1x256x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: func.func @test_no_canonicalize_stride_not_0
// CHECK:         tts.make_tptr %arg0 to sizes: [1, 256], strides: [256, 1], offsets: [0, 0], shape: [0, 0], order: []
// CHECK-SAME:    : <f32> to tensor<1x256x!tt.ptr<f32>>

// -----

// Test 7: 4D tensor with mixed cases
// sizes: [1, 32, 1, 64], strides: [0, 64, 0, 1]
// For dim 0: size=1, stride=0 -> replace with 32*1*64 = 2048
// For dim 1: size=32, stride=64 -> keep as is
// For dim 2: size=1, stride=0 -> replace with 64
// For dim 3: size=64, stride=1 -> keep as is
module {
  func.func @test_4d_mixed(%arg0: !tt.ptr<f32>) -> tensor<1x32x1x64x!tt.ptr<f32>> {
    %0 = tts.make_tptr %arg0 to sizes: [1, 32, 1, 64], strides: [0, 64, 0, 1], offsets: [0, 0, 0, 0], shape: [0, 0, 0, 0], order: [] : <f32> to tensor<1x32x1x64x!tt.ptr<f32>>
    return %0 : tensor<1x32x1x64x!tt.ptr<f32>>
  }
}

// CHECK-LABEL: func.func @test_4d_mixed
// CHECK:         tts.make_tptr %arg0 to sizes: [1, 32, 1, 64], strides: [2048, 64, 64, 1], offsets: [0, 0, 0, 0], shape: [0, 0, 0, 0], order: []
// CHECK-SAME:    : <f32> to tensor<1x32x1x64x!tt.ptr<f32>>
