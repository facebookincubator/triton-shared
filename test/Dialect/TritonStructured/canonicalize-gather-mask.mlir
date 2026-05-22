// RUN: triton-shared-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: @gather_strip_dense_true_mask
// CHECK: tts.gather %{{.*}}[%{{.*}}] :
// CHECK-NOT: mask =
tt.func @gather_strip_dense_true_mask(%ptr: !tt.ptr<f16>, %idx: tensor<64xi32>) -> tensor<64xf16> {
  %true = arith.constant dense<true> : tensor<64xi1>
  %0 = tts.gather %ptr[%idx] mask = %true : (!tt.ptr<f16>, tensor<64xi32>) -> tensor<64xf16>
  tt.return %0 : tensor<64xf16>
}

// CHECK-LABEL: @gather_strip_splat_true_mask
// CHECK: tts.gather %{{.*}}[%{{.*}}] :
// CHECK-NOT: mask =
tt.func @gather_strip_splat_true_mask(%ptr: !tt.ptr<f32>, %idx: tensor<32xi32>) -> tensor<32xf32> {
  %true_scalar = arith.constant true
  %true_mask = tt.splat %true_scalar : i1 -> tensor<32xi1>
  %0 = tts.gather %ptr[%idx] mask = %true_mask : (!tt.ptr<f32>, tensor<32xi32>) -> tensor<32xf32>
  tt.return %0 : tensor<32xf32>
}

// CHECK-LABEL: @gather_strip_broadcast_true_mask
// CHECK: tts.gather %{{.*}}[%{{.*}}] :
// CHECK-NOT: mask =
tt.func @gather_strip_broadcast_true_mask(%ptr: !tt.ptr<bf16>, %idx: tensor<64xi32>) -> tensor<64xbf16> {
  %true_small = arith.constant dense<true> : tensor<1xi1>
  %true_mask = tt.broadcast %true_small : tensor<1xi1> -> tensor<64xi1>
  %0 = tts.gather %ptr[%idx] mask = %true_mask : (!tt.ptr<bf16>, tensor<64xi32>) -> tensor<64xbf16>
  tt.return %0 : tensor<64xbf16>
}

// CHECK-LABEL: @gather_keep_partial_mask
// CHECK: tts.gather %{{.*}}[%{{.*}}] mask =
tt.func @gather_keep_partial_mask(%ptr: !tt.ptr<f32>, %idx: tensor<4xi32>) -> tensor<4xf32> {
  %mask = arith.constant dense<[true, true, false, true]> : tensor<4xi1>
  %0 = tts.gather %ptr[%idx] mask = %mask : (!tt.ptr<f32>, tensor<4xi32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @gather_keep_runtime_mask
// CHECK: tts.gather %{{.*}}[%{{.*}}] mask =
tt.func @gather_keep_runtime_mask(%ptr: !tt.ptr<f16>, %idx: tensor<64xi32>, %rt_mask: tensor<64xi1>) -> tensor<64xf16> {
  %0 = tts.gather %ptr[%idx] mask = %rt_mask : (!tt.ptr<f16>, tensor<64xi32>) -> tensor<64xf16>
  tt.return %0 : tensor<64xf16>
}

// CHECK-LABEL: @gather_no_mask
// CHECK: tts.gather
// CHECK-NOT: mask =
tt.func @gather_no_mask(%ptr: !tt.ptr<f16>, %idx: tensor<64xi32>) -> tensor<64xf16> {
  %0 = tts.gather %ptr[%idx] : (!tt.ptr<f16>, tensor<64xi32>) -> tensor<64xf16>
  tt.return %0 : tensor<64xf16>
}
