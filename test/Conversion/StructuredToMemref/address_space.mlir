// RUN: %triton-opt --triton-to-linalg-experimental %s | %FileCheck %s

// Verify that tt.ptr address spaces are preserved through the full
// triton-to-linalg-experimental pipeline, including StructuredToMemref and
// PtrAnalysis.

module {
  tt.func public @add_kernel(%x_ptr: !tt.ptr<f16>, %y_ptr: !tt.ptr<f16>, %output_ptr: !tt.ptr<f16>, %n_elements: i32) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %pid = tt.get_program_id x : i32
    %block_start = arith.muli %pid, %c256_i32 : i32
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = tt.splat %block_start : i32 -> tensor<256xi32>
    %offsets = arith.addi %1, %0 : tensor<256xi32>
    %2 = tt.splat %n_elements : i32 -> tensor<256xi32>
    %mask = arith.cmpi slt, %offsets, %2 : tensor<256xi32>
    %3 = tt.splat %x_ptr : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %4 = tt.addptr %3, %offsets : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %x = tt.load %4, %mask : tensor<256x!tt.ptr<f16>>
    %5 = tt.splat %y_ptr : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %6 = tt.addptr %5, %offsets : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    %y = tt.load %6, %mask : tensor<256x!tt.ptr<f16>>
    %output = arith.addf %x, %y : tensor<256xf16>
    %7 = tt.splat %output_ptr : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>>
    %8 = tt.addptr %7, %offsets : tensor<256x!tt.ptr<f16>>, tensor<256xi32>
    tt.store %8, %output, %mask : tensor<256x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL: func.func @add_kernel
// CHECK-SAME: memref<*xf16, 1>
// CHECK-SAME: memref<*xf16, 1>
// CHECK-SAME: memref<*xf16, 1>
// CHECK-SAME: i32
// CHECK: memref.reinterpret_cast {{.*}} : memref<*xf16, 1> to memref<256xf16, strided<[1], offset: ?>, 1>
// CHECK: memref.reinterpret_cast {{.*}} : memref<*xf16, 1> to memref<256xf16, strided<[1], offset: ?>, 1>
// CHECK: memref.reinterpret_cast {{.*}} : memref<*xf16, 1> to memref<256xf16, strided<[1], offset: ?>, 1>
