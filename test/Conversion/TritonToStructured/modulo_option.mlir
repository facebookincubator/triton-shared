// RUN: triton-shared-opt --triton-to-structured="enable-modulo-support=false" --canonicalize %s | FileCheck %s --check-prefix=CHECK-DISABLED
// RUN: triton-shared-opt --triton-to-structured="enable-modulo-support=true" --canonicalize %s | FileCheck %s --check-prefix=CHECK-ENABLED
module {
  tt.func public @modulo(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c16_i32 : i32
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.splat %1 : i32 -> tensor<16xi32>
    %4 = arith.addi %3, %2 : tensor<16xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<16xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<16xi32>
    %7 = tt.splat %arg3 : i32 -> tensor<16xi32>
    %8 = arith.remsi %4, %7 : tensor<16xi32>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %11 = tt.load %10, %6 : tensor<16x!tt.ptr<f32>>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %13 = tt.addptr %12, %4 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %13, %11, %6 : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}

// Check when modulo is disabled, we should not generate a tts.load
// CHECK-DISABLED-LABEL: @modulo
// CHECK-DISABLED-NOT: tts.load
// CHECK-DISABLED: tt.load
// CHECK-DISABLED-NOT: tt.store
// CHECK-DISABLED: tts.store


// Check when modulo is enabled, we should generate a tts.load
// CHECK-ENABLED-LABEL: @modulo(
// CHECK-ENABLED-SAME: %{{.+}}: !tt.ptr<f32>, %{{.+}}: !tt.ptr<f32>, %{{.+}}: i32, [[MODARG:%[a-zA-Z0-9_]+]]: i32, %{{.+}}: i32)
// CHECK-ENABLED-NOT: tt.load
// CHECK-ENABLED-NOT: tt.store
// CHECK-ENABLED: [[SHAPE:%.+]] = arith.index_cast [[MODARG]] : i32 to index
// CHECK-ENABLED: [[PTR:%.+]] = tts.make_tptr %{{.+}} to sizes: [16], strides: [1], offsets: [{{%.+}}], shape: {{\[}}[[SHAPE]]{{\]}}, order: []
// CHECK-ENABLED: "tts.load"([[PTR]], {{.+}})
// CHECK-ENABLED: tts.store
