// RUN: triton-shared-opt --triton-to-unstructured --split-input-file %s 2>&1 | FileCheck %s

// The error from Test 4 (i8->i16 pointee byte size mismatch) appears first
// in the combined output because stderr is unbuffered.
// CHECK: error: bitcast between pointer types with different strides

// Test 1: addptr -> bitcast(i1->i8) -> load
// sizeof(i1)==sizeof(i8)==1 byte, so offset N equals N bytes for both types.

module {
  tt.func public @bitcast_ptr_to_ptr(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<i8>) {
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<1024x!tt.ptr<i1>>
    %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<i1>>, tensor<1024xi32>
    %3 = tt.bitcast %2 : tensor<1024x!tt.ptr<i1>> -> tensor<1024x!tt.ptr<i8>>
    %4 = tt.load %3 : tensor<1024x!tt.ptr<i8>>
    %5 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<1024x!tt.ptr<i8>>
    %6 = tt.addptr %5, %0 : tensor<1024x!tt.ptr<i8>>, tensor<1024xi32>
    tt.store %6, %4 : tensor<1024x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @bitcast_ptr_to_ptr
// CHECK: [[base:%.+]] = tt.bitcast %arg0 : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK: tts.gather [[base]][{{.+}}] : (<i8>, tensor<1024xi32>) -> tensor<1024xi8>
// CHECK: tts.scatter

// -----

// Test 2: addptr -> bitcast(i1->i8) -> addptr -> load/store
// Both offsets accumulate via arith.addi since sizeof(i1)==sizeof(i8)==1.
// Proves addptr after bitcast produces correct combined offset.

module {
  tt.func public @addptr_bitcast_i1_i8_addptr(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<i8>) {
    %cst = arith.constant dense<32> : tensor<128xi32>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
    %3 = tt.bitcast %2 : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
    %4 = tt.addptr %3, %cst : tensor<128x!tt.ptr<i8>>, tensor<128xi32>
    %5 = tt.load %4 : tensor<128x!tt.ptr<i8>>
    %6 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>>
    %7 = tt.addptr %6, %0 : tensor<128x!tt.ptr<i8>>, tensor<128xi32>
    tt.store %7, %5 : tensor<128x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @addptr_bitcast_i1_i8_addptr
// CHECK: [[BASE2:%.+]] = tt.bitcast %arg0 : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK: [[OFF:%.+]] = arith.addi {{.+}}, {{.+}} : tensor<128xi32>
// CHECK: tts.gather [[BASE2]]{{\[}}[[OFF]]{{\]}} : (<i8>, tensor<128xi32>) -> tensor<128xi8>
// CHECK: tts.scatter

// -----

// Test 3: addptr -> bitcast(i1->i8) -> addptr -> bitcast(i8->i1) -> addptr -> bitcast(i1->i8) -> load
// Multiple bitcast chain (i1->i8->i1->i8). All types are 1-byte stride,
// so offsets accumulate correctly through all bitcasts.
// Final offset = range + 32 + 16, base ends as ptr<i8>.

module {
  tt.func public @bitcast_i1_i8_i1_chain(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<i8>) {
    %cst = arith.constant dense<32> : tensor<128xi32>
    %cst_0 = arith.constant dense<16> : tensor<128xi32>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
    %3 = tt.bitcast %2 : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
    %4 = tt.addptr %3, %cst : tensor<128x!tt.ptr<i8>>, tensor<128xi32>
    %5 = tt.bitcast %4 : tensor<128x!tt.ptr<i8>> -> tensor<128x!tt.ptr<i1>>
    %6 = tt.addptr %5, %cst_0 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
    %7 = tt.bitcast %6 : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
    %8 = tt.load %7 : tensor<128x!tt.ptr<i8>>
    %9 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>>
    %10 = tt.addptr %9, %0 : tensor<128x!tt.ptr<i8>>, tensor<128xi32>
    tt.store %10, %8 : tensor<128x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @bitcast_i1_i8_i1_chain
// CHECK: tt.bitcast %arg0 : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK: [[OFF1:%.+]] = arith.addi {{.+}}, {{.+}} : tensor<128xi32>
// CHECK: tt.bitcast {{.+}} : !tt.ptr<i8> -> !tt.ptr<i1>
// CHECK: [[OFF2:%.+]] = arith.addi [[OFF1]], {{.+}} : tensor<128xi32>
// CHECK: [[BC3:%.+]] = tt.bitcast {{.+}} : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK: tts.gather [[BC3]]{{\[}}[[OFF2]]{{\]}} : (<i8>, tensor<128xi32>) -> tensor<128xi8>
// CHECK: tts.scatter

// -----

// Test 4: addptr -> bitcast(i1->i8) -> addptr -> bitcast(i8->i16) -> addptr -> load
// The first bitcast (i1->i8) is valid (both 1 byte), but the second bitcast
// (i8->i16) has different pointee byte sizes (1 byte vs 2 bytes). The pass
// rejects this because the accumulated element-offset would be misinterpreted:
// offset N means N*1 bytes for ptr<i8> but N*2 bytes for ptr<i16>.

module {
  tt.func public @bitcast_i8_to_i16_rejected(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<i16>) {
    %cst = arith.constant dense<32> : tensor<128xi32>
    %cst_1 = arith.constant dense<4> : tensor<128xi32>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
    %3 = tt.bitcast %2 : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
    %4 = tt.addptr %3, %cst : tensor<128x!tt.ptr<i8>>, tensor<128xi32>
    %5 = tt.bitcast %4 : tensor<128x!tt.ptr<i8>> -> tensor<128x!tt.ptr<i16>>
    %6 = tt.addptr %5, %cst_1 : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
    %7 = tt.load %6 : tensor<128x!tt.ptr<i16>>
    %8 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
    %9 = tt.addptr %8, %0 : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
    tt.store %9, %7 : tensor<128x!tt.ptr<i16>>
    tt.return
  }
}
