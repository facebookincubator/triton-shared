// RUN: triton-shared-opt --triton-to-unstructured --split-input-file %s | FileCheck %s

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

// Test 4: bitcast inside scf.for with loop-carried pointer iter-arg.
// The scf.for handler retypes the iter-arg to an integer offset type before
// the bitcast handler processes it. The bitcast handler must use the saved
// ptrType from offsetMap (not src.getType()) to avoid asserting on the
// already-retyped integer type.

module {
  tt.func public @loop_carried_bitcast(%arg0: !tt.ptr<i1>, %arg1: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %step = arith.constant dense<8> : tensor<128xi32>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
    %res = scf.for %i = %c0 to %c4 step %c1 iter_args(%p = %2)
        -> (tensor<128x!tt.ptr<i1>>) : i32 {
      %bc = tt.bitcast %p : tensor<128x!tt.ptr<i1>> -> tensor<128x!tt.ptr<i8>>
      %ld = tt.load %bc : tensor<128x!tt.ptr<i8>>
      %sp = tt.splat %arg1 : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>>
      %so = tt.addptr %sp, %0 : tensor<128x!tt.ptr<i8>>, tensor<128xi32>
      tt.store %so, %ld : tensor<128x!tt.ptr<i8>>
      %next = tt.addptr %p, %step : tensor<128x!tt.ptr<i1>>, tensor<128xi32>
      scf.yield %next : tensor<128x!tt.ptr<i1>>
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func public @loop_carried_bitcast
// CHECK: scf.for
// CHECK:   [[BC:%.+]] = tt.bitcast %arg0 : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK:   tts.gather [[BC]][{{.+}}] : (<i8>, tensor<128xi32>) -> tensor<128xi8>
// CHECK:   tts.scatter
