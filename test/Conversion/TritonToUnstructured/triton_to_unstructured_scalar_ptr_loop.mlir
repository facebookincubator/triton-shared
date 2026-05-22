// RUN: triton-shared-opt --triton-to-unstructured --split-input-file %s | FileCheck %s

// This reproduces a bug where scf.for loops with scalar pointer iter_args
// failed verification because the pass changed the iter_arg and result types to
// offsets while leaving the init_arg as a pointer.

module {
  tt.func public @scalar_ptr_loop(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32

    %0:2 = scf.for %i = %c0_i32 to %c2_i32 step %c1_i32
        iter_args(%ptr_in = %arg0, %ptr_out = %arg1) -> (!tt.ptr<i64>, !tt.ptr<i64>) : i32 {
      %val = tt.load %ptr_in : !tt.ptr<i64>
      tt.store %ptr_out, %val : !tt.ptr<i64>

      %c8 = arith.constant 8 : i32
      %next_ptr_in = tt.addptr %ptr_in, %c8 : !tt.ptr<i64>, i32
      %next_ptr_out = tt.addptr %ptr_out, %c8 : !tt.ptr<i64>, i32

      scf.yield %next_ptr_in, %next_ptr_out : !tt.ptr<i64>, !tt.ptr<i64>
    }

    tt.return
  }
}

// CHECK-LABEL: @scalar_ptr_loop
// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store
// CHECK: tts.gather
// CHECK: tts.scatter

// -----

// This reproduces a bug where an scf.yield in an scf.for nested under scf.if
// incorrectly triggered the scf.if pointer-result bailout.

module {
  tt.func public @scalar_ptr_in_for_nested_in_if(
      %base: !tt.ptr<f32>, %out: !tt.ptr<f32>, %cond: i1, %n: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32

    scf.if %cond {
      %0:2 = scf.for %i = %c0_i32 to %n step %c1_i32
          iter_args(%scalar_ptr = %base, %out_ptr = %out)
          -> (!tt.ptr<f32>, !tt.ptr<f32>) : i32 {
        %sv = tt.load %scalar_ptr : !tt.ptr<f32>
        tt.store %out_ptr, %sv : !tt.ptr<f32>

        %next_ptr = tt.addptr %scalar_ptr, %c32_i32 : !tt.ptr<f32>, i32
        %next_out = tt.addptr %out_ptr, %c1_i32 : !tt.ptr<f32>, i32

        scf.yield %next_ptr, %next_out : !tt.ptr<f32>, !tt.ptr<f32>
      }
    }

    tt.return
  }
}

// CHECK-LABEL: @scalar_ptr_in_for_nested_in_if
// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store
// CHECK: tts.gather
// CHECK: tts.scatter
