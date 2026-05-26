// RUN: %triton-opt --triton-to-unstructured -split-input-file %s | %FileCheck %s

// This test reproduces a bug where scf.for loops with scalar pointer iter_args
// would fail verification with: "'scf.for' op 0-th init and 0-th region
// iter_arg have different type: '!tt.ptr<i64>' != 'i32'".

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
