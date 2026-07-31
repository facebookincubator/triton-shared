// RUN: triton-shared-opt --triton-to-unstructured --split-input-file --verify-diagnostics %s

// Test: bitcast between pointer types with different pointee byte sizes is rejected.
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
    // expected-error @+1 {{bitcast between pointer types with different strides}}
    %5 = tt.bitcast %4 : tensor<128x!tt.ptr<i8>> -> tensor<128x!tt.ptr<i16>>
    %6 = tt.addptr %5, %cst_1 : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
    %7 = tt.load %6 : tensor<128x!tt.ptr<i16>>
    %8 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<128x!tt.ptr<i16>>
    %9 = tt.addptr %8, %0 : tensor<128x!tt.ptr<i16>>, tensor<128xi32>
    tt.store %9, %7 : tensor<128x!tt.ptr<i16>>
    tt.return
  }
}
