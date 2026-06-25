// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s
//
// Verify that tt.bitcast on a ptr<i1> tensor is correctly handled during
// unstructured pointer analysis. The pass must propagate offset info through
// the bitcast and lower the load/store to gather/scatter using the
// scalar-bitcast base pointer.

module {
  tt.func public @bitcast_ptr(%mask_ptr: !tt.ptr<i1>, %output_ptr: !tt.ptr<i8>) {
    %offsets = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %splat   = tt.splat %mask_ptr : !tt.ptr<i1> -> tensor<1024x!tt.ptr<i1>>
    %addptr  = tt.addptr %splat, %offsets : tensor<1024x!tt.ptr<i1>>, tensor<1024xi32>
    %bitcast = tt.bitcast %addptr : tensor<1024x!tt.ptr<i1>> -> tensor<1024x!tt.ptr<i8>>
    %loaded  = tt.load %bitcast : tensor<1024x!tt.ptr<i8>>
    %out_splat  = tt.splat %output_ptr : !tt.ptr<i8> -> tensor<1024x!tt.ptr<i8>>
    %out_addptr = tt.addptr %out_splat, %offsets : tensor<1024x!tt.ptr<i8>>, tensor<1024xi32>
    tt.store %out_addptr, %loaded : tensor<1024x!tt.ptr<i8>>
    tt.return
  }
}

// Tensor bitcast consumed; scalar bitcast of base pointer inserted.
// CHECK: tt.bitcast %arg0 : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK: tts.gather
// CHECK: tts.scatter
