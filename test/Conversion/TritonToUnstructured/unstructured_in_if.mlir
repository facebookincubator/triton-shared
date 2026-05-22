// RUN: triton-shared-opt --triton-to-unstructured --canonicalize %s | FileCheck %s

module {
  tt.func @gather_inside_if(
    %ptr_in: !tt.ptr<f32>,
    %ptr_out: !tt.ptr<f32>,
    %idx: tensor<512xi64>,
    %mask: tensor<512xi1>,
    %do_load: i1
  ) {
    scf.if %do_load {
        %ptr_in_splat = tt.splat %ptr_in : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
        %in_ptrs = tt.addptr %ptr_in_splat, %idx : tensor<512x!tt.ptr<f32>>, tensor<512xi64>
        %data = tt.load %in_ptrs, %mask : tensor<512x!tt.ptr<f32>>
        %ptr_out_splat = tt.splat %ptr_out : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
        %out_ptrs = tt.addptr %ptr_out_splat, %idx : tensor<512x!tt.ptr<f32>>, tensor<512xi64>
        tt.store %out_ptrs, %data, %mask : tensor<512x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK: tts.gather
// CHECK: tts.scatter
