// RUN: triton-shared-opt %s -split-input-file --verify-diagnostics

// Test that a tts.get_structured_state op on a real pointer value cannot return a pointer<none>

module {
  tt.func public @invalid_ptr_to_none(%arg0: tensor<16x!tt.ptr<f32>>) {
    // expected-error@below {{verification of operation 'tts.get_structured_state' failed}}
    %0, %1, %2, %3 = "tts.get_structured_state"(%arg0) <{resultSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<16x!tt.ptr<f32>>) -> (tensor<16x!tt.ptr<f32>>, index, index, !tt.ptr<none>)
    tt.return
  }
}

// -----

// Test that a tts.get_structured_state op on a tensor of indices cannot return a pointer of non-none type

module {
  tt.func public @invalid_indices_to_ptr(%arg0: tensor<16xi32>) {
    // expected-error@below {{verification of operation 'tts.get_structured_state' failed}}
    %0, %1, %2, %3 = "tts.get_structured_state"(%arg0) <{resultSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<16xi32>) -> (tensor<16xi32>, index, index, !tt.ptr<f32>)
    tt.return
  }
}
