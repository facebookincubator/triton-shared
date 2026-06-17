// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @expand_dims_ptr_load(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<2x!tt.ptr<f32>> -> tensor<1x2x!tt.ptr<f32>>
    %4 = tt.load %3 : tensor<1x2x!tt.ptr<f32>>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %6 = tt.addptr %5, %0 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<2x!tt.ptr<f32>> -> tensor<1x2x!tt.ptr<f32>>
    tt.store %7, %4 : tensor<1x2x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @expand_dims_ptr_load
// CHECK:       %[[RANGE:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK:       %[[EXPANDED:.*]] = tt.expand_dims %[[RANGE]] {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
// CHECK:       %[[GATHER:.*]] = tts.gather {{.*}}{{\[}}%[[EXPANDED]]] : (<f32>, tensor<1x2xi32>) -> tensor<1x2xf32>
// CHECK:       tts.scatter %[[GATHER]] into {{.*}}{{\[}}%[[EXPANDED]]] : tensor<1x2xf32> into (<f32>, tensor<1x2xi32>)
