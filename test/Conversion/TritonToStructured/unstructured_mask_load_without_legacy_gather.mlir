// RUN: triton-shared-opt --triton-to-structured=enable-make-gather-scatter=false --remove-dead-values --cse --canonicalize %s | FileCheck %s --implicit-check-not=tts.make_gather_scatter_tptr

// With enable-make-gather-scatter=false, the holed row mask (r != 1) & (r < 3)
// remains on tt.load because it cannot be represented by a single extent.
module {
// CHECK-LABEL: tt.func public @holed_row_mask_load(
// CHECK:         [[MASK:%.+]] = tt.broadcast %{{.*}} : tensor<4x1xi1> -> tensor<4x8xi1>
// CHECK:         tt.load %{{.*}}, [[MASK]], %{{.*}} : tensor<4x8x!tt.ptr<f32>>
  tt.func public @holed_row_mask_load(%in: !tt.ptr<f32>) -> tensor<4x8xf32> {
    %other = arith.constant dense<0xFF800000> : tensor<4x8xf32>
    %c1 = arith.constant dense<1> : tensor<4xi32>
    %c3 = arith.constant dense<3> : tensor<4xi32>
    %c8 = arith.constant dense<8> : tensor<4x1xi32>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %hole_mask = arith.cmpi ne, %r, %c1 : tensor<4xi32>
    %row_bound = arith.cmpi slt, %r, %c3 : tensor<4xi32>
    %row_mask = arith.andi %hole_mask, %row_bound : tensor<4xi1>
    %row_mask2d = tt.expand_dims %row_mask {axis = 1 : i32} : tensor<4xi1> -> tensor<4x1xi1>
    %mask = tt.broadcast %row_mask2d : tensor<4x1xi1> -> tensor<4x8xi1>
    %r2d = tt.expand_dims %r {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %row_off = arith.muli %r2d, %c8 : tensor<4x1xi32>
    %row_offb = tt.broadcast %row_off : tensor<4x1xi32> -> tensor<4x8xi32>
    %c2d = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %col_offb = tt.broadcast %c2d : tensor<1x8xi32> -> tensor<4x8xi32>
    %off = arith.addi %row_offb, %col_offb : tensor<4x8xi32>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %off : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
    %value = tt.load %ptr, %mask, %other : tensor<4x8x!tt.ptr<f32>>
    tt.return %value : tensor<4x8xf32>
  }
}
