// RUN: triton-shared-opt --triton-to-structured=enable-make-gather-scatter=false --remove-dead-values --cse --canonicalize %s | FileCheck %s --enable-var-scope --implicit-check-not=tts.make_gather_scatter_tptr

// A non-wrapping dilated bound c * r < R (c > 0) is r < ceildiv(R, c),
// and c * r >= 0 is r >= 0, so these masks stay structured block-pointer loads.

module {
// (2 * r < 6) & (r < 3): rows [0, 3).
// CHECK-LABEL: tt.func public @dilated_row_mask(
// CHECK:         "tts.load"({{.*}}) <{{{.*}}static_mask_dims = array<i64: 3, 8>}>
  tt.func public @dilated_row_mask(%in: !tt.ptr<f32>) -> tensor<4x8xf32> {
    %other = arith.constant dense<0xFF800000> : tensor<4x8xf32>
    %c2 = arith.constant dense<2> : tensor<4xi32>
    %c3 = arith.constant dense<3> : tensor<4xi32>
    %c6 = arith.constant dense<6> : tensor<4xi32>
    %c8 = arith.constant dense<8> : tensor<4x1xi32>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %dilated = arith.muli %r, %c2 : tensor<4xi32>
    %dilation_mask = arith.cmpi slt, %dilated, %c6 : tensor<4xi32>
    %row_bound = arith.cmpi slt, %r, %c3 : tensor<4xi32>
    %row_mask = arith.andi %dilation_mask, %row_bound : tensor<4xi1>
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

// (extsi(2 * r) >= 0) & (extsi(2 * r) < 5) with i64 bounds: rows [0, ceildiv(5, 2)) = [0, 3).
// CHECK-LABEL: tt.func public @dilated_row_mask_ceildiv_extsi(
// CHECK:         "tts.load"({{.*}}) <{{{.*}}static_mask_dims = array<i64: 3, 8>}>
  tt.func public @dilated_row_mask_ceildiv_extsi(%in: !tt.ptr<f32>) -> tensor<4x8xf32> {
    %other = arith.constant dense<0xFF800000> : tensor<4x8xf32>
    %c2 = arith.constant dense<2> : tensor<4xi32>
    %c0_i64 = arith.constant dense<0> : tensor<4xi64>
    %c5_i64 = arith.constant dense<5> : tensor<4xi64>
    %c8 = arith.constant dense<8> : tensor<4x1xi32>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %c = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %dilated = arith.muli %r, %c2 : tensor<4xi32>
    %dilated_i64 = arith.extsi %dilated : tensor<4xi32> to tensor<4xi64>
    %lower = arith.cmpi sge, %dilated_i64, %c0_i64 : tensor<4xi64>
    %upper = arith.cmpi slt, %dilated_i64, %c5_i64 : tensor<4xi64>
    %row_mask = arith.andi %lower, %upper : tensor<4xi1>
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

  // CHECK-LABEL: tt.func public @unsupported_scaled_predicates(
  // CHECK-DAG: [[OTHER:%.+]] = arith.constant dense<-9.000000e+00> : tensor<4xf32>
  // CHECK-DAG: [[BOUND:%.+]] = arith.constant dense<3> : tensor<4xi32>
  // CHECK-DAG: [[LOWER_BOUND:%.+]] = arith.constant dense<1> : tensor<4xi32>
  // CHECK: [[SCALED:%.+]] = arith.muli %{{.*}}, %{{.*}} : tensor<4xi32>
  // CHECK-DAG: [[INCLUSIVE:%.+]] = arith.cmpi sle, [[SCALED]], [[BOUND]] : tensor<4xi32>
  // CHECK-DAG: [[GREATER:%.+]] = arith.cmpi sgt, [[SCALED]], [[BOUND]] : tensor<4xi32>
  // CHECK-DAG: [[NONZERO_LOWER:%.+]] = arith.cmpi sge, [[SCALED]], [[LOWER_BOUND]] : tensor<4xi32>
  // CHECK: [[INCLUSIVE_VALUE:%.+]] = tt.load %{{.*}}, [[INCLUSIVE]], [[OTHER]] : tensor<4x!tt.ptr<f32>>
  // CHECK: [[GREATER_VALUE:%.+]] = tt.load %{{.*}}, [[GREATER]], [[OTHER]] : tensor<4x!tt.ptr<f32>>
  // CHECK: [[NONZERO_LOWER_VALUE:%.+]] = tt.load %{{.*}}, [[NONZERO_LOWER]], [[OTHER]] : tensor<4x!tt.ptr<f32>>
  // CHECK: tt.return [[INCLUSIVE_VALUE]], [[GREATER_VALUE]], [[NONZERO_LOWER_VALUE]] : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  tt.func public @unsupported_scaled_predicates(%in: !tt.ptr<f32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
    %other = arith.constant dense<-9.0> : tensor<4xf32>
    %scale = arith.constant dense<2> : tensor<4xi32>
    %bound = arith.constant dense<3> : tensor<4xi32>
    %lower_bound = arith.constant dense<1> : tensor<4xi32>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %scaled = arith.muli %r, %scale : tensor<4xi32>
    %inclusive = arith.cmpi sle, %scaled, %bound : tensor<4xi32>
    %greater = arith.cmpi sgt, %scaled, %bound : tensor<4xi32>
    %nonzero_lower = arith.cmpi sge, %scaled, %lower_bound : tensor<4xi32>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %inclusive_value = tt.load %ptr, %inclusive, %other : tensor<4x!tt.ptr<f32>>
    %greater_value = tt.load %ptr, %greater, %other : tensor<4x!tt.ptr<f32>>
    %nonzero_lower_value = tt.load %ptr, %nonzero_lower, %other : tensor<4x!tt.ptr<f32>>
    tt.return %inclusive_value, %greater_value, %nonzero_lower_value : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
  }

  // i32 multiplication wraps: 2^30 * [0, 1, 2, 3] includes negative values.
  // Keep the full mask, including when the wrapped product is sign-extended.
  // CHECK-LABEL: tt.func public @overflow_lower_mask(
  // CHECK-DAG: [[OTHER:%.+]] = arith.constant dense<-9.000000e+00> : tensor<4xf32>
  // CHECK-DAG: [[SCALE:%.+]] = arith.constant dense<1073741824> : tensor<4xi32>
  // CHECK-DAG: [[BOUND:%.+]] = arith.constant dense<0> : tensor<4xi32>
  // CHECK: [[RANGE:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  // CHECK: [[SCALED:%.+]] = arith.muli [[RANGE]], [[SCALE]] : tensor<4xi32>
  // CHECK: [[MASK:%.+]] = arith.cmpi sge, [[SCALED]], [[BOUND]] : tensor<4xi32>
  // CHECK: [[LOADED:%.+]] = tt.load %{{.*}}, [[MASK]], [[OTHER]] : tensor<4x!tt.ptr<f32>>
  // CHECK: tt.return [[LOADED]] : tensor<4xf32>
  tt.func public @overflow_lower_mask(%in: !tt.ptr<f32>) -> tensor<4xf32> {
    %other = arith.constant dense<-9.0> : tensor<4xf32>
    %scale = arith.constant dense<1073741824> : tensor<4xi32>
    %zero = arith.constant dense<0> : tensor<4xi32>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %scaled = arith.muli %r, %scale : tensor<4xi32>
    %mask = arith.cmpi sge, %scaled, %zero : tensor<4xi32>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %value = tt.load %ptr, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.return %value : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func public @overflow_lower_mask_extsi(
  // CHECK-DAG: [[OTHER:%.+]] = arith.constant dense<-9.000000e+00> : tensor<4xf32>
  // CHECK-DAG: [[SCALE:%.+]] = arith.constant dense<1073741824> : tensor<4xi32>
  // CHECK-DAG: [[BOUND:%.+]] = arith.constant dense<0> : tensor<4xi64>
  // CHECK: [[RANGE:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  // CHECK: [[SCALED:%.+]] = arith.muli [[RANGE]], [[SCALE]] : tensor<4xi32>
  // CHECK: [[EXTENDED:%.+]] = arith.extsi [[SCALED]] : tensor<4xi32> to tensor<4xi64>
  // CHECK: [[MASK:%.+]] = arith.cmpi sge, [[EXTENDED]], [[BOUND]] : tensor<4xi64>
  // CHECK: [[LOADED:%.+]] = tt.load %{{.*}}, [[MASK]], [[OTHER]] : tensor<4x!tt.ptr<f32>>
  // CHECK: tt.return [[LOADED]] : tensor<4xf32>
  tt.func public @overflow_lower_mask_extsi(%in: !tt.ptr<f32>) -> tensor<4xf32> {
    %other = arith.constant dense<-9.0> : tensor<4xf32>
    %scale = arith.constant dense<1073741824> : tensor<4xi32>
    %zero = arith.constant dense<0> : tensor<4xi64>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %scaled = arith.muli %r, %scale : tensor<4xi32>
    %extended = arith.extsi %scaled : tensor<4xi32> to tensor<4xi64>
    %mask = arith.cmpi sge, %extended, %zero : tensor<4xi64>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %value = tt.load %ptr, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.return %value : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func public @overflow_upper_mask(
  // CHECK-DAG: [[OTHER:%.+]] = arith.constant dense<-9.000000e+00> : tensor<4xf32>
  // CHECK-DAG: [[SCALE:%.+]] = arith.constant dense<1073741824> : tensor<4xi32>
  // CHECK-DAG: [[BOUND:%.+]] = arith.constant dense<1> : tensor<4xi32>
  // CHECK: [[RANGE:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  // CHECK: [[SCALED:%.+]] = arith.muli [[RANGE]], [[SCALE]] : tensor<4xi32>
  // CHECK: [[MASK:%.+]] = arith.cmpi slt, [[SCALED]], [[BOUND]] : tensor<4xi32>
  // CHECK: [[LOADED:%.+]] = tt.load %{{.*}}, [[MASK]], [[OTHER]] : tensor<4x!tt.ptr<f32>>
  // CHECK: tt.return [[LOADED]] : tensor<4xf32>
  tt.func public @overflow_upper_mask(%in: !tt.ptr<f32>) -> tensor<4xf32> {
    %other = arith.constant dense<-9.0> : tensor<4xf32>
    %scale = arith.constant dense<1073741824> : tensor<4xi32>
    %one = arith.constant dense<1> : tensor<4xi32>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %scaled = arith.muli %r, %scale : tensor<4xi32>
    %mask = arith.cmpi slt, %scaled, %one : tensor<4xi32>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %value = tt.load %ptr, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.return %value : tensor<4xf32>
  }

  // CHECK-LABEL: tt.func public @overflow_upper_mask_extsi(
  // CHECK-DAG: [[OTHER:%.+]] = arith.constant dense<-9.000000e+00> : tensor<4xf32>
  // CHECK-DAG: [[SCALE:%.+]] = arith.constant dense<1073741824> : tensor<4xi32>
  // CHECK-DAG: [[BOUND:%.+]] = arith.constant dense<1> : tensor<4xi64>
  // CHECK: [[RANGE:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  // CHECK: [[SCALED:%.+]] = arith.muli [[RANGE]], [[SCALE]] : tensor<4xi32>
  // CHECK: [[EXTENDED:%.+]] = arith.extsi [[SCALED]] : tensor<4xi32> to tensor<4xi64>
  // CHECK: [[MASK:%.+]] = arith.cmpi slt, [[EXTENDED]], [[BOUND]] : tensor<4xi64>
  // CHECK: [[LOADED:%.+]] = tt.load %{{.*}}, [[MASK]], [[OTHER]] : tensor<4x!tt.ptr<f32>>
  // CHECK: tt.return [[LOADED]] : tensor<4xf32>
  tt.func public @overflow_upper_mask_extsi(%in: !tt.ptr<f32>) -> tensor<4xf32> {
    %other = arith.constant dense<-9.0> : tensor<4xf32>
    %scale = arith.constant dense<1073741824> : tensor<4xi32>
    %one = arith.constant dense<1> : tensor<4xi64>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %scaled = arith.muli %r, %scale : tensor<4xi32>
    %extended = arith.extsi %scaled : tensor<4xi32> to tensor<4xi64>
    %mask = arith.cmpi slt, %extended, %one : tensor<4xi64>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %value = tt.load %ptr, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.return %value : tensor<4xf32>
  }

  // Widening before multiplication makes the same products fit in i64.
  // CHECK-LABEL: tt.func public @safe_widen_before_scaled_mask(
  // CHECK: [[PTR:%.+]] = tts.make_tptr %{{.*}} to sizes: [4], strides: [1], offsets: [0]
  // CHECK: [[LOADED:%.+]] = "tts.load"([[PTR]], %{{.*}}) <{{{.*}}static_mask_dims = array<i64: 3>}>
  // CHECK: tt.return [[LOADED]] : tensor<4xf32>
  tt.func public @safe_widen_before_scaled_mask(%in: !tt.ptr<f32>) -> tensor<4xf32> {
    %other = arith.constant dense<-9.0> : tensor<4xf32>
    %scale = arith.constant dense<1073741824> : tensor<4xi64>
    %zero = arith.constant dense<0> : tensor<4xi64>
    %bound = arith.constant dense<2147483649> : tensor<4xi64>
    %r = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %wide = arith.extsi %r : tensor<4xi32> to tensor<4xi64>
    %scaled = arith.muli %wide, %scale : tensor<4xi64>
    %lower = arith.cmpi sge, %scaled, %zero : tensor<4xi64>
    %upper = arith.cmpi slt, %scaled, %bound : tensor<4xi64>
    %mask = arith.andi %lower, %upper : tensor<4xi1>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %r : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %value = tt.load %ptr, %mask, %other : tensor<4x!tt.ptr<f32>>
    tt.return %value : tensor<4xf32>
  }
  // CHECK-LABEL: tt.func public @scaled_singleton_upper_mask(
  // CHECK: [[OTHER:%.+]] = arith.constant dense<-9.000000e+00> : tensor<1xf32>
  // CHECK: tt.return [[OTHER]] : tensor<1xf32>
  tt.func public @scaled_singleton_upper_mask(%in: !tt.ptr<f32>) -> tensor<1xf32> {
    %other = arith.constant dense<-9.0> : tensor<1xf32>
    %scale = arith.constant dense<2> : tensor<1xi32>
    %zero = arith.constant dense<0> : tensor<1xi32>
    %r = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
    %scaled = arith.muli %r, %scale : tensor<1xi32>
    %mask = arith.cmpi slt, %scaled, %zero : tensor<1xi32>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %r : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
    %value = tt.load %ptr, %mask, %other : tensor<1x!tt.ptr<f32>>
    tt.return %value : tensor<1xf32>
  }

  // CHECK-LABEL: tt.func public @scaled_singleton_lower_mask(
  // CHECK: [[PTR:%.+]] = tt.splat %{{.*}} : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
  // CHECK: [[LOADED:%.+]] = tt.load [[PTR]] : tensor<1x!tt.ptr<f32>>
  // CHECK: tt.return [[LOADED]] : tensor<1xf32>
  tt.func public @scaled_singleton_lower_mask(%in: !tt.ptr<f32>) -> tensor<1xf32> {
    %other = arith.constant dense<-9.0> : tensor<1xf32>
    %scale = arith.constant dense<2> : tensor<1xi32>
    %zero = arith.constant dense<0> : tensor<1xi32>
    %r = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
    %scaled = arith.muli %r, %scale : tensor<1xi32>
    %mask = arith.cmpi sge, %scaled, %zero : tensor<1xi32>
    %base = tt.splat %in : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
    %ptr = tt.addptr %base, %r : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
    %value = tt.load %ptr, %mask, %other : tensor<1x!tt.ptr<f32>>
    tt.return %value : tensor<1xf32>
  }
}
