// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize --cse %s | FileCheck %s

module {
  tt.func public @scalar_mask_loop(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4.480000e+02> : tensor<1xf32>
    %cst_0 = arith.constant dense<-4.480000e+02> : tensor<1xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = arith.addi %arg3, %1 : i32
    %3 = arith.subi %2, %c1_i32 : i32
    %4 = arith.divsi %3, %1 : i32
    %5 = tt.load %arg2 : !tt.ptr<f32>
    %6 = tt.splat %0 : i32 -> tensor<1xi32>
    %7 = scf.for %arg4 = %c0_i32 to %4 step %c1_i32 iter_args(%arg5 = %6) -> (tensor<1xi32>)  : i32 {
      %8 = tt.splat %arg3 : i32 -> tensor<1xi32>
      %9 = arith.cmpi slt, %arg5, %8 : tensor<1xi32>
      %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1x!tt.ptr<bf16>>
      %11 = tt.addptr %10, %arg5 : tensor<1x!tt.ptr<bf16>>, tensor<1xi32>
      %12 = tt.load %11, %9 : tensor<1x!tt.ptr<bf16>>
      %13 = arith.extf %12 : tensor<1xbf16> to tensor<1xf32>
      %14 = tt.splat %5 : f32 -> tensor<1xf32>
      %15 = arith.mulf %13, %14 : tensor<1xf32>
      %16 = tt.clampf %15, %cst_0, %cst, propagateNan = none : tensor<1xf32>
      %17 = tt.fp_to_fp %16, rounding = rtne : tensor<1xf32> -> tensor<1xf8E4M3FN>
      %18 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<1x!tt.ptr<f8E4M3FN>>
      %19 = tt.addptr %18, %arg5 : tensor<1x!tt.ptr<f8E4M3FN>>, tensor<1xi32>
      tt.store %19, %17, %9 : tensor<1x!tt.ptr<f8E4M3FN>>
      %20 = tt.splat %1 : i32 -> tensor<1xi32>
      %21 = arith.addi %arg5, %20 : tensor<1xi32>
      scf.yield %21 : tensor<1xi32>
    }
    tt.return
  }

  tt.func public @constant_zero_singleton_mask_loop(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %bound: i32, %iterations: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %initial = arith.constant dense<0> : tensor<1xi32>
    %increment = arith.constant dense<1> : tensor<1xi32>
    %result = scf.for %iv = %zero to %iterations step %one iter_args(%index = %initial) -> (tensor<1xi32>) : i32 {
      %limit = tt.splat %bound : i32 -> tensor<1xi32>
      %mask = arith.cmpi slt, %index, %limit : tensor<1xi32>
      %in_base = tt.splat %in : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
      %in_ptr = tt.addptr %in_base, %index : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
      %value = tt.load %in_ptr, %mask : tensor<1x!tt.ptr<f32>>
      %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
      %out_ptr = tt.addptr %out_base, %index : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
      tt.store %out_ptr, %value, %mask : tensor<1x!tt.ptr<f32>>
      %next = arith.addi %index, %increment : tensor<1xi32>
      scf.yield %next : tensor<1xi32>
    }
    tt.return
  }

  tt.func public @constant_two_singleton_mask_loop(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %bound: i32, %iterations: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %initial = arith.constant dense<2> : tensor<1xi32>
    %increment = arith.constant dense<1> : tensor<1xi32>
    %result = scf.for %iv = %zero to %iterations step %one iter_args(%index = %initial) -> (tensor<1xi32>) : i32 {
      %limit = tt.splat %bound : i32 -> tensor<1xi32>
      %mask = arith.cmpi slt, %index, %limit : tensor<1xi32>
      %in_base = tt.splat %in : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
      %in_ptr = tt.addptr %in_base, %index : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
      %value = tt.load %in_ptr, %mask : tensor<1x!tt.ptr<f32>>
      %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
      %out_ptr = tt.addptr %out_base, %index : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
      tt.store %out_ptr, %value, %mask : tensor<1x!tt.ptr<f32>>
      %next = arith.addi %index, %increment : tensor<1xi32>
      scf.yield %next : tensor<1xi32>
    }
    tt.return
  }

  tt.func public @expanded_scalar_singleton_mask_fallback(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %bound: i32, %iterations: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %increment = arith.constant dense<1> : tensor<1x1xi32>
    %other = arith.constant dense<-9.0> : tensor<1x1xf32>
    %pid = tt.get_program_id x : i32
    %scalar = tt.splat %pid : i32 -> tensor<1xi32>
    %initial = tt.expand_dims %scalar {axis = 0 : i32} : tensor<1xi32> -> tensor<1x1xi32>
    %result = scf.for %iv = %zero to %iterations step %one iter_args(%index = %initial) -> (tensor<1x1xi32>) : i32 {
      %limit = tt.splat %bound : i32 -> tensor<1x1xi32>
      %mask = arith.cmpi slt, %index, %limit : tensor<1x1xi32>
      %in_base = tt.splat %in : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
      %in_ptr = tt.addptr %in_base, %index : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
      %value = tt.load %in_ptr, %mask, %other : tensor<1x1x!tt.ptr<f32>>
      %out_base = tt.splat %out : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>>
      %out_ptr = tt.addptr %out_base, %index : tensor<1x1x!tt.ptr<f32>>, tensor<1x1xi32>
      tt.store %out_ptr, %value, %mask : tensor<1x1x!tt.ptr<f32>>
      %next = arith.addi %index, %increment : tensor<1x1xi32>
      scf.yield %next : tensor<1x1xi32>
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func public @scalar_mask_loop(
// CHECK-SAME: [[OUTPUT:%[^ ,]+]]: !tt.ptr<f8E4M3FN>
// CHECK-SAME: [[INPUT:%[^ ,]+]]: !tt.ptr<bf16>
// CHECK-SAME: [[SCALE_PTR:%[^ ,]+]]: !tt.ptr<f32>
// CHECK-SAME: [[SIZE:%[^ ,)]+]]: i32
// CHECK-DAG: [[ZERO:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[ONE:%.+]] = arith.constant 1 : index
// CHECK: scf.for {{.*}} iter_args([[OFFSET:%.+]] = {{.*}}) -> (index)
// CHECK-DAG: [[LOAD_PTR:%.+]] = tts.make_tptr [[INPUT]] to sizes: [1], strides: {{\[}}[[ZERO]]{{\]}}, offsets: {{\[}}[[OFFSET]]{{\]}}
// CHECK-DAG: [[BOUND:%.+]] = arith.index_cast [[SIZE]] : i32 to index
// CHECK: [[IN_BOUNDS:%.+]] = arith.cmpi slt, [[OFFSET]], [[BOUND]] : index
// CHECK: [[EXTENT:%.+]] = arith.select [[IN_BOUNDS]], [[ONE]], [[ZERO]] : index
// CHECK: [[LOADED:%.+]] = "tts.load"([[LOAD_PTR]], [[EXTENT]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1x!tt.ptr<bf16>>, index) -> tensor<1xbf16>
// CHECK: [[CONVERTED:%.+]] = tt.fp_to_fp {{.*}}, rounding = rtne : tensor<1xf32> -> tensor<1xf8E4M3FN>
// CHECK: [[STORE_PTR:%.+]] = tts.make_tptr [[OUTPUT]] to sizes: [1], strides: {{\[}}[[ZERO]]{{\]}}, offsets: {{\[}}[[OFFSET]]{{\]}}
// CHECK: "tts.store"([[STORE_PTR]], [[CONVERTED]], [[EXTENT]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1x!tt.ptr<f8E4M3FN>>, tensor<1xf8E4M3FN>, index) -> ()

// CHECK-LABEL: tt.func public @constant_zero_singleton_mask_loop(
// CHECK-SAME: [[C0_INPUT:%[^ ,:]+]]: !tt.ptr<f32>
// CHECK-SAME: [[C0_OUTPUT:%[^ ,:]+]]: !tt.ptr<f32>
// CHECK-SAME: [[C0_BOUND:%[^ ,:)]+]]: i32
// CHECK-SAME: [[C0_ITERATIONS:%[^ ,:)]+]]: i32
// CHECK-DAG: [[C0_ZERO:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[C0_ONE:%.+]] = arith.constant 1 : index
// CHECK: scf.for {{.*}} to [[C0_ITERATIONS]] step {{.*}} iter_args([[C0_OFFSET:%[^ ,=]+]] = [[C0_ZERO]]) -> (index)
// CHECK-DAG: [[C0_LOAD_PTR:%.+]] = tts.make_tptr [[C0_INPUT]] to sizes: [1], strides: {{\[}}[[C0_ZERO]]{{\]}}, offsets: {{\[}}[[C0_OFFSET]]{{\]}}
// CHECK-DAG: [[C0_BOUND_INDEX:%.+]] = arith.index_cast [[C0_BOUND]] : i32 to index
// CHECK: [[C0_VALID:%.+]] = arith.cmpi slt, [[C0_OFFSET]], [[C0_BOUND_INDEX]] : index
// CHECK: [[C0_EXTENT:%.+]] = arith.select [[C0_VALID]], [[C0_ONE]], [[C0_ZERO]] : index
// CHECK-DAG: [[C0_LOADED:%.+]] = "tts.load"([[C0_LOAD_PTR]], [[C0_EXTENT]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1x!tt.ptr<f32>>, index) -> tensor<1xf32>
// CHECK-DAG: [[C0_STORE_PTR:%.+]] = tts.make_tptr [[C0_OUTPUT]] to sizes: [1], strides: {{\[}}[[C0_ZERO]]{{\]}}, offsets: {{\[}}[[C0_OFFSET]]{{\]}}
// CHECK: "tts.store"([[C0_STORE_PTR]], [[C0_LOADED]], [[C0_EXTENT]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1x!tt.ptr<f32>>, tensor<1xf32>, index) -> ()
// CHECK: tt.return

// CHECK-LABEL: tt.func public @constant_two_singleton_mask_loop(
// CHECK-SAME: [[C2_INPUT:%[^ ,:]+]]: !tt.ptr<f32>
// CHECK-SAME: [[C2_OUTPUT:%[^ ,:]+]]: !tt.ptr<f32>
// CHECK-SAME: [[C2_BOUND:%[^ ,:)]+]]: i32
// CHECK-SAME: [[C2_ITERATIONS:%[^ ,:)]+]]: i32
// CHECK-DAG: [[C2_ZERO:%.+]] = arith.constant 0 : index
// CHECK-DAG: [[C2_ONE:%.+]] = arith.constant 1 : index
// CHECK-DAG: [[C2_INITIAL:%.+]] = arith.constant 2 : index
// CHECK: scf.for {{.*}} to [[C2_ITERATIONS]] step {{.*}} iter_args([[C2_OFFSET:%[^ ,=]+]] = [[C2_INITIAL]]) -> (index)
// CHECK-DAG: [[C2_LOAD_PTR:%.+]] = tts.make_tptr [[C2_INPUT]] to sizes: [1], strides: {{\[}}[[C2_ZERO]]{{\]}}, offsets: {{\[}}[[C2_OFFSET]]{{\]}}
// CHECK-DAG: [[C2_BOUND_INDEX:%.+]] = arith.index_cast [[C2_BOUND]] : i32 to index
// CHECK: [[C2_VALID:%.+]] = arith.cmpi slt, [[C2_OFFSET]], [[C2_BOUND_INDEX]] : index
// CHECK: [[C2_EXTENT:%.+]] = arith.select [[C2_VALID]], [[C2_ONE]], [[C2_ZERO]] : index
// CHECK-DAG: [[C2_LOADED:%.+]] = "tts.load"([[C2_LOAD_PTR]], [[C2_EXTENT]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1x!tt.ptr<f32>>, index) -> tensor<1xf32>
// CHECK-DAG: [[C2_STORE_PTR:%.+]] = tts.make_tptr [[C2_OUTPUT]] to sizes: [1], strides: {{\[}}[[C2_ZERO]]{{\]}}, offsets: {{\[}}[[C2_OFFSET]]{{\]}}
// CHECK: "tts.store"([[C2_STORE_PTR]], [[C2_LOADED]], [[C2_EXTENT]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<1x!tt.ptr<f32>>, tensor<1xf32>, index) -> ()
// CHECK: tt.return

// CHECK-LABEL: tt.func public @expanded_scalar_singleton_mask_fallback(
// CHECK-SAME: [[RANK2_INPUT:%[^ ,:]+]]: !tt.ptr<f32>
// CHECK-SAME: [[RANK2_OUTPUT:%[^ ,:]+]]: !tt.ptr<f32>
// CHECK-SAME: [[RANK2_BOUND:%[^ ,:)]+]]: i32
// CHECK-DAG: [[RANK2_OTHER:%.+]] = arith.constant dense<-9.000000e+00> : tensor<1x1xf32>
// CHECK-DAG: [[RANK2_LIMIT:%.+]] = tt.splat [[RANK2_BOUND]] : i32 -> tensor<1x1xi32>
// CHECK: [[RANK2_MASK:%.+]] = arith.cmpi slt, %{{.*}}, [[RANK2_LIMIT]] : tensor<1x1xi32>
// CHECK: [[RANK2_LOADED:%.+]] = tt.load %{{.*}}, [[RANK2_MASK]], [[RANK2_OTHER]] : tensor<1x1x!tt.ptr<f32>>
// CHECK: tt.store %{{.*}}, [[RANK2_LOADED]], [[RANK2_MASK]] : tensor<1x1x!tt.ptr<f32>>
// CHECK: tt.return
