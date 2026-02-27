// RUN: triton-shared-opt --triton-to-structured --canonicalize --remove-dead-values -cse %s | FileCheck %s

module {
  tt.func public @nested_who_knows_how_many_levels(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %14 = tt.broadcast %13 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %15 = tt.addptr %14, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %16 = arith.muli %arg3, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18 = arith.muli %arg3, %c2_i32 : i32
    %19 = tt.splat %18 : i32 -> tensor<2x2xi32>
    %20 = arith.muli %arg3, %c2_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<2x2xi32>
    %22 = arith.muli %arg3, %c2_i32 : i32
    %23 = tt.splat %22 : i32 -> tensor<2x2xi32>
    %24:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %25 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      %26:3 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5, %arg9 = %arg6, %arg10 = %25) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>)  : i32 {
        %29 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %30 = tt.load %29 : tensor<2x2x!tt.ptr<f32>>
        %31:4 = scf.for %arg11 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg12 = %29, %arg13 = %arg9, %arg14 = %arg10, %arg15 = %30) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
          %33 = tt.addptr %arg12, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %34 = tt.load %33 : tensor<2x2x!tt.ptr<f32>>
          tt.store %arg13, %arg14 : tensor<2x2x!tt.ptr<f32>>
          %35 = tt.addptr %arg13, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %35, %arg15 : tensor<2x2x!tt.ptr<f32>>
          %36 = tt.addptr %35, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %36, %34 : tensor<2x2x!tt.ptr<f32>>
          %37 = tt.addptr %36, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %38:5 = scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg17 = %arg14, %arg18 = %33, %arg19 = %arg15, %arg20 = %34, %arg21 = %37) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
            %40 = tt.load %arg18 : tensor<2x2x!tt.ptr<f32>>
            %41:5 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20, %arg26 = %arg21, %arg27 = %40) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>)  : i32 {
              %42 = tt.addptr %arg23, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
              %43 = tt.load %42 : tensor<2x2x!tt.ptr<f32>>
              %44:5 = scf.for %arg28 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg29 = %42, %arg30 = %arg25, %arg31 = %arg26, %arg32 = %arg27, %arg33 = %43) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
                %45 = tt.addptr %arg29, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                %46 = tt.load %45 : tensor<2x2x!tt.ptr<f32>>
                tt.store %arg31, %arg32 : tensor<2x2x!tt.ptr<f32>>
                %47 = tt.addptr %arg31, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %47, %arg33 : tensor<2x2x!tt.ptr<f32>>
                %48 = tt.addptr %47, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %48, %46 : tensor<2x2x!tt.ptr<f32>>
                %49 = tt.addptr %48, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                %50:5 = scf.for %arg34 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg35 = %arg32, %arg36 = %45, %arg37 = %arg33, %arg38 = %46, %arg39 = %49) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                  %51 = tt.load %arg36 : tensor<2x2x!tt.ptr<f32>>
                  %52:5 = scf.for %arg40 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg41 = %arg36, %arg42 = %arg37, %arg43 = %arg38, %arg44 = %arg39, %arg45 = %51) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>)  : i32 {
                    %53 = tt.addptr %arg41, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                    %54 = tt.load %53 : tensor<2x2x!tt.ptr<f32>>
                    %55:5 = scf.for %arg46 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg47 = %53, %arg48 = %arg43, %arg49 = %arg44, %arg50 = %arg45, %arg51 = %54) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
                      %56 = tt.addptr %arg47, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      %57 = tt.load %56 : tensor<2x2x!tt.ptr<f32>>
                      tt.store %arg49, %arg50 : tensor<2x2x!tt.ptr<f32>>
                      %58 = tt.addptr %arg49, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      tt.store %58, %arg51 : tensor<2x2x!tt.ptr<f32>>
                      %59 = tt.addptr %58, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      tt.store %59, %57 : tensor<2x2x!tt.ptr<f32>>
                      %60 = tt.addptr %59, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                      %61:5 = scf.for %arg52 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg53 = %arg50, %arg54 = %56, %arg55 = %arg51, %arg56 = %57, %arg57 = %60) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                        %62 = tt.load %arg54 : tensor<2x2x!tt.ptr<f32>>
                        %63:4 = scf.for %arg58 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg59 = %arg54, %arg60 = %arg55, %arg61 = %arg56, %arg62 = %arg57) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                          %64 = tt.addptr %arg59, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                          %65 = tt.load %64 : tensor<2x2x!tt.ptr<f32>>
                          %66:3 = scf.for %arg63 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg64 = %64, %arg65 = %arg61, %arg66 = %arg62) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                            %67 = tt.addptr %arg64, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            %68 = tt.load %67 : tensor<2x2x!tt.ptr<f32>>
                            tt.store %arg66, %62 : tensor<2x2x!tt.ptr<f32>>
                            %69 = tt.addptr %arg66, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            tt.store %69, %65 : tensor<2x2x!tt.ptr<f32>>
                            %70 = tt.addptr %69, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            tt.store %70, %68 : tensor<2x2x!tt.ptr<f32>>
                            %71 = tt.addptr %70, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                            scf.yield %67, %68, %71 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                          }
                          scf.yield %66#0, %65, %66#1, %66#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                        }
                        scf.yield %62, %63#0, %63#1, %63#2, %63#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                      }
                      scf.yield %61#1, %61#3, %61#4, %61#0, %61#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>
                    }
                    scf.yield %55#0, %55#4, %55#1, %55#2, %55#3 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>
                  }
                  scf.yield %52#4, %52#0, %52#1, %52#2, %52#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
                }
                scf.yield %50#1, %50#3, %50#4, %50#0, %50#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>
              }
              scf.yield %44#0, %44#4, %44#1, %44#2, %44#3 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>
            }
            scf.yield %41#4, %41#0, %41#1, %41#2, %41#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
          }
          %39:5 = scf.for %arg16 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg17 = %38#0, %arg18 = %38#1, %arg19 = %38#2, %arg20 = %38#3, %arg21 = %38#4) -> (tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
            %40 = tt.load %arg18 : tensor<2x2x!tt.ptr<f32>>
            %41:4 = scf.for %arg22 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20, %arg26 = %arg21) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
              %42 = tt.addptr %arg23, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
              %43 = tt.load %42 : tensor<2x2x!tt.ptr<f32>>
              %44:3 = scf.for %arg27 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg28 = %42, %arg29 = %arg25, %arg30 = %arg26) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
                %45 = tt.addptr %arg28, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                %46 = tt.load %45 : tensor<2x2x!tt.ptr<f32>>
                tt.store %arg30, %40 : tensor<2x2x!tt.ptr<f32>>
                %47 = tt.addptr %arg30, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %47, %43 : tensor<2x2x!tt.ptr<f32>>
                %48 = tt.addptr %47, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                tt.store %48, %46 : tensor<2x2x!tt.ptr<f32>>
                %49 = tt.addptr %48, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
                scf.yield %45, %46, %49 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
              }
              scf.yield %44#0, %43, %44#1, %44#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
            }
            scf.yield %40, %41#0, %41#1, %41#2, %41#3 : tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2x!tt.ptr<f32>>
          }
          scf.yield %39#1, %39#4, %39#0, %39#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xf32>
        }
        %32 = tt.addptr %31#0, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %32, %31#1, %31#2 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>
      }
      %27:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %26#0, %arg9 = %26#1) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %29 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        %30:2 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
          %32 = tt.addptr %arg11, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %33 = tt.load %32 : tensor<2x2x!tt.ptr<f32>>
          %34:2 = scf.for %arg13 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg14 = %32, %arg15 = %arg12) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
            %35 = tt.addptr %arg14, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            %36 = tt.load %35 : tensor<2x2x!tt.ptr<f32>>
            tt.store %arg15, %29 : tensor<2x2x!tt.ptr<f32>>
            %37 = tt.addptr %arg15, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            tt.store %37, %33 : tensor<2x2x!tt.ptr<f32>>
            %38 = tt.addptr %37, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            tt.store %38, %36 : tensor<2x2x!tt.ptr<f32>>
            %39 = tt.addptr %38, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
            scf.yield %35, %39 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
          }
          scf.yield %34#0, %34#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
        }
        %31 = tt.addptr %30#0, %21 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %31, %30#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %28 = tt.addptr %27#0, %23 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %28, %27#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-LABEL:   tt.func public @nested_who_knows_how_many_levels(
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<f32>, %[[ARG1:.*]]: !tt.ptr<f32>, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32) attributes {noinline = false} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK:           %[[INDEX_CAST_1:.*]] = arith.index_cast %[[ARG3]] : i32 to index
// CHECK:           %[[MULI_0:.*]] = arith.muli %[[ARG3]], %[[CONSTANT_3]] : i32
// CHECK:           %[[INDEX_CAST_2:.*]] = arith.index_cast %[[MULI_0]] : i32 to index
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_0]], %[[VAL_2:.*]] = %[[CONSTANT_0]]) -> (index, index)  : i32 {
// CHECK:             %[[MAKE_TPTR_0:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_1]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:             %[[VAL_3:.*]] = "tts.load"(%[[MAKE_TPTR_0]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:             %[[FOR_1:.*]]:3 = scf.for %[[VAL_4:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_5:.*]] = %[[VAL_1]], %[[VAL_6:.*]] = %[[VAL_2]], %[[VAL_7:.*]] = %[[VAL_3]]) -> (index, index, tensor<2x2xf32>)  : i32 {
// CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_5]], %[[INDEX_CAST_2]] : index
// CHECK:               %[[MAKE_TPTR_1:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_0]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:               %[[VAL_8:.*]] = "tts.load"(%[[MAKE_TPTR_1]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:               %[[FOR_2:.*]]:4 = scf.for %[[VAL_9:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_10:.*]] = %[[ADDI_0]], %[[VAL_11:.*]] = %[[VAL_6]], %[[VAL_12:.*]] = %[[VAL_7]], %[[VAL_13:.*]] = %[[VAL_8]]) -> (index, index, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
// CHECK:                 %[[MAKE_TPTR_2:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_11]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 %[[ADDI_1:.*]] = arith.addi %[[VAL_10]], %[[INDEX_CAST_2]] : index
// CHECK:                 %[[MAKE_TPTR_3:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_1]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 %[[VAL_14:.*]] = "tts.load"(%[[MAKE_TPTR_3]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                 "tts.store"(%[[MAKE_TPTR_2]], %[[VAL_12]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                 %[[ADDI_2:.*]] = arith.addi %[[VAL_11]], %[[INDEX_CAST_2]] : index
// CHECK:                 %[[MAKE_TPTR_4:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_2]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 "tts.store"(%[[MAKE_TPTR_4]], %[[VAL_13]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                 %[[ADDI_3:.*]] = arith.addi %[[ADDI_2]], %[[INDEX_CAST_2]] : index
// CHECK:                 %[[MAKE_TPTR_5:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_3]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 "tts.store"(%[[MAKE_TPTR_5]], %[[VAL_14]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                 %[[ADDI_4:.*]] = arith.addi %[[ADDI_3]], %[[INDEX_CAST_2]] : index
// CHECK:                 %[[FOR_3:.*]]:2 = scf.for %[[VAL_15:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_16:.*]] = %[[ADDI_1]], %[[VAL_17:.*]] = %[[ADDI_4]]) -> (index, index)  : i32 {
// CHECK:                   %[[MAKE_TPTR_6:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_16]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                   %[[VAL_18:.*]] = "tts.load"(%[[MAKE_TPTR_6]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                   %[[FOR_4:.*]]:3 = scf.for %[[VAL_19:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_20:.*]] = %[[VAL_16]], %[[VAL_21:.*]] = %[[VAL_17]], %[[VAL_22:.*]] = %[[VAL_18]]) -> (index, index, tensor<2x2xf32>)  : i32 {
// CHECK:                     %[[ADDI_5:.*]] = arith.addi %[[VAL_20]], %[[INDEX_CAST_2]] : index
// CHECK:                     %[[MAKE_TPTR_7:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_5]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                     %[[VAL_23:.*]] = "tts.load"(%[[MAKE_TPTR_7]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                     %[[FOR_5:.*]]:4 = scf.for %[[VAL_24:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_25:.*]] = %[[ADDI_5]], %[[VAL_26:.*]] = %[[VAL_21]], %[[VAL_27:.*]] = %[[VAL_22]], %[[VAL_28:.*]] = %[[VAL_23]]) -> (index, index, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
// CHECK:                       %[[MAKE_TPTR_8:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_26]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       %[[ADDI_6:.*]] = arith.addi %[[VAL_25]], %[[INDEX_CAST_2]] : index
// CHECK:                       %[[MAKE_TPTR_9:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_6]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       %[[VAL_29:.*]] = "tts.load"(%[[MAKE_TPTR_9]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                       "tts.store"(%[[MAKE_TPTR_8]], %[[VAL_27]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       %[[ADDI_7:.*]] = arith.addi %[[VAL_26]], %[[INDEX_CAST_2]] : index
// CHECK:                       %[[MAKE_TPTR_10:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_7]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       "tts.store"(%[[MAKE_TPTR_10]], %[[VAL_28]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       %[[ADDI_8:.*]] = arith.addi %[[ADDI_7]], %[[INDEX_CAST_2]] : index
// CHECK:                       %[[MAKE_TPTR_11:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_8]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       "tts.store"(%[[MAKE_TPTR_11]], %[[VAL_29]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       %[[ADDI_9:.*]] = arith.addi %[[ADDI_8]], %[[INDEX_CAST_2]] : index
// CHECK:                       %[[POISON_0:.*]] = ub.poison : tensor<2x2xf32>
// CHECK:                       %[[FOR_6:.*]]:4 = scf.for %[[VAL_30:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_31:.*]] = %[[POISON_0]], %[[VAL_32:.*]] = %[[ADDI_6]], %[[VAL_33:.*]] = %[[POISON_0]], %[[VAL_34:.*]] = %[[ADDI_9]]) -> (tensor<2x2xf32>, index, tensor<2x2xf32>, index)  : i32 {
// CHECK:                         %[[MAKE_TPTR_12:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_32]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                         %[[VAL_35:.*]] = "tts.load"(%[[MAKE_TPTR_12]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                         %[[FOR_7:.*]]:4 = scf.for %[[VAL_36:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_37:.*]] = %[[VAL_32]], %[[VAL_38:.*]] = %[[POISON_0]], %[[VAL_39:.*]] = %[[VAL_34]], %[[VAL_40:.*]] = %[[VAL_35]]) -> (index, tensor<2x2xf32>, index, tensor<2x2xf32>)  : i32 {
// CHECK:                           %[[ADDI_10:.*]] = arith.addi %[[VAL_37]], %[[INDEX_CAST_2]] : index
// CHECK:                           %[[MAKE_TPTR_13:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_10]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                           %[[VAL_41:.*]] = "tts.load"(%[[MAKE_TPTR_13]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                           %[[FOR_8:.*]]:4 = scf.for %[[VAL_42:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_43:.*]] = %[[ADDI_10]], %[[VAL_44:.*]] = %[[VAL_39]], %[[VAL_45:.*]] = %[[VAL_40]], %[[VAL_46:.*]] = %[[VAL_41]]) -> (index, index, tensor<2x2xf32>, tensor<2x2xf32>)  : i32 {
// CHECK:                             %[[MAKE_TPTR_14:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_44]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                             %[[ADDI_11:.*]] = arith.addi %[[VAL_43]], %[[INDEX_CAST_2]] : index
// CHECK:                             %[[MAKE_TPTR_15:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_11]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                             %[[VAL_47:.*]] = "tts.load"(%[[MAKE_TPTR_15]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                             "tts.store"(%[[MAKE_TPTR_14]], %[[VAL_45]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                             %[[ADDI_12:.*]] = arith.addi %[[VAL_44]], %[[INDEX_CAST_2]] : index
// CHECK:                             %[[MAKE_TPTR_16:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_12]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                             "tts.store"(%[[MAKE_TPTR_16]], %[[VAL_46]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                             %[[ADDI_13:.*]] = arith.addi %[[ADDI_12]], %[[INDEX_CAST_2]] : index
// CHECK:                             %[[MAKE_TPTR_17:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_13]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                             "tts.store"(%[[MAKE_TPTR_17]], %[[VAL_47]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                             %[[ADDI_14:.*]] = arith.addi %[[ADDI_13]], %[[INDEX_CAST_2]] : index
// CHECK:                             %[[FOR_9:.*]]:4 = scf.for %[[VAL_48:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_49:.*]] = %[[POISON_0]], %[[VAL_50:.*]] = %[[ADDI_11]], %[[VAL_51:.*]] = %[[POISON_0]], %[[VAL_52:.*]] = %[[ADDI_14]]) -> (tensor<2x2xf32>, index, tensor<2x2xf32>, index)  : i32 {
// CHECK:                               %[[MAKE_TPTR_18:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_50]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                               %[[VAL_53:.*]] = "tts.load"(%[[MAKE_TPTR_18]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                               %[[FOR_10:.*]]:3 = scf.for %[[VAL_54:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_55:.*]] = %[[VAL_50]], %[[VAL_56:.*]] = %[[POISON_0]], %[[VAL_57:.*]] = %[[VAL_52]]) -> (index, tensor<2x2xf32>, index)  : i32 {
// CHECK:                                 %[[ADDI_15:.*]] = arith.addi %[[VAL_55]], %[[INDEX_CAST_2]] : index
// CHECK:                                 %[[MAKE_TPTR_19:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_15]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                                 %[[VAL_58:.*]] = "tts.load"(%[[MAKE_TPTR_19]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                                 %[[FOR_11:.*]]:2 = scf.for %[[VAL_59:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_60:.*]] = %[[ADDI_15]], %[[VAL_61:.*]] = %[[VAL_57]]) -> (index, index)  : i32 {
// CHECK:                                   %[[MAKE_TPTR_20:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_61]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                                   %[[ADDI_16:.*]] = arith.addi %[[VAL_60]], %[[INDEX_CAST_2]] : index
// CHECK:                                   %[[MAKE_TPTR_21:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_16]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                                   %[[VAL_62:.*]] = "tts.load"(%[[MAKE_TPTR_21]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                                   "tts.store"(%[[MAKE_TPTR_20]], %[[VAL_53]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                                   %[[ADDI_17:.*]] = arith.addi %[[VAL_61]], %[[INDEX_CAST_2]] : index
// CHECK:                                   %[[MAKE_TPTR_22:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_17]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                                   "tts.store"(%[[MAKE_TPTR_22]], %[[VAL_58]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                                   %[[ADDI_18:.*]] = arith.addi %[[ADDI_17]], %[[INDEX_CAST_2]] : index
// CHECK:                                   %[[MAKE_TPTR_23:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_18]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                                   "tts.store"(%[[MAKE_TPTR_23]], %[[VAL_62]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                                   %[[ADDI_19:.*]] = arith.addi %[[ADDI_18]], %[[INDEX_CAST_2]] : index
// CHECK:                                   scf.yield %[[ADDI_16]], %[[ADDI_19]] : index, index
// CHECK:                                 }
// CHECK:                                 scf.yield %[[VAL_63:.*]]#0, %[[VAL_58]], %[[VAL_63]]#1 : index, tensor<2x2xf32>, index
// CHECK:                               }
// CHECK:                               scf.yield %[[VAL_53]], %[[VAL_64:.*]]#0, %[[VAL_64]]#1, %[[VAL_64]]#2 : tensor<2x2xf32>, index, tensor<2x2xf32>, index
// CHECK:                             }
// CHECK:                             scf.yield %[[VAL_65:.*]]#1, %[[VAL_65]]#3, %[[VAL_65]]#0, %[[VAL_65]]#2 : index, index, tensor<2x2xf32>, tensor<2x2xf32>
// CHECK:                           }
// CHECK:                           scf.yield %[[VAL_66:.*]]#0, %[[VAL_66]]#3, %[[VAL_66]]#1, %[[VAL_66]]#2 : index, tensor<2x2xf32>, index, tensor<2x2xf32>
// CHECK:                         }
// CHECK:                         scf.yield %[[VAL_67:.*]]#3, %[[VAL_67]]#0, %[[VAL_67]]#1, %[[VAL_67]]#2 : tensor<2x2xf32>, index, tensor<2x2xf32>, index
// CHECK:                       }
// CHECK:                       scf.yield %[[VAL_68:.*]]#1, %[[VAL_68]]#3, %[[VAL_68]]#0, %[[VAL_68]]#2 : index, index, tensor<2x2xf32>, tensor<2x2xf32>
// CHECK:                     }
// CHECK:                     scf.yield %[[VAL_69:.*]]#0, %[[VAL_69]]#1, %[[VAL_69]]#2 : index, index, tensor<2x2xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_70:.*]]#0, %[[VAL_70]]#1 : index, index
// CHECK:                 }
// CHECK:                 %[[POISON_1:.*]] = ub.poison : tensor<2x2xf32>
// CHECK:                 %[[FOR_12:.*]]:4 = scf.for %[[VAL_71:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_72:.*]] = %[[POISON_1]], %[[VAL_73:.*]] = %[[VAL_74:.*]]#0, %[[VAL_75:.*]] = %[[POISON_1]], %[[VAL_76:.*]] = %[[VAL_74]]#1) -> (tensor<2x2xf32>, index, tensor<2x2xf32>, index)  : i32 {
// CHECK:                   %[[MAKE_TPTR_24:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_73]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                   %[[VAL_77:.*]] = "tts.load"(%[[MAKE_TPTR_24]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                   %[[FOR_13:.*]]:3 = scf.for %[[VAL_78:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_79:.*]] = %[[VAL_73]], %[[VAL_80:.*]] = %[[POISON_1]], %[[VAL_81:.*]] = %[[VAL_76]]) -> (index, tensor<2x2xf32>, index)  : i32 {
// CHECK:                     %[[ADDI_20:.*]] = arith.addi %[[VAL_79]], %[[INDEX_CAST_2]] : index
// CHECK:                     %[[MAKE_TPTR_25:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_20]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                     %[[VAL_82:.*]] = "tts.load"(%[[MAKE_TPTR_25]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                     %[[FOR_14:.*]]:2 = scf.for %[[VAL_83:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_84:.*]] = %[[ADDI_20]], %[[VAL_85:.*]] = %[[VAL_81]]) -> (index, index)  : i32 {
// CHECK:                       %[[MAKE_TPTR_26:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_85]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       %[[ADDI_21:.*]] = arith.addi %[[VAL_84]], %[[INDEX_CAST_2]] : index
// CHECK:                       %[[MAKE_TPTR_27:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_21]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       %[[VAL_86:.*]] = "tts.load"(%[[MAKE_TPTR_27]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                       "tts.store"(%[[MAKE_TPTR_26]], %[[VAL_77]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       %[[ADDI_22:.*]] = arith.addi %[[VAL_85]], %[[INDEX_CAST_2]] : index
// CHECK:                       %[[MAKE_TPTR_28:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_22]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       "tts.store"(%[[MAKE_TPTR_28]], %[[VAL_82]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       %[[ADDI_23:.*]] = arith.addi %[[ADDI_22]], %[[INDEX_CAST_2]] : index
// CHECK:                       %[[MAKE_TPTR_29:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_23]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                       "tts.store"(%[[MAKE_TPTR_29]], %[[VAL_86]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                       %[[ADDI_24:.*]] = arith.addi %[[ADDI_23]], %[[INDEX_CAST_2]] : index
// CHECK:                       scf.yield %[[ADDI_21]], %[[ADDI_24]] : index, index
// CHECK:                     }
// CHECK:                     scf.yield %[[VAL_87:.*]]#0, %[[VAL_82]], %[[VAL_87]]#1 : index, tensor<2x2xf32>, index
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_77]], %[[VAL_88:.*]]#0, %[[VAL_88]]#1, %[[VAL_88]]#2 : tensor<2x2xf32>, index, tensor<2x2xf32>, index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_89:.*]]#1, %[[VAL_89]]#3, %[[VAL_89]]#0, %[[VAL_89]]#2 : index, index, tensor<2x2xf32>, tensor<2x2xf32>
// CHECK:               }
// CHECK:               %[[ADDI_25:.*]] = arith.addi %[[VAL_90:.*]]#0, %[[INDEX_CAST_2]] : index
// CHECK:               scf.yield %[[ADDI_25]], %[[VAL_90]]#1, %[[VAL_90]]#2 : index, index, tensor<2x2xf32>
// CHECK:             }
// CHECK:             %[[FOR_15:.*]]:2 = scf.for %[[VAL_91:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_92:.*]] = %[[VAL_93:.*]]#0, %[[VAL_94:.*]] = %[[VAL_93]]#1) -> (index, index)  : i32 {
// CHECK:               %[[MAKE_TPTR_30:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_92]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:               %[[VAL_95:.*]] = "tts.load"(%[[MAKE_TPTR_30]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:               %[[FOR_16:.*]]:2 = scf.for %[[VAL_96:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_97:.*]] = %[[VAL_92]], %[[VAL_98:.*]] = %[[VAL_94]]) -> (index, index)  : i32 {
// CHECK:                 %[[ADDI_26:.*]] = arith.addi %[[VAL_97]], %[[INDEX_CAST_2]] : index
// CHECK:                 %[[MAKE_TPTR_31:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_26]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 %[[VAL_99:.*]] = "tts.load"(%[[MAKE_TPTR_31]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                 %[[FOR_17:.*]]:2 = scf.for %[[VAL_100:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_101:.*]] = %[[ADDI_26]], %[[VAL_102:.*]] = %[[VAL_98]]) -> (index, index)  : i32 {
// CHECK:                   %[[MAKE_TPTR_32:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[VAL_102]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                   %[[ADDI_27:.*]] = arith.addi %[[VAL_101]], %[[INDEX_CAST_2]] : index
// CHECK:                   %[[MAKE_TPTR_33:.*]] = tts.make_tptr %[[ARG0]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_27]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                   %[[VAL_103:.*]] = "tts.load"(%[[MAKE_TPTR_33]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                   "tts.store"(%[[MAKE_TPTR_32]], %[[VAL_95]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                   %[[ADDI_28:.*]] = arith.addi %[[VAL_102]], %[[INDEX_CAST_2]] : index
// CHECK:                   %[[MAKE_TPTR_34:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_28]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                   "tts.store"(%[[MAKE_TPTR_34]], %[[VAL_99]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                   %[[ADDI_29:.*]] = arith.addi %[[ADDI_28]], %[[INDEX_CAST_2]] : index
// CHECK:                   %[[MAKE_TPTR_35:.*]] = tts.make_tptr %[[ARG1]] to sizes: [2, 2], strides: {{\[}}%[[INDEX_CAST_0]], %[[INDEX_CAST_1]]], offsets: {{\[}}%[[ADDI_29]], %[[CONSTANT_0]]], shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                   "tts.store"(%[[MAKE_TPTR_35]], %[[VAL_103]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                   %[[ADDI_30:.*]] = arith.addi %[[ADDI_29]], %[[INDEX_CAST_2]] : index
// CHECK:                   scf.yield %[[ADDI_27]], %[[ADDI_30]] : index, index
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_104:.*]]#0, %[[VAL_104]]#1 : index, index
// CHECK:               }
// CHECK:               %[[ADDI_31:.*]] = arith.addi %[[VAL_105:.*]]#0, %[[INDEX_CAST_2]] : index
// CHECK:               scf.yield %[[ADDI_31]], %[[VAL_105]]#1 : index, index
// CHECK:             }
// CHECK:             %[[ADDI_32:.*]] = arith.addi %[[VAL_106:.*]]#0, %[[INDEX_CAST_2]] : index
// CHECK:             scf.yield %[[ADDI_32]], %[[VAL_106]]#1 : index, index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }