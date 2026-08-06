// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --cse --canonicalize %s | FileCheck %s --implicit-check-not=tts.make_gather_scatter_tptr

// Combining broadcast boundary masks for two dimensions leaves one side only
// partially parsed. The load must remain elementwise-masked when that side has
// no OpFoldResult that can be materialized as a structured mask.
module {
// CHECK-LABEL: tt.func public @nested_bool_and_mask_fallback(
// CHECK-DAG:     [[OTHER:%.+]] = arith.constant dense<-128> : tensor<2x2xi8>
// CHECK-DAG:     [[X_BOUND:%.+]] = tt.broadcast %{{.*}} : tensor<1x2xi1> -> tensor<2x2xi1>
// CHECK-DAG:     [[Y_BOUND:%.+]] = tt.broadcast %{{.*}} : tensor<2x1xi1> -> tensor<2x2xi1>
// CHECK:         [[MASK:%.+]] = arith.andi [[Y_BOUND]], [[X_BOUND]] : tensor<2x2xi1>
// CHECK:         tt.load %{{.*}}, [[MASK]], [[OTHER]] : tensor<2x2x!tt.ptr<i8>>
  tt.func public @nested_bool_and_mask_fallback(%in: !tt.ptr<i8>) -> tensor<2x2xi8> attributes {noinline = false} {
    %cst = arith.constant dense<-128> : tensor<2x2xi8>
    %c0x = arith.constant dense<0> : tensor<1x2xi64>
    %c4x = arith.constant dense<4> : tensor<1x2xi64>
    %c0y = arith.constant dense<0> : tensor<2x1xi64>
    %c6y = arith.constant dense<6> : tensor<2x1xi64>
    %c2x = arith.constant dense<2> : tensor<1x2xi32>
    %c3y = arith.constant dense<3> : tensor<2x1xi32>

    %range = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %x = tt.expand_dims %range {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %xmul = arith.muli %x, %c2x : tensor<1x2xi32>
    %xext = arith.extsi %xmul : tensor<1x2xi32> to tensor<1x2xi64>
    %xge = arith.cmpi sge, %xext, %c0x : tensor<1x2xi64>
    %xlt = arith.cmpi slt, %xext, %c4x : tensor<1x2xi64>
    %xbound = arith.andi %xge, %xlt : tensor<1x2xi1>

    %y = tt.expand_dims %range {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %ymul = arith.muli %y, %c3y : tensor<2x1xi32>
    %yext = arith.extsi %ymul : tensor<2x1xi32> to tensor<2x1xi64>
    %yge = arith.cmpi sge, %yext, %c0y : tensor<2x1xi64>
    %ylt = arith.cmpi slt, %yext, %c6y : tensor<2x1xi64>
    %ybound = arith.andi %yge, %ylt : tensor<2x1xi1>

    %xb = tt.broadcast %xbound : tensor<1x2xi1> -> tensor<2x2xi1>
    %yb = tt.broadcast %ybound : tensor<2x1xi1> -> tensor<2x2xi1>
    %mask = arith.andi %yb, %xb : tensor<2x2xi1>
    %xoffb = tt.broadcast %xmul : tensor<1x2xi32> -> tensor<2x2xi32>
    %yoffb = tt.broadcast %ymul : tensor<2x1xi32> -> tensor<2x2xi32>
    %off = arith.addi %xoffb, %yoffb : tensor<2x2xi32>
    %base = tt.splat %in : !tt.ptr<i8> -> tensor<2x2x!tt.ptr<i8>>
    %ptr = tt.addptr %base, %off : tensor<2x2x!tt.ptr<i8>>, tensor<2x2xi32>
    %value = tt.load %ptr, %mask, %cst : tensor<2x2x!tt.ptr<i8>>
    tt.return %value : tensor<2x2xi8>
  }
}
