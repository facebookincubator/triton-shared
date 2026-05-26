// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s

// Goal of this test is just to not crash on ops that are unsupported by
// PtrAnalysis like arith.select.

module {
  tt.func public @concat_2D_jagged(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: i64, %arg4: i64) attributes {noinline = false} {
    %cst = arith.constant dense<256> : tensor<1x1xi64>
    %0 = arith.cmpi slt, %arg3, %arg4 : i64
    %1 = arith.select %0, %arg0, %arg1 : !tt.ptr<bf16>
    %2 = tt.splat %1 : !tt.ptr<bf16> -> tensor<1x256x!tt.ptr<bf16>>
    %3 = tt.splat %arg3 : i64 -> tensor<1x1xi64>
    %4 = arith.muli %3, %cst : tensor<1x1xi64>
    %5 = tt.broadcast %4 : tensor<1x1xi64> -> tensor<1x256xi64>
    %6 = tt.addptr %2, %5 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi64>
    %7 = tt.load %6 : tensor<1x256x!tt.ptr<bf16>>
    %8 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<1x256x!tt.ptr<bf16>>
    tt.store %8, %7 : tensor<1x256x!tt.ptr<bf16>>
    tt.return
  }
}
