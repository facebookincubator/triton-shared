//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_GATHER_TO_LINALG_CONVERSION_PASSES
#define TENSOR_GATHER_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TensorGatherToLinalg : Pass<"tensor-gather-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert tt.gather-derived tensor.gather back to linalg.generic";
  let description = [{
    Rewrites `tensor.gather` ops that were produced by lowering `tt.gather`
    (identified by a marker attribute set by TritonArithToLinalg's
    GatherConverter) back into the single-axis `linalg.generic` +
    `tensor.extract` form. This is necessary because upstream
    `tensor.gather` has no `BufferizableOpInterface` implementation, so it
    cannot survive to bufferization. `tensor.gather` ops without the marker
    attribute are left untouched.
  }];
  let constructor = "triton::createTensorGatherToLinalgPass()";
}

#endif
