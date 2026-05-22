//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef VERIFY_TRITON_CHECKPOINT_TRANSFORM_PASSES
#define VERIFY_TRITON_CHECKPOINT_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def VerifyTritonCheckpoint : Pass<"verify-triton-checkpoint", "mlir::ModuleOp"> {
  let summary = "Verify that no Triton-related ops remain in the IR";
  let constructor = "triton::createVerifyTritonCheckpointPass()";
}

#endif
