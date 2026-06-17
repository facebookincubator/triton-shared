//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_PTR_TO_PTR_CONVERSION_PASSES
#define TRITON_PTR_TO_PTR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonPtrToPtr : Pass<"triton-ptr-to-ptr", "mlir::ModuleOp"> {
  let summary = "Convert triton pointer to mlir ptr";
  let constructor = "triton::createTritonPtrToPtrPass()";
}

#endif
