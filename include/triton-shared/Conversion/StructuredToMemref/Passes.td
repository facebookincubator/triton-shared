//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_TO_MEMREF_CONVERSION_PASSES
#define STRUCTURED_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMemref : Pass<"structured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
}

#endif
