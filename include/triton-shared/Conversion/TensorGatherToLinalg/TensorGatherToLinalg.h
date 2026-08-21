//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TENSOR_GATHER_TO_LINALG_TENSOR_GATHER_TO_LINALG_H
#define TRITON_CONVERSION_TENSOR_GATHER_TO_LINALG_TENSOR_GATHER_TO_LINALG_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTensorGatherToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TENSOR_GATHER_TO_LINALG_TENSOR_GATHER_TO_LINALG_H
