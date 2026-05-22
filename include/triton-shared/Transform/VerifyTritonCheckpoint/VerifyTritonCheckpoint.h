//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TRANSFORM_VERIFYTRITONCHECKPOINT_VERIFYTRITONCHECKPOINT_H
#define TRITON_TRANSFORM_VERIFYTRITONCHECKPOINT_VERIFYTRITONCHECKPOINT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createVerifyTritonCheckpointPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_TRANSFORM_VERIFYTRITONCHECKPOINT_VERIFYTRITONCHECKPOINT_H
