//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Transform/VerifyTritonCheckpoint/VerifyTritonCheckpoint.h"

#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "verify-triton-checkpoint"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_VERIFYTRITONCHECKPOINT
#include "triton-shared/Transform/VerifyTritonCheckpoint/Passes.h.inc"

namespace {

class VerifyTritonCheckpointPass
    : public ::impl::VerifyTritonCheckpointBase<VerifyTritonCheckpointPass> {
public:
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    WalkResult walkResult = moduleOp.walk([&](Operation *op) -> WalkResult {
      Dialect *dialect = op->getDialect();
      if (!dialect) {
        return WalkResult::advance();
      }
      if (isa<mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
              mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
              mlir::tts::TritonStructuredDialect, mlir::tptr::TPtrDialect,
              mlir::ttx::TritonTilingExtDialect>(dialect)) {
        op->emitOpError(
            "unexpected Triton-related op remaining after lowering");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createVerifyTritonCheckpointPass() {
  return std::make_unique<VerifyTritonCheckpointPass>();
}
