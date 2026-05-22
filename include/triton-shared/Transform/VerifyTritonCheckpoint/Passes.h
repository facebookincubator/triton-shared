//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef VERIFY_TRITON_CHECKPOINT_TRANSFORM_PASSES_H
#define VERIFY_TRITON_CHECKPOINT_TRANSFORM_PASSES_H

#include "triton-shared/Transform/VerifyTritonCheckpoint/VerifyTritonCheckpoint.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Transform/VerifyTritonCheckpoint/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
