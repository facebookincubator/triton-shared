//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_GATHER_TO_LINALG_CONVERSION_PASSES_H
#define TENSOR_GATHER_TO_LINALG_CONVERSION_PASSES_H

#include "triton-shared/Conversion/TensorGatherToLinalg/TensorGatherToLinalg.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Conversion/TensorGatherToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
