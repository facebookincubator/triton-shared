//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_UTILITY_H
#define TRITON_SHARED_UTILITY_H

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
// Return true if the input type is a triton pointer or a tensor of triton
// pointers
bool isPtrTypeLike(Type t);

// Marks a tensor.gather op (emitted by TritonArithToLinalg's GatherConverter)
// as having been lowered from tt.gather, carrying the original tt.gather
// axis. Only tensor.gather ops carrying this attribute are rewritten back to
// linalg.generic by the tensor-gather-to-linalg pass.
static constexpr char const *kTensorGatherFromTtGatherAxisAttrName =
    "triton_shared.tt_gather_axis";
} // namespace triton

} // namespace mlir

#endif // TRITON_SHARED_UTILITY_H
