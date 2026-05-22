//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TRITON_STRUCTURED_IR_TRITON_STRUCTURED_DIALECT_H_
#define MLIR_DIALECT_TRITON_STRUCTURED_IR_TRITON_STRUCTURED_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace tts {
namespace utils {
mlir::Value getScalarValue(mlir::Value operand, mlir::Location loc,
                           mlir::OpBuilder &builder);

// Extract the source pointer type from various input types used in pointer
// analysis. This is primarily used to determine the base pointer type that
// serves as the "source" for structured state tracking in operations like
// tts.get_structured_state.
//
// Examples:
//   - Tensor of pointers (tensor<!tt.ptr<f32>>):
//       Returns !tt.ptr<f32> (the element type)
//
//   - Pointer to tensor (!tt.ptr<tensor<16xf32>>):
//       Returns !tt.ptr<f32> (pointer to element type with address space 1)
//
//   - Scalar pointer (!tt.ptr<f32>):
//       Returns !tt.ptr<f32> (unchanged)
//
//   - Tensor of indices (tensor<16xi32>):
//       Returns !tt.ptr<none> (no meaningful pointer type)
//
//   - Other types:
//       Returns !tt.ptr<none> (fallback for non-pointer types)
Type getSrcPtrType(Type t);

// Check if a value has a valid pointer type with a concrete pointee.
// Returns true if the value is a pointer that points to a concrete type
// (not NoneType).
bool hasPtrValue(Value v);
} // namespace utils
} // namespace tts
} // namespace mlir

//===----------------------------------------------------------------------===//
// TritonStructured Operations
//===----------------------------------------------------------------------===//
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h.inc"

// Include the auto-generated header file containing the declarations of the
// TritonStructured operations.
#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.h.inc"

#endif
