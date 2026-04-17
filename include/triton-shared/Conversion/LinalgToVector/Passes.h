#ifndef LINALG_TO_VECTOR_CONVERSION_PASSES_H
#define LINALG_TO_VECTOR_CONVERSION_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "triton-shared/Conversion/LinalgToVector/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
