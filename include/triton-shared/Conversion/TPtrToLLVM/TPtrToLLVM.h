//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TPTRTOLLVM_TPTRTOLLVM_H
#define TRITON_CONVERSION_TPTRTOLLVM_TPTRTOLLVM_H

#include <memory>

namespace mlir {
class DialectRegistry;
class LLVMTypeConverter;
class RewritePatternSet;
namespace tptr {
/// Populate the convert to LLVM patterns for the `tptr` dialect.
void populateTPtrToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns);
/// Register the convert to LLVM interface for the `tptr` dialect.
void registerConvertTPtrToLLVMInterface(DialectRegistry &registry);
} // namespace tptr
} // namespace mlir

#endif // TRITON_CONVERSION_TPTRTOLLVM_TPTRTOLLVM_H
