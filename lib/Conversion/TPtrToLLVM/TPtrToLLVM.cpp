//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/IR/TypeUtilities.h"
#include <type_traits>

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// FromPtrOpConversion
//===----------------------------------------------------------------------===//
struct FromPtrOpConversion : public ConvertOpToLLVMPattern<tptr::FromPtrOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(tptr::FromPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Get the target memref type
    auto mTy = dyn_cast<MemRefType>(op.getResult().getType());
    if (!mTy)
      return rewriter.notifyMatchFailure(op, "Expected memref result type");

    // NOTE: here is different from the original pattern in
    // mlir/lib/Conversion/PtrToLLVM/PtrToLLVM.cpp. We allow ptr without
    // metadata. In that case, we will assume the metadata is trivial and only
    // use the pointer value.
    bool hasMetadata = static_cast<bool>(op.getMetadata());

    // Convert the result type
    Type descriptorTy = getTypeConverter()->convertType(mTy);
    if (!descriptorTy)
      return rewriter.notifyMatchFailure(op, "Failed to convert result type");

    // Get the strides, offsets and shape.
    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(mTy.getStridesAndOffset(strides, offset))) {
      return rewriter.notifyMatchFailure(
          op, "Failed to get the strides and offset");
    }
    ArrayRef<int64_t> shape = mTy.getShape();

    // Create a new memref descriptor
    Location loc = op.getLoc();
    auto desc = MemRefDescriptor::poison(rewriter, loc, descriptorTy);

    // Set the allocated and aligned pointers.
    desc.setAllocatedPtr(rewriter, loc,
                         hasMetadata
                             ? LLVM::ExtractValueOp::create(
                                   rewriter, loc, adaptor.getMetadata(), 0)
                             : adaptor.getPtr());
    desc.setAlignedPtr(rewriter, loc, adaptor.getPtr());

    // Extract metadata from the passed struct.
    unsigned fieldIdx = 1;

    // Set dynamic offset if needed.
    if (offset == ShapedType::kDynamic) {
      if (hasMetadata) {
        Value offsetValue = LLVM::ExtractValueOp::create(
            rewriter, loc, adaptor.getMetadata(), fieldIdx++);
        desc.setOffset(rewriter, loc, offsetValue);
      }
    } else {
      desc.setConstantOffset(rewriter, loc, offset);
    }

    // Set dynamic sizes if needed.
    for (auto [i, dim] : llvm::enumerate(shape)) {
      if (dim == ShapedType::kDynamic) {
        if (hasMetadata) {
          Value sizeValue = LLVM::ExtractValueOp::create(
              rewriter, loc, adaptor.getMetadata(), fieldIdx++);
          desc.setSize(rewriter, loc, i, sizeValue);
        }
      } else {
        desc.setConstantSize(rewriter, loc, i, dim);
      }
    }

    // Set dynamic strides if needed.
    for (auto [i, stride] : llvm::enumerate(strides)) {
      if (stride == ShapedType::kDynamic) {
        if (hasMetadata) {
          Value strideValue = LLVM::ExtractValueOp::create(
              rewriter, loc, adaptor.getMetadata(), fieldIdx++);
          desc.setStride(rewriter, loc, i, strideValue);
        }
      } else {
        desc.setConstantStride(rewriter, loc, i, stride);
      }
    }

    rewriter.replaceOp(op, static_cast<Value>(desc));
    return success();
  };
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertToLLVMPatternInterface implementation
//===----------------------------------------------------------------------===//

namespace {
/// Implement the interface to convert TPtr to LLVM.
struct TPtrToLLVMDialectInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;
  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  /// Hook for derived dialect interface to provide conversion patterns
  /// and mark dialect legal for the conversion target.
  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &converter,
      RewritePatternSet &patterns) const final {
    tptr::populateTPtrToLLVMConversionPatterns(converter, patterns);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

void mlir::tptr::populateTPtrToLLVMConversionPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  // Add conversion patterns.
  patterns.add<FromPtrOpConversion,
               VectorConvertToLLVMPattern<tptr::IntToPtrOp, LLVM::IntToPtrOp>,
               VectorConvertToLLVMPattern<tptr::PtrToIntOp, LLVM::PtrToIntOp>>(
      converter);
}

void mlir::tptr::registerConvertTPtrToLLVMInterface(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tptr::TPtrDialect *dialect) {
    dialect->addInterfaces<TPtrToLLVMDialectInterface>();
  });
}
