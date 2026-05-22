//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"

#include <cstdint>
#include <optional>
#include <utility>

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.h.inc"

using namespace mlir;
using namespace mlir::tts;

namespace mlir {
namespace tts {

namespace utils {

Type getSrcPtrType(Type t) {
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    if (auto ptrType =
            dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      return ptrType;
    }
  }
  if (auto ptrType = dyn_cast<triton::PointerType>(t)) {
    if (auto tensorType =
            dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      return triton::PointerType::get(tensorType.getElementType(), 1);
    }
    return ptrType;
  }
  return triton::PointerType::get(NoneType::get(t.getContext()), 1);
}

bool hasPtrValue(Value v) {
  if (!v) {
    return false;
  }
  if (auto ptrType = dyn_cast<triton::PointerType>(v.getType())) {
    return !isa<NoneType>(ptrType.getPointeeType());
  }
  return false;
}

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
Value getScalarValue(Value operand, Location loc, OpBuilder &builder) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return arith::SIToFPOp::create(builder, loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return arith::TruncFOp::create(builder, loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            builder, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

} // namespace utils

template <typename T, typename = std::enable_if_t<llvm::is_one_of<
                          T, LoadOp, StoreOp, AtomicRMWOp>::value>>
static auto foldMemoryAccessOp(T op) {
  SmallVector<OpFoldResult> mixedMaskDims(op.getMixedMaskDims());

  // No constant operands were folded, just return;
  if (failed(foldDynamicIndexList(mixedMaskDims, /*onlyNonNegative=*/true))) {
    if constexpr (std::is_same_v<T, StoreOp>) {
      return failure();
    } else {
      return OpFoldResult{};
    }
  }

  auto [staticMaskDims, variableMaskDims] = decomposeMixedValues(mixedMaskDims);

  op.setStaticMaskDims(staticMaskDims);
  op.getMaskDimsMutable().assign(variableMaskDims);

  if constexpr (std::is_same_v<T, StoreOp>) {
    return success();
  } else {
    return OpFoldResult{op.getResult()};
  }
}

template <typename T,
          typename = std::enable_if_t<llvm::is_one_of<
              T, MakeTensorPtrOp, MakeGatherScatterTensorPtrOp>::value>>
static OpFoldResult foldMakeTensorPtrOp(T op) {
  SmallVector<OpFoldResult> mixedOffsets(op.getMixedOffsets());
  SmallVector<OpFoldResult> mixedStrides(op.getMixedStrides());
  SmallVector<OpFoldResult> mixedShape;
  if constexpr (std::is_same_v<T, MakeTensorPtrOp>) {
    mixedShape = op.getMixedShape();
  }

  // No constant operands were folded, just return;
  if (failed(foldDynamicIndexList(mixedOffsets, /*onlyNonNegative=*/true)) &&
      failed(foldDynamicIndexList(mixedShape, /*onlyNonNegative=*/true)) &&
      failed(foldDynamicIndexList(mixedStrides))) {
    return OpFoldResult{};
  }

  auto [staticOffsets, variableOffsets] = decomposeMixedValues(mixedOffsets);
  auto [staticStrides, variableStrides] = decomposeMixedValues(mixedStrides);
  auto [staticShape, variableShape] = decomposeMixedValues(mixedShape);

  op.setStaticOffsets(staticOffsets);
  op.setStaticStrides(staticStrides);
  if constexpr (std::is_same_v<T, MakeTensorPtrOp>) {
    op.setStaticShape(staticShape);
  }

  op.getOffsetsMutable().assign(variableOffsets);
  op.getStridesMutable().assign(variableStrides);
  if constexpr (std::is_same_v<T, MakeTensorPtrOp>) {
    op.getShapeMutable().assign(variableShape);
  }

  return op.getResult();
}

OpFoldResult MakeTensorPtrOp::fold(FoldAdaptor) {
  return foldMakeTensorPtrOp(*this);
}

void MakeTensorPtrOp::build(OpBuilder &b, OperationState &state, Value base,
                            ArrayRef<int64_t> sizes,
                            ArrayRef<OpFoldResult> strides,
                            ArrayRef<OpFoldResult> offsets,
                            ArrayRef<OpFoldResult> shape,
                            ArrayRef<int32_t> order) {
  SmallVector<int64_t> staticStrides, staticOffsets, staticShape;
  SmallVector<Value> dynamicStrides, dynamicOffsets, dynamicShape;

  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  dispatchIndexOpFoldResults(shape, dynamicShape, staticShape);

  Type resType;
  auto basePtr = cast<triton::PointerType>(base.getType());
  auto elemType = basePtr.getPointeeType();
  // non-block pointer
  if (order.empty()) {
    resType = RankedTensorType::get(sizes, basePtr);
  }
  // block pointer
  else {
    resType = RankedTensorType::get(
        sizes, triton::PointerType::get(elemType, basePtr.getAddressSpace()));
  }

  build(b, state, resType, base, sizes, dynamicStrides, dynamicOffsets,
        dynamicShape, b.getDenseI64ArrayAttr(staticStrides),
        b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticShape), order);
}

LogicalResult MakeTensorPtrOp::verify() {
  // Get the expected rank from the result type
  int64_t expectedRank = 0;
  Type resultType = getResult().getType();

  if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
    // Tensor of pointers: tensor<!tt.ptr<type>>
    expectedRank = tensorType.getRank();
  } else if (auto ptrType = dyn_cast<triton::PointerType>(resultType)) {
    // Pointer to tensor: !tt.ptr<tensor<type>>
    if (auto tensorType =
            dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      expectedRank = tensorType.getRank();
    } else {
      return emitOpError(
          "result must be either a tensor of pointers or a pointer to tensor");
    }
  } else {
    return emitOpError(
        "result must be either a tensor of pointers or a pointer to tensor");
  }

  if (expectedRank == 0) {
    return emitOpError("result rank must be greater than 0");
  }

  // sizes, strides, and offsets should have length equal to the result rank
  if (getMixedSizes().size() != expectedRank) {
    return emitOpError("sizes length (")
           << getMixedSizes().size() << ") must match result rank ("
           << expectedRank << ")";
  }
  if (getMixedStrides().size() != expectedRank) {
    return emitOpError("strides length (")
           << getMixedStrides().size() << ") must match result rank ("
           << expectedRank << ")";
  }
  if (getMixedOffsets().size() != expectedRank) {
    return emitOpError("offsets length (")
           << getMixedOffsets().size() << ") must match result rank ("
           << expectedRank << ")";
  }
  if (getMixedShape().size() != expectedRank) {
    return emitOpError("shape length (")
           << getMixedShape().size() << ") must match result rank ("
           << expectedRank << ")";
  }

  // If order is non-empty, it must also match the expected rank
  auto order = getOrder();
  if (!order.empty() && order.size() != expectedRank) {
    return emitOpError("order length (")
           << order.size() << ") must match result rank (" << expectedRank
           << ")";
  }

  return success();
}

OpFoldResult MakeGatherScatterTensorPtrOp::fold(FoldAdaptor) {
  return foldMakeTensorPtrOp(*this);
}

void MakeGatherScatterTensorPtrOp::build(OpBuilder &b, OperationState &state,
                                         Value base, Value gatherScatterOffset,
                                         int gatherScatterDim,
                                         ArrayRef<int64_t> sizes,
                                         ArrayRef<OpFoldResult> strides,
                                         ArrayRef<OpFoldResult> offsets) {
  SmallVector<int64_t> staticStrides, staticOffsets;
  SmallVector<Value> dynamicStrides, dynamicOffsets;
  for (auto [i, offset] : llvm::enumerate(offsets)) {
    if (i != gatherScatterDim)
      dispatchIndexOpFoldResult(offset, dynamicOffsets, staticOffsets);
    else
      staticOffsets.push_back(0);
  }
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  Type resType;
  auto basePtr = cast<triton::PointerType>(base.getType());
  auto elemType = basePtr.getPointeeType();

  resType = RankedTensorType::get(
      sizes, triton::PointerType::get(elemType, basePtr.getAddressSpace()));

  build(b, state, resType, base, gatherScatterOffset,
        b.getI32IntegerAttr(gatherScatterDim), b.getDenseI64ArrayAttr(sizes),
        dynamicStrides, dynamicOffsets, b.getDenseI64ArrayAttr(staticStrides),
        b.getDenseI64ArrayAttr(staticOffsets), Value());
}

void MakeGatherScatterTensorPtrOp::build(
    OpBuilder &b, OperationState &state, Value base, Value gatherScatterOffset,
    Value gatherScatterMask, int gatherScatterDim, ArrayRef<int64_t> sizes,
    ArrayRef<OpFoldResult> strides, ArrayRef<OpFoldResult> offsets) {
  SmallVector<int64_t> staticStrides, staticOffsets;
  SmallVector<Value> dynamicStrides, dynamicOffsets;
  for (auto [i, offset] : llvm::enumerate(offsets)) {
    if (i != gatherScatterDim)
      dispatchIndexOpFoldResult(offset, dynamicOffsets, staticOffsets);
    else
      staticOffsets.push_back(0);
  }
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);

  Type resType;
  auto basePtr = cast<triton::PointerType>(base.getType());
  auto elemType = basePtr.getPointeeType();

  if (gatherScatterOffset.getType().isIntOrIndex()) {
    assert(sizes.size() == 1 && sizes[0] == 1 &&
           "gatherScatterOffset should be a scalar for 1D gather/scatter");
    resType = triton::PointerType::get(elemType, basePtr.getAddressSpace());

  } else {
    resType = RankedTensorType::get(
        sizes, triton::PointerType::get(elemType, basePtr.getAddressSpace()));
  }

  build(b, state, resType, base, gatherScatterOffset,
        b.getI32IntegerAttr(gatherScatterDim), b.getDenseI64ArrayAttr(sizes),
        dynamicStrides, dynamicOffsets, b.getDenseI64ArrayAttr(staticStrides),
        b.getDenseI64ArrayAttr(staticOffsets), gatherScatterMask);
}

LogicalResult MakeGatherScatterTensorPtrOp::verify() {
  // Verify that the gatherScatterDim is within the valid range.
  if (getGatherScatterDim() < 0 || getGatherScatterDim() >= getSizes().size()) {
    return emitError("gatherScatterDim is out of bounds");
  }

  // Verify that the sizes, strides, and offsets have compatible dimensions.
  if (getMixedSizes().size() != getMixedStrides().size() ||
      getMixedSizes().size() != getMixedOffsets().size()) {
    return emitError(
        "sizes, strides, and offsets must have the same number of dimensions");
  }

  Type offsetType = getGatherScatterOffset().getType();
  int64_t offsetSize = 0;
  Type offsetEltType = offsetType;
  // Verify that the gatherScatterOffset is a 1D tensor.
  auto rankedTensorType = dyn_cast<RankedTensorType>(offsetType);
  if (!rankedTensorType) {
    return emitError("gatherScatterOffset must be a 1D tensor");
  }
  if (rankedTensorType.getRank() != 1) {
    return emitError("gatherScatterOffset must be a 1D tensor");
  }
  offsetSize = rankedTensorType.getShape()[0];
  offsetEltType = rankedTensorType.getElementType();

  if (!offsetEltType.isIntOrIndex()) {
    return emitError("gatherScatterOffset must be a 1D tensor of "
                     "int or index type");
  }

  // Verify that the gatherScatterMask, if provided, is a 1D tensor.
  if (getGatherScatterMask()) {
    Type maskType = getGatherScatterMask().getType();
    Type maskEltType = maskType;
    auto rankedTensorType = dyn_cast<RankedTensorType>(maskType);
    if (!rankedTensorType) {
      return emitError("gatherScatterMask must be a 1D tensor");
    }
    if (rankedTensorType.getRank() != 1) {
      return emitError("gatherScatterMask must be a 1D tensor of boolean type");
    }
    // Verify that the gatherScatterMask has the same size as the
    // gatherScatterOffset.
    if (rankedTensorType.getShape()[0] != offsetSize) {
      return emitError(
          "gatherScatterMask must have the same size as gatherScatterOffset");
    }
    maskEltType = rankedTensorType.getElementType();
    if (!maskEltType.isInteger(1)) {
      return emitError("gatherScatterMask must be a 1D tensor of boolean type");
    }
  }

  // Verify that when gatherScatterMask is provided, all the user of
  // MakeGatherScatterTensorPtrOp must have mask with size of 0.
  if (getGatherScatterMask()) {
    for (auto user : (*this)->getUsers()) {
      if (auto loadOp = dyn_cast<LoadOp>(user)) {
        if (loadOp.hasMask()) {
          OpFoldResult MaskedSize =
              loadOp.getMixedMaskDims()[getGatherScatterDim()];
          auto intAttr =
              dyn_cast_if_present<IntegerAttr>(dyn_cast<Attribute>(MaskedSize));
          if (!intAttr || intAttr.getInt() != 0) {
            return emitError("tts.load user of tts.make_gather_scatter_tptr "
                             "with gather_scatter_mask must have "
                             "mask size of 0 for gather_scatter_dim");
          }
        } else {
          return emitError("tts.load user of tts.make_gather_scatter_tptr with "
                           "gather_scatter_mask must have "
                           "mask provided");
        }
      } else if (auto storeOp = dyn_cast<StoreOp>(user)) {
        if (storeOp.hasMask()) {
          OpFoldResult MaskedSize =
              storeOp.getMixedMaskDims()[getGatherScatterDim()];
          auto intAttr =
              dyn_cast_if_present<IntegerAttr>(dyn_cast<Attribute>(MaskedSize));
          if (!intAttr || intAttr.getInt() != 0) {
            return emitError("tts.store user of tts.make_gather_scatter_tptr "
                             "with gather_scatter_mask must have "
                             "mask size of 0 for gather_scatter_dim");
          }
        } else {
          return emitError(
              "tts.store user of tts.make_gather_scatter_tptr with "
              "gather_scatter_mask must have "
              "mask provided");
        }
      } else {
        return emitError("tts.make_gather_scatter_tptr can only be used in "
                         "tts.load or tts.store operations");
      }
    }
  }

  return success();
}

OpFoldResult LoadOp::fold(FoldAdaptor) { return foldMemoryAccessOp(*this); }

void LoadOp::build(OpBuilder &b, OperationState &state, Value ptr,
                   ArrayRef<OpFoldResult> dims, Value other) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;

  dispatchIndexOpFoldResults(dims, dynamicDims, staticDims);

  // non-block pointer type
  auto ptrTensorType = dyn_cast<RankedTensorType>(ptr.getType());
  // block pointer type
  auto tensorPtrType = dyn_cast<triton::PointerType>(ptr.getType());

  Type resType;
  if (ptrTensorType) {
    auto ptrType = cast<triton::PointerType>(ptrTensorType.getElementType());
    auto elemType = ptrType.getPointeeType();
    resType = RankedTensorType::get(ptrTensorType.getShape(), elemType);

  } else if (tensorPtrType) {
    auto tensorType = cast<ShapedType>(tensorPtrType.getPointeeType());
    resType = RankedTensorType::get(tensorType.getShape(),
                                    tensorType.getElementType());
  }
  build(b, state, resType, ptr, dynamicDims, b.getDenseI64ArrayAttr(staticDims),
        other);
}

LogicalResult StoreOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return foldMemoryAccessOp(*this);
}

void StoreOp::build(OpBuilder &b, OperationState &state, Value ptr, Value value,
                    ArrayRef<OpFoldResult> dims) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;

  dispatchIndexOpFoldResults(dims, dynamicDims, staticDims);

  build(b, state, ptr, value, dynamicDims, b.getDenseI64ArrayAttr(staticDims));
}

OpFoldResult AtomicRMWOp::fold(FoldAdaptor) {
  return foldMemoryAccessOp(*this);
}

void AtomicRMWOp::build(OpBuilder &b, OperationState &state,
                        triton::RMWOp atomicRmwOp, Value ptr, Value value,
                        ArrayRef<OpFoldResult> dims, triton::MemSemantic sem,
                        triton::MemSyncScope scope) {
  SmallVector<int64_t> staticDims;
  SmallVector<Value> dynamicDims;

  dispatchIndexOpFoldResults(dims, dynamicDims, staticDims);

  Type resType;
  if (auto ptrTensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
    // Non-block pointer type.
    auto ptrType = cast<triton::PointerType>(ptrTensorType.getElementType());
    auto elemType = ptrType.getPointeeType();
    resType = RankedTensorType::get(ptrTensorType.getShape(), elemType);
  } else if (auto tensorPtrType =
                 dyn_cast<triton::PointerType>(ptr.getType())) {
    // Block pointer type.
    auto tensorType = cast<ShapedType>(tensorPtrType.getPointeeType());
    resType = RankedTensorType::get(tensorType.getShape(),
                                    tensorType.getElementType());
  }

  build(b, state, resType, atomicRmwOp, ptr, value, dynamicDims,
        b.getDenseI64ArrayAttr(staticDims), sem, scope);
}

LogicalResult AtomicRMWOp::verify() {
  auto rmwKind = getAtomicRmwOpAttr().getValue();
  auto ptr = getPtr();
  auto val = getVal();

  Type ptrElemType;
  if (auto ptrTensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
    auto ptrType = cast<triton::PointerType>(ptrTensorType.getElementType());
    ptrElemType = ptrType.getPointeeType();
  } else if (auto tensorPtrType =
                 dyn_cast<triton::PointerType>(ptr.getType())) {
    auto tensorType = cast<ShapedType>(tensorPtrType.getPointeeType());
    ptrElemType = tensorType.getElementType();
  } else {
    return emitOpError(
        "ptr must be either a tensor of pointers or a pointer to tensor");
  }

  auto valTensorType = dyn_cast<RankedTensorType>(val.getType());
  if (!valTensorType) {
    return emitOpError("val must be a tensor");
  }

  Type valElemType = valTensorType.getElementType();
  if (ptrElemType != valElemType) {
    return emitOpError("ptr and val must have the same element type, got ")
           << ptrElemType << " and " << valElemType;
  }

  // Triton MIN/MAX are signed integer operations. The frontend emits UMIN/UMAX
  // for unsigned integer min/max and lowers floating point min/max separately.
  if (rmwKind == mlir::triton::RMWOp::MAX ||
      rmwKind == mlir::triton::RMWOp::MIN ||
      rmwKind == mlir::triton::RMWOp::UMAX ||
      rmwKind == mlir::triton::RMWOp::UMIN) {
    if (!ptrElemType.isInteger()) {
      return emitOpError(
                 "MIN/MAX/UMIN/UMAX operations require integer element type, "
                 "got ")
             << ptrElemType;
    }
  }

  return success();
}

void AtomicCASOp::build(OpBuilder &b, OperationState &state,
                        triton::MemSemantic sem, triton::MemSyncScope scope,
                        Value ptr, Value cmp, Value val) {
  Type resType;
  if (auto ptrTensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
    // Non-block pointer type.
    auto ptrType = cast<triton::PointerType>(ptrTensorType.getElementType());
    auto elemType = ptrType.getPointeeType();
    resType = RankedTensorType::get(ptrTensorType.getShape(), elemType);
  } else if (auto tensorPtrType =
                 dyn_cast<triton::PointerType>(ptr.getType())) {
    // Block pointer type.
    auto tensorType = cast<ShapedType>(tensorPtrType.getPointeeType());
    resType = RankedTensorType::get(tensorType.getShape(),
                                    tensorType.getElementType());
  }

  build(b, state, resType, ptr, cmp, val, sem, scope);
}

LogicalResult AtomicCASOp::verify() {
  auto ptr = getPtr();
  auto cmp = getCmp();
  auto val = getVal();

  Type ptrElemType;
  if (auto ptrTensorType = dyn_cast<RankedTensorType>(ptr.getType())) {
    auto ptrType = cast<triton::PointerType>(ptrTensorType.getElementType());
    ptrElemType = ptrType.getPointeeType();
  } else if (auto tensorPtrType =
                 dyn_cast<triton::PointerType>(ptr.getType())) {
    auto tensorType = cast<ShapedType>(tensorPtrType.getPointeeType());
    ptrElemType = tensorType.getElementType();
  } else {
    return emitOpError(
        "ptr must be either a tensor of pointers or a pointer to tensor");
  }

  auto cmpTensorType = dyn_cast<RankedTensorType>(cmp.getType());
  if (!cmpTensorType) {
    return emitOpError("cmp must be a tensor");
  }
  Type cmpElemType = cmpTensorType.getElementType();

  auto valTensorType = dyn_cast<RankedTensorType>(val.getType());
  if (!valTensorType) {
    return emitOpError("val must be a tensor");
  }
  Type valElemType = valTensorType.getElementType();

  if (ptrElemType != cmpElemType) {
    return emitOpError("ptr and cmp must have the same element type, got ")
           << ptrElemType << " and " << cmpElemType;
  }
  if (ptrElemType != valElemType) {
    return emitOpError("ptr and val must have the same element type, got ")
           << ptrElemType << " and " << valElemType;
  }

  return success();
}

LogicalResult GetStructuredStateOp::verify() {
  auto expectedOffsetAndStrideTypes =
      getOffsetAndStrideTypes(getContext(), getInput().getType());

  if (!expectedOffsetAndStrideTypes.has_value()) {
    return emitOpError("invalid input type for get_structured_state");
  }

  auto [expectedOffsetTypes, expectedStrideTypes] =
      *expectedOffsetAndStrideTypes;

  auto offsetTypesMatched =
      expectedOffsetTypes.size() == getOffsets().size() &&
      llvm::equal(expectedOffsetTypes, getOffsets().getTypes());

  auto strideTypesMatched =
      expectedStrideTypes.size() == getStrides().size() &&
      llvm::equal(expectedStrideTypes, getStrides().getTypes());

  auto srcTypeMatched =
      getSrc().getType() == utils::getSrcPtrType(getInput().getType());

  if (!offsetTypesMatched || !strideTypesMatched || !srcTypeMatched) {
    return emitOpError(
        "verification of operation 'tts.get_structured_state' failed");
  }

  return success();
}

void GetStructuredStateOp::build(OpBuilder &b, OperationState &state,
                                 Value val) {
  auto type = val.getType();

  // Builder cannot fail, so we default to empty offset and stride types.
  // The invalid op will be rejected by the verifier later.
  auto [offsetTypes, strideTypes] =
      getOffsetAndStrideTypes(b.getContext(), type)
          .value_or(std::make_pair(SmallVector<Type>{}, SmallVector<Type>{}));

  build(b, state, val.getType(), offsetTypes, strideTypes,
        utils::getSrcPtrType(val.getType()), val);
}

std::optional<std::pair<SmallVector<Type>, SmallVector<Type>>>
GetStructuredStateOp::getOffsetAndStrideTypes(MLIRContext *context, Type type) {
  auto sizes = getOffsetAndStrideSegmentSizes(type);
  if (!sizes.has_value()) {
    return std::nullopt;
  }
  return std::make_pair(
      SmallVector<Type>(sizes->first, IndexType::get(context)),
      SmallVector<Type>(sizes->second, IndexType::get(context)));
}

std::optional<std::pair<int32_t, int32_t>>
GetStructuredStateOp::getOffsetAndStrideSegmentSizes(Type type) {
  int32_t offsetSegmentSize = 0;
  int32_t strideSegmentSize = 0;

  if (auto tensorType = llvm::dyn_cast<RankedTensorType>(type)) {
    if (tensorType.getElementType().isIntOrIndex()) {
      // Tensors of offsets
      // Important note:
      // We only care about tensor of index / int (in addition to pointer type)
      // because only values of int and index type can potentially be part of a
      // pointer arithmetic sequence.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    } else if (auto ptrType =
                   dyn_cast<triton::PointerType>(tensorType.getElementType())) {
      // Unstructured pointers (tensor<!tt.ptr<type>>)
      // Each tensor of rank k gets k values for its offsets and k values for
      // its strides, all of which has Index type.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    }
  }
  // Block pointers (tensor<!tt.ptr<type>> or !tt.ptr<type>)
  else if (auto ptrType = llvm::dyn_cast<triton::PointerType>(type)) {
    if (auto tensorType =
            llvm::dyn_cast<RankedTensorType>(ptrType.getPointeeType())) {
      // Each tensor of rank k gets k values for its offsets and k values for
      // its strides, all of which has Index type.
      offsetSegmentSize = strideSegmentSize = tensorType.getRank();
    } else {
      // The only relevant state that can be updated in loops for scalar
      // pointers are offset. No need to include stride here.
      offsetSegmentSize = 1;
    }
  } else {
    return std::nullopt;
  }

  return std::make_pair(offsetSegmentSize, strideSegmentSize);
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

namespace {

// If there are dimensions with size 1 and stride 0, replace 0 stride with
// the product of sizes of all lower dimensions. This avoids creating
// tts.make_tptr with zero stride for degenerate dimensions.
class CanonicalizeZeroStrides : public OpRewritePattern<MakeTensorPtrOp> {
public:
  using OpRewritePattern<MakeTensorPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MakeTensorPtrOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<OpFoldResult> newStrides;
    int64_t product = 1;
    bool changed = false;
    for (auto [size, stride] :
         llvm::reverse(llvm::zip(op.getSizes(), op.getMixedStrides()))) {
      auto strideIntAttr = getConstantIntValue(stride);
      // Zero stride can be valid when the size is not 1.
      if (size == 1 && strideIntAttr && *strideIntAttr == 0) {
        changed = true;
        newStrides.push_back(rewriter.getIndexAttr(product));
      } else {
        newStrides.push_back(stride);
      }
      product *= size;
    }

    if (!changed) {
      return failure();
    }

    std::reverse(newStrides.begin(), newStrides.end());

    rewriter.replaceOpWithNewOp<MakeTensorPtrOp>(
        op, op.getBase(), op.getSizes(), newStrides, op.getMixedOffsets(),
        op.getMixedShape(), op.getOrder());

    return success();
  }
};

static bool isAllTrueMask(Value mask) {
  if (matchPattern(mask, m_One())) {
    return true;
  }

  if (auto constantOp = mask.getDefiningOp<arith::ConstantOp>()) {
    if (auto boolAttr = dyn_cast<BoolAttr>(constantOp.getValue())) {
      return boolAttr.getValue();
    }
  }

  if (auto splatOp = mask.getDefiningOp<triton::SplatOp>()) {
    return isAllTrueMask(splatOp.getSrc());
  }

  if (auto broadcastOp = mask.getDefiningOp<triton::BroadcastOp>()) {
    return isAllTrueMask(broadcastOp.getSrc());
  }

  return false;
}

class StripAllTrueGatherMask : public OpRewritePattern<GatherOp> {
public:
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp op,
                                PatternRewriter &rewriter) const override {
    Value mask = op.getMask();
    if (!mask || !isAllTrueMask(mask)) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<GatherOp>(op, op.getType(), op.getPtr(),
                                          op.getOffset(), /*mask=*/Value{},
                                          op.getOther());
    return success();
  }
};

} // namespace

void MakeTensorPtrOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<CanonicalizeZeroStrides>(context);
}

void GatherOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<StripAllTrueGatherMask>(context);
}

} // namespace tts
} // namespace mlir
