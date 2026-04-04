//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h" // Required for IR/TPtrOps.h.inc

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TPtr/IR/TPtrOps.cpp.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// FromPtrOp
//===----------------------------------------------------------------------===//

OpFoldResult tptr::FromPtrOp::fold(FoldAdaptor adaptor) {
  // Fold the pattern:
  // %ptr = ptr.to_ptr %v : type -> ptr
  // (%mda = ptr.get_metadata %v : type)?
  // %val = tptr.from_ptr %ptr (metadata %mda)? : ptr -> type
  // To:
  // %val -> %v
  Value ptrLike;
  tptr::FromPtrOp fromPtr = *this;
  while (fromPtr != nullptr) {
    auto toPtr = fromPtr.getPtr().getDefiningOp<ptr::ToPtrOp>();
    // Cannot fold if it's not a `to_ptr` op or the initial and final types are
    // different.
    if (!toPtr || toPtr.getPtr().getType() != fromPtr.getType())
      return ptrLike;
    Value md = fromPtr.getMetadata();
    // If the type has trivial metadata fold.
    if (!fromPtr.getType().hasPtrMetadata()) {
      ptrLike = toPtr.getPtr();
    } else if (md) {
      // Fold if the metadata can be verified to be equal.
      if (auto mdOp = md.getDefiningOp<ptr::GetMetadataOp>();
          mdOp && mdOp.getPtr() == toPtr.getPtr())
        ptrLike = toPtr.getPtr();
    }
    // Check for a sequence of casts.
    fromPtr = ptrLike ? ptrLike.getDefiningOp<tptr::FromPtrOp>() : nullptr;
  }
  return ptrLike;
}

LogicalResult tptr::FromPtrOp::verify() {
  if (isa<ptr::PtrType>(getType()))
    return emitError() << "the result type cannot be `!ptr.ptr`";
  if (getType().getMemorySpace() != getPtr().getType().getMemorySpace()) {
    return emitError()
           << "expected the input and output to have the same memory space";
  }
  return success();
}