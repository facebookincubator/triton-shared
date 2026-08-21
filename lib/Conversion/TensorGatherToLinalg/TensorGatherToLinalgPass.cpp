//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TensorGatherToLinalg/TensorGatherToLinalg.h"
#include "triton-shared/Conversion/TritonArithToLinalg/ConversionTools.h"
#include "triton-shared/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "tensor-gather-to-linalg"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_TENSORGATHERTOLINALG
#include "triton-shared/Conversion/TensorGatherToLinalg/Passes.h.inc"

namespace {

// GatherConverter (TritonArithToLinalg) builds the axis component of
// `coords` as a bare tensor-wide arith.index_cast, but arith ops on tensors
// are dynamically illegal in that same pass's ConversionTarget, so
// linalg::populateElementwiseToLinalgConversionPatterns -- registered in the
// same pattern set -- always legalizes it, within that same
// applyPartialConversion, into a linalg.generic that scalarizes the cast:
// ins(originalIndices) outs(init) with identity maps, body
// `arith.index_cast` + `linalg.yield`. By the time this pass runs, the axis
// component is therefore guaranteed to already be in this scalarized form,
// never the bare op. Recovers the original (un-cast) indices operand from
// that form, or returns null if the shape doesn't match.
static Value getOriginalIndicesFromAxisComponent(Value axisComponent) {
  auto genericOp = axisComponent.getDefiningOp<linalg::GenericOp>();
  if (!genericOp || genericOp.getInputs().size() != 1 ||
      genericOp.getOutputs().size() != 1)
    return nullptr;

  Block &body = genericOp.getRegion().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yieldOp || yieldOp.getValues().size() != 1)
    return nullptr;

  auto bodyCast = yieldOp.getValues()[0].getDefiningOp<arith::IndexCastOp>();
  if (!bodyCast || bodyCast.getIn() != body.getArgument(0))
    return nullptr;

  return genericOp.getInputs()[0];
}

// Rewrites a tt.gather-derived tensor.gather (identified by
// kTensorGatherFromTtGatherAxisAttrName) back into the single-axis
// linalg.generic + tensor.extract form originally hand-written for
// tt.gather. Safe only because GatherConverter (TritonArithToLinalg) is
// guaranteed to be the sole producer of such ops, always builds `coords` in
// one fixed shape (a chain of `rank` tensor.insert_slice ops, one of which
// carries the real tt.gather indices, scalarized-cast per
// getOriginalIndicesFromAxisComponent above, at the tt.gather axis, the rest
// carrying per-dim iota broadcasts), and this pass runs immediately after
// TritonArithToLinalg with no intervening canonicalization that could
// disturb that shape.
struct TtGatherDerivedTensorGatherConverter
    : public OpRewritePattern<tensor::GatherOp> {
  using OpRewritePattern<tensor::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto axisAttr =
        op->getAttrOfType<IntegerAttr>(kTensorGatherFromTtGatherAxisAttrName);
    if (!axisAttr)
      return rewriter.notifyMatchFailure(
          op, "not a tt.gather-derived tensor.gather (missing marker attr)");
    int64_t axis = axisAttr.getInt();

    Value src = op.getSource();
    Value coords = op.getIndices();
    auto resType = cast<RankedTensorType>(op.getResult().getType());
    int64_t rank = cast<RankedTensorType>(src.getType()).getRank();

    // Walk the exact chain GatherConverter is guaranteed to build: `rank`
    // chained tensor.insert_slice ops over `coords`, one per source dim,
    // each inserting at a static offset (along the trailing coordinate dim)
    // equal to that dim. Find the one at offset == axis and take its source
    // operand.
    Value axisComponent;
    Value cur = coords;
    for (int64_t i = 0; i < rank; ++i) {
      auto insertSlice = cur.getDefiningOp<tensor::InsertSliceOp>();
      if (!insertSlice)
        return rewriter.notifyMatchFailure(op,
                                           "unexpected coords construction");
      auto offset = getConstantIntValue(insertSlice.getMixedOffsets().back());
      if (!offset)
        return rewriter.notifyMatchFailure(op, "non-static coords offset");
      if (*offset == axis)
        axisComponent = insertSlice.getSource();
      cur = insertSlice.getDest();
    }
    if (!axisComponent)
      return rewriter.notifyMatchFailure(op, "could not locate axis component");

    Value originalIndices = getOriginalIndicesFromAxisComponent(axisComponent);
    if (!originalIndices)
      return rewriter.notifyMatchFailure(
          op, "axis component not a scalarized index_cast generic");

    // Rebuild the pre-retarget GatherConverter linalg.generic verbatim: ins
    // = {originalIndices}, outs = {init}, both identity maps; src captured
    // directly (not driven through ins); body casts the scalar index and
    // fills in linalg.index() for every non-axis dim.
    SmallVector<AffineMap> indexingMaps(2,
                                        rewriter.getMultiDimIdentityMap(rank));
    Value init = tensor::EmptyOp::create(
        rewriter, op.getLoc(), resType.getShape(), resType.getElementType());

    auto genericOp = linalg::GenericOp::create(
        rewriter, op.getLoc(), op->getResultTypes(),
        ValueRange{originalIndices}, ValueRange{init}, indexingMaps,
        getNParallelLoopsAttrs(rank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value idxAsIndex = arith::IndexCastOp::create(
              nestedBuilder, nestedLoc, nestedBuilder.getIndexType(),
              blockArgs[0]);

          SmallVector<Value> coordVals;
          coordVals.reserve(rank);
          for (int64_t dim = 0; dim < rank; ++dim) {
            if (dim == axis) {
              coordVals.push_back(idxAsIndex);
            } else {
              coordVals.push_back(
                  linalg::IndexOp::create(nestedBuilder, nestedLoc, dim));
            }
          }

          Value gathered = tensor::ExtractOp::create(nestedBuilder, nestedLoc,
                                                     src, coordVals);
          linalg::YieldOp::create(nestedBuilder, nestedLoc, gathered);
        });

    rewriter.replaceOp(op, genericOp->getResults());
    return success();
  }
};

class TensorGatherToLinalgPass
    : public ::impl::TensorGatherToLinalgBase<TensorGatherToLinalgPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TtGatherDerivedTensorGatherConverter>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTensorGatherToLinalgPass() {
  return std::make_unique<TensorGatherToLinalgPass>();
}
