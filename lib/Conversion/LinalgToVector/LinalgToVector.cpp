//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Types.h"

#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Conversion/LinalgToVector/LinalgToVectorUtility.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR//MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/DebugLog.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "linalg-to-vector"

using namespace mlir;

namespace {

/// Compute iteration-space loop ranges for a linalg op.
/// Return vector of length = op.getNumLoops(), each element is either
/// static size or ShapedType::kDynamic if cannot be determined.
static SmallVector<int64_t> computeLoopRanges(linalg::LinalgOp op) {
  constexpr int64_t kDynamic = ShapedType::kDynamic;
  unsigned numLoops = op.getNumLoops();
  SmallVector<int64_t> loopSizes(numLoops, kDynamic);

  // Collect indexing maps and operand shaped types into indexable containers.
  SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();

  SmallVector<ShapedType> shapedOperands;
  shapedOperands.reserve(op->getNumOperands());
  for (Value v : op->getOperands()) {
    if (auto st = dyn_cast<ShapedType>(v.getType()))
      shapedOperands.push_back(st);
    else
      shapedOperands.push_back(ShapedType()); // nullptr-like (rank -1)
  }

  // For each iterator dimension `d`, try to deduce its static size.
  for (unsigned d = 0; d < numLoops; ++d) {
    int64_t deducedSize = kDynamic;
    bool conflict = false;

    // Scan all indexing maps & corresponding operands
    for (auto it : llvm::enumerate(indexingMaps)) {
      AffineMap map = it.value();
      unsigned operandIdx = it.index();

      // Skip if we don't have shaped type for this operand.
      if (operandIdx >= shapedOperands.size())
        continue;
      ShapedType shape = shapedOperands[operandIdx];
      if (!shape)
        continue; // non-shaped operand; skip

      // Iterate over map results: the result position indicates tensor dim.
      for (unsigned resPos = 0; resPos < map.getNumResults(); ++resPos) {
        AffineExpr expr = map.getResult(resPos);

        // We're only interested in pure AffineDimExpr that directly uses
        // iterator d.
        if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
          if (static_cast<unsigned>(dimExpr.getPosition()) == d) {
            // This iterator 'd' maps to operandIdx's dim 'resPos'.
            // Get that tensor dim size (may be dynamic).
            if (resPos >= shape.getRank())
              continue; // defensive, shouldn't usually happen

            int64_t thisSize = shape.getDimSize(resPos);

            // Merge logic:
            // - if deducedSize is dynamic, accept this static size (if static)
            // - if both static and equal, keep
            // - if static and differ, mark conflict -> dynamic
            if (deducedSize == kDynamic) {
              deducedSize = thisSize;
            } else {
              // both static?
              if (deducedSize != thisSize) {
                conflict = true;
                break;
              }
            }
          }
        } // else: skip non-dim expr (e.g. constant or symbol)
      } // for each result expr

      if (conflict)
        break;
    } // for each indexing map

    if (conflict)
      loopSizes[d] = kDynamic;
    else
      loopSizes[d] = deducedSize;
  } // for each iterator d

  return loopSizes;
}

/// Recursively extract element type and return its bitwidth.
/// Returns 0 if the type is unsupported.
unsigned getElementBitwidth(mlir::Type type) {
  if (auto tensorTy = llvm::dyn_cast<TensorType>(type))
    return getElementBitwidth(tensorTy.getElementType());
  if (auto memTy = llvm::dyn_cast<MemRefType>(type))
    return getElementBitwidth(memTy.getElementType());
  if (auto vecTy = llvm::dyn_cast<VectorType>(type))
    return getElementBitwidth(vecTy.getElementType());
  if (auto intTy = llvm::dyn_cast<IntegerType>(type))
    return intTy.getWidth();
  if (auto fTy = llvm::dyn_cast<FloatType>(type))
    return fTy.getWidth();
  if (type.isIndex())
    return 64;

  // Unsupported type
  return 0;
}

unsigned getValueBitwidth(mlir::Value v) {
  return getElementBitwidth(v.getType());
}

/// Compute the smallest power-of-2 greater than or equal to x.
/// If x is 0, returns 1.
/// This is equivalent to "ceil to next power of two".
inline uint64_t alignUpToNextPow2(uint64_t x) {
  if (x == 0)
    return 1;

  // Decrement x to handle the case where x is already a power of 2
  x--;

  // Propagate the highest set bit to all lower bits
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32; // For 64-bit integers

  // Add 1 to get the next power of 2
  return x + 1;
}

void computeTilingHelper(linalg::LinalgOp op, uint vectorBits,
                         bool use1DTiling) {
  // Compute loop ranges.
  SmallVector<int64_t> loopRanges = computeLoopRanges(op);
  if (loopRanges.empty())
    return;
  unsigned int maxBitwidth = 0;
  bool valid = true;
  op->walk([&maxBitwidth, &valid](Operation *nestedOp) {
    for (Value v : nestedOp->getOperands()) {
      auto bitwidth = getValueBitwidth(v);
      if (bitwidth == 0)
        valid = false;
      maxBitwidth = std::max(maxBitwidth, bitwidth);
    }
  });
  if (!valid || maxBitwidth == 0)
    return;
  int64_t adjustedVectorSize = vectorBits / maxBitwidth;
  // Determine tile sizes (in elements) for each loop.
  SmallVector<int64_t> tileSizes(loopRanges.size());
  // if shape is 1, assign tile size 0 (no tiling)
  for (int64_t i = loopRanges.size() - 1; i >= 0; --i) {
    auto loopRange = loopRanges[i];
    if (loopRange == 1) {
      tileSizes[i] = 0;
      continue;
    }
    if (adjustedVectorSize <= 1) {
      // no more vector size to assign
      tileSizes[i] = 1;
      continue;
    }
    // if using 1D tiling or shape is dynamic, just assign the remaining
    // adjustedVectorSize. else, assign min(next pow2 of loopRange,
    // adjustedVectorSize)
    int64_t tileSize =
        (use1DTiling || (loopRange == ShapedType::kDynamic))
            ? adjustedVectorSize
            : std::min(static_cast<int64_t>(alignUpToNextPow2(loopRange)),
                       adjustedVectorSize);
    adjustedVectorSize /= tileSize;
    tileSizes[i] = tileSize;
  }

  // Store tile sizes as an attribute on the op.
  auto tileSizeAttr = DenseIntElementsAttr::get(
      VectorType::get({static_cast<int64_t>(tileSizes.size())},
                      IntegerType::get(op.getContext(), 64)),
      tileSizes);
  op->setAttr(tts::kTilingSizeAttrName, tileSizeAttr);
}

struct TileByAttrPattern : public RewritePattern {
  explicit TileByAttrPattern(MLIRContext *context, uint vectorBits,
                             bool use1DTiling)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        vectorBits(vectorBits), use1DTiling(use1DTiling) {}

  LogicalResult postProcess(Operation *op, PatternRewriter &rewriter,
                            FailureOr<scf::SCFTilingResult> maybeTilingResult,
                            SmallVector<int64_t> &tileSizes) const {
    if (failed(maybeTilingResult)) {
      return rewriter.notifyMatchFailure(op, "tiling failed");
    }

    // set VectorizeSizeAttr
    SmallVector<int64_t> vectorizeSizes = tileSizes;
    for (auto &v : vectorizeSizes) {
      if (v == 0) {
        v = 1;
      }
    }
    auto vectorizeSizeAttr = DenseIntElementsAttr::get(
        VectorType::get({static_cast<int64_t>(vectorizeSizes.size())},
                        IntegerType::get(op->getContext(), 64)),
        vectorizeSizes);
    for (auto v : maybeTilingResult->initialValues) {
      if (auto defOp = v.getDefiningOp()) {
        if (dyn_cast<linalg::LinalgOp>(defOp)) {
          defOp->setAttr(tts::kVectorizeSizeAttrName, vectorizeSizeAttr);
        }
      }
    }
    for (auto op : maybeTilingResult->tiledOps) {
      if (dyn_cast<linalg::LinalgOp>(op)) {
        op->setAttr(tts::kVectorizeSizeAttrName, vectorizeSizeAttr);
      }
    }
    for (auto op : maybeTilingResult->mergeOps) {
      if (dyn_cast<linalg::LinalgOp>(op)) {
        op->setAttr(tts::kVectorizeSizeAttrName, vectorizeSizeAttr);
      }
    }
    rewriter.replaceOp(op, maybeTilingResult->replacements);
    return success();
  }

  SmallVector<int64_t>
  computeInterchangeForSinkReduction(linalg::LinalgOp linalgOp) const {
    SmallVector<int64_t> parallelDims;
    SmallVector<int64_t> reductionDims;

    auto iteratorTypes = linalgOp.getIteratorTypesArray();

    for (auto it : llvm::enumerate(iteratorTypes)) {
      if (it.value() == utils::IteratorType::parallel)
        parallelDims.push_back(it.index());
      else
        reductionDims.push_back(it.index());
    }

    SmallVector<int64_t> interchange;
    interchange.append(parallelDims.begin(), parallelDims.end());
    interchange.append(reductionDims.begin(), reductionDims.end());
    return interchange;
  }

  LogicalResult
  doTilingReduction(Operation *op, PatternRewriter &rewriter,
                    TilingInterface iface, SmallVector<int64_t> &tileSizes,
                    SmallVector<utils::IteratorType> &iteratorTypes) const {
    auto partialReductionOp = dyn_cast<PartialReductionOpInterface>(op);
    if (!partialReductionOp) {
      return rewriter.notifyMatchFailure(
          op, "Operation should implement PartialReductionOpInterface");
    }

    if (iteratorTypes.back() != utils::IteratorType::reduction) {
      // in this case, we can directly tile reduction loops without sinking
      // parallel loops. after that, we can vectorize the last
      // iterator(parallel).
      SmallVector<int64_t> tileSizesForReuction;
      for (auto [it, tileSize] : llvm::zip(iteratorTypes, tileSizes)) {
        tileSizesForReuction.push_back(
            (it == utils::IteratorType::reduction && tileSize) ? 1 : 0);
      }
      FailureOr<scf::SCFTilingResult> maybeTilingResult =
          scf::tileReductionUsingScf(
              rewriter, partialReductionOp,
              getAsIndexOpFoldResult(rewriter.getContext(),
                                     tileSizesForReuction));
      if (failed(maybeTilingResult)) {
        return rewriter.notifyMatchFailure(op,
                                           "Reduction tiling Reduction failed");
      }
      // recompute tiling size for new op
      computeTilingHelper(
          dyn_cast<linalg::LinalgOp>(maybeTilingResult->tiledOps.back()),
          vectorBits,
          /*use1DTiling=*/use1DTiling);

      rewriter.replaceOp(op, maybeTilingResult->replacements);
      return success();
    }

    // first tile parallel loops
    SmallVector<int64_t> tileSizesForParallel;
    for (auto [it, tileSize] : llvm::zip(iteratorTypes, tileSizes)) {
      tileSizesForParallel.push_back(
          (it == utils::IteratorType::parallel && tileSize) ? 1 : 0);
    }
    if (!llvm::all_equal(tileSizesForParallel)) {
      // Apply tiling using the tile sizes.
      scf::SCFTilingOptions opts;
      opts.setInterchange(
          computeInterchangeForSinkReduction(dyn_cast<linalg::LinalgOp>(op)));
      opts.setTileSizes(
          getAsIndexOpFoldResult(rewriter.getContext(), tileSizesForParallel));
      FailureOr<scf::SCFTilingResult> maybeTilingResult =
          scf::tileUsingSCF(rewriter, iface, opts);
      if (failed(maybeTilingResult)) {
        return rewriter.notifyMatchFailure(op,
                                           "Reduction tiling parallel failed");
      }
      // recompute tiling size for new op
      computeTilingHelper(
          dyn_cast<linalg::LinalgOp>(maybeTilingResult->tiledOps.back()),
          vectorBits,
          /*use1DTiling=*/use1DTiling);

      rewriter.replaceOp(op, maybeTilingResult->replacements);
      return success();
    }

    // Now tile the reduction loops.
    FailureOr<scf::SCFTilingResult> result = scf::tileReductionUsingScf(
        rewriter, partialReductionOp,
        getAsIndexOpFoldResult(rewriter.getContext(), tileSizes));
    return postProcess(op, rewriter, result, tileSizes);
  }

  bool
  isReductionTiling(SmallVector<int64_t> &tileSizes,
                    SmallVector<utils::IteratorType> &iteratorTypes) const {
    bool contianReduction = llvm::any_of(iteratorTypes, [](auto t) {
      return t == utils::IteratorType::reduction;
    });
    if (!contianReduction) {
      return false;
    }
    if (iteratorTypes.back() == utils::IteratorType::reduction) {
      return true;
    }
    for (auto [it, tileSize] : llvm::zip(iteratorTypes, tileSizes)) {
      if (it == utils::IteratorType::reduction && tileSize > 1) {
        return true;
      }
    }
    return false;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto linalgOp = dyn_cast_or_null<linalg::LinalgOp>(op);
    if (!linalgOp)
      return failure();
    auto tileSizeAttr =
        op->getAttrOfType<DenseIntElementsAttr>(tts::kTilingSizeAttrName);
    if (!tileSizeAttr)
      return failure();
    auto iface = dyn_cast_or_null<TilingInterface>(op);
    if (!iface)
      return failure();

    // Convert attribute to vector of int64_t.
    SmallVector<int64_t> tileSizes;
    for (auto v : tileSizeAttr.getValues<int64_t>()) {
      tileSizes.push_back(v);
    }
    op->removeAttr(tts::kTilingSizeAttrName);

    // due to scf::tileUsingSCF cannot handle tiling for both parallel loops and
    // reduction loops at the same time. so we tile parallel first, then tile
    // reduction.
    SmallVector<utils::IteratorType> iteratorTypes =
        linalgOp.getIteratorTypesArray();
    if (isReductionTiling(tileSizes, iteratorTypes)) {
      return doTilingReduction(op, rewriter, iface, tileSizes, iteratorTypes);
    }

    // Apply tiling using the tile sizes.
    scf::SCFTilingOptions opts;
    opts.setTileSizes(getAsIndexOpFoldResult(rewriter.getContext(), tileSizes));
    // tileReductionUsingScf
    FailureOr<scf::SCFTilingResult> maybeTilingResult =
        scf::tileUsingSCF(rewriter, iface, opts);
    return postProcess(op, rewriter, maybeTilingResult, tileSizes);
  }

private:
  uint vectorBits;
  bool use1DTiling;
};

struct VectorizationPattern : public RewritePattern {
  explicit VectorizationPattern(MLIRContext *context, bool vectorizeExtract,
                                bool flattenConv)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context),
        vectorizeNDExtract(vectorizeExtract),
        flatten1DDepthwiseConv(flattenConv) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto tileSizeAttr =
        op->getAttrOfType<DenseIntElementsAttr>(tts::kVectorizeSizeAttrName);
    if (!tileSizeAttr)
      return failure();

    if (!linalg::hasVectorizationImpl(op))
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported Op, cannot vectorize");
    // Convert attribute to vector of int64_t.
    SmallVector<int64_t> vectorizeSizes;
    for (auto val : tileSizeAttr.getValues<APInt>()) {
      vectorizeSizes.push_back(val.getSExtValue());
    }

    SmallVector<bool, 8> inputScalableVecDims(vectorizeSizes.size(), false);
    FailureOr<linalg::VectorizationResult> vectorResults =
        linalg::vectorize(rewriter, op, /*inputVectorSizes=*/vectorizeSizes,
                          /*inputScalableVecDims=*/inputScalableVecDims,
                          vectorizeNDExtract, flatten1DDepthwiseConv);
    if (failed(vectorResults)) {
      return rewriter.notifyMatchFailure(op, "vectorize failed");
    }
    rewriter.replaceOp(op, vectorResults->replacements);
    return success();
  }

private:
  /// Controls whether to vectorize `tensor.extract` when the input tensor is
  /// rank >= 2.
  bool vectorizeNDExtract = true;
  /// Controls whether to "flatten" the channel dimension when vectorising 1D
  /// depthwise convolutions. This should lead to bette vectorization for
  /// tensors with a low number of channel dimensions.
  bool flatten1DDepthwiseConv = false;
};

struct ScalarizeSingleElementTransferRead
    : OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    auto vecTy = op.getVectorType();
    if (!vecTy || vecTy.getNumElements() != 1)
      return failure();

    if (llvm::all_of(op->getUsers(), [](Operation *user) {
          return isa<vector::ExtractOp>(user);
        }))
      return failure();

    // Create a scalar extract.
    SmallVector<int64_t, 8> scalarIndices(vecTy.getRank(), 0);
    Value scalar = vector::ExtractOp::create(rewriter, op.getLoc(),
                                             op.getResult(), scalarIndices);
    Value newVec =
        vector::FromElementsOp::create(rewriter, op.getLoc(), vecTy, scalar);
    rewriter.replaceAllUsesExcept(op, newVec, scalar.getDefiningOp());
    return success();
  }
};

} // namespace

void mlir::tts::computeTiling(ModuleOp moduleOp, uint vectorBits,
                              bool use1DTiling) {
  moduleOp.walk([&](linalg::LinalgOp op) -> void {
    // If has skip attribute, do nothing
    if (op->hasAttr(tts::kTilingSkipAttrName))
      return;
    computeTilingHelper(op, vectorBits, use1DTiling);
  });
}

void mlir::tts::populateApplyTilingConversionPatterns(
    RewritePatternSet &patterns, uint vectorBits, bool use1DTiling) {
  patterns.add<TileByAttrPattern>(patterns.getContext(), vectorBits,
                                  use1DTiling);
  // linalg::ControlDropUnitDims options;
  // linalg::populateFoldUnitExtentDimsPatterns(patterns, options);
}

void mlir::tts::populateVectorizeConversionPatterns(RewritePatternSet &patterns,
                                                    bool vectorizeExtract,
                                                    bool flattenConv) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<VectorizationPattern>(ctx, vectorizeExtract, flattenConv);
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);

  vector::populateSinkVectorOpsPatterns(patterns);

  vector::populateDropInnerMostUnitDimsXferOpPatterns(patterns);
  vector::populateVectorTransferDropUnitDimsPatterns(patterns);
  vector::populateDropUnitDimWithShapeCastPatterns(patterns);
  vector::populateSinkVectorMemOpsPatterns(patterns);
  vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
  vector::populateFlattenVectorTransferPatterns(patterns);
  vector::populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
  vector::MaskOp::getCanonicalizationPatterns(patterns, ctx);
  vector::populateScalarVectorTransferLoweringPatterns(
      patterns, /*benefit=*/1,
      /*allowMultipleUses=*/true);

  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
               linalg::LinalgCopyVTWForwardingPattern,
               ScalarizeSingleElementTransferRead>(ctx,
                                                   /*benefit=*/2);
  vector::TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  vector::TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);

  // patterns.add<linalg::CopyVectorizationPattern>(ctx);

  // foldArithExtensionPatterns
  vector::populateFoldArithExtensionPatterns(patterns);

  linalg::populatePadOpVectorizationPatterns(patterns);
  // This creates an alternative path for lowering tensor.pad - by
  // decomposing it into e.g. linalg.fill.
  linalg::populateDecomposePadPatterns(patterns);
}
