//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/LinalgToVector/LinalgToVectorUtility.h"
#include "triton-shared/Conversion/LinalgToVector/Passes.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "linalg-to-vector"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_LINALGTOVECTOR
#include "triton-shared/Conversion/LinalgToVector/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class LinalgToVectorPass
    : public triton::impl::LinalgToVectorBase<LinalgToVectorPass> {
  using LinalgToVectorBase<LinalgToVectorPass>::LinalgToVectorBase;

public:
  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto ctx = moduleOp.getContext();

    tts::computeTiling(moduleOp, vectorBits, use1DTiling);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// LinalgToVector internal IR Dump After: computeTiling\n"
          << moduleOp << "\n\n\n";
    }

    {
      RewritePatternSet patterns(ctx);
      tts::populateApplyTilingConversionPatterns(patterns, vectorBits,
                                                 use1DTiling);
      (void)applyPatternsGreedily(moduleOp, std::move(patterns));
      if (dumpIntermediateSteps) {
        llvm::dbgs()
            << "// -----// LinalgToVector internal IR Dump After: applyTiling\n"
            << moduleOp << "\n\n\n";
      }
    }

    // Fuse producers into the tiled op where possible.
    PatternRewriter rewriter(ctx);

    // After fusion, perform canonicalization and simplification passes /
    // patterns.
    {
      RewritePatternSet patterns(ctx);
      // populate canonicalization/cleanup patterns for linalg / tensor /
      // affine, etc.
      linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
      (void)applyPatternsGreedily(moduleOp, std::move(patterns));
    }

    // Vectorize: attempt to vectorize the tiled linalg op.
    // Many MLIR versions provide linalg::vectorize or linalg::vectorizeLinalgOp
    // helpers. If not present, rely on the vectorization patterns invoked via
    // applyPatternsGreedily.
    {
      RewritePatternSet vecPatterns(ctx);
      // If you have a vectorization helper, populate patterns here.
      tts::populateVectorizeConversionPatterns(vecPatterns, vectorizeNDExtract,
                                               flatten1DDepthwiseConv);
      (void)applyPatternsGreedily(moduleOp, std::move(vecPatterns));
    }
  }
};
} // namespace
