#ifndef CONVERSION_LINALGTOVECTOR_LINALGTOVECTORUTILITY_H
#define CONVERSION_LINALGTOVECTOR_LINALGTOVECTORUTILITY_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class ModuleOp;
namespace tts {

// if op has this attribute, skip tiling and vectorization
static const char *kTilingSkipAttrName = "tts.skip";

static const char *kTilingSizeAttrName = "tts.tile_sizes";

static const char *kVectorizeSizeAttrName = "tts.vectorize_sizes";

void computeTiling(ModuleOp moduleOp, uint vectorBits, bool use1DTiling);

void populateApplyTilingConversionPatterns(RewritePatternSet &patterns,
                                           uint vectorBits, bool use1DTiling);

void populateVectorizeConversionPatterns(RewritePatternSet &patterns,
                                         bool vectorizeExtract,
                                         bool flattenConv);

} // namespace tts
} // namespace mlir

#endif