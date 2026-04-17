//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TritonToLinalgExperimental/CollapseShape.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TritonArithToLinalg/ConversionTools.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "collapse-shape"

using namespace mlir;
using namespace triton;

#define GEN_PASS_DEF_COLLAPSESHAPE
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"

namespace {

static SmallVector<ReassociationIndices>
makeLinearReassociation(unsigned rank) {
  SmallVector<ReassociationIndices> reassoc;
  reassoc.emplace_back();
  for (int64_t i = 0; i < (int64_t)rank; ++i)
    reassoc.back().push_back(i);
  return reassoc;
}

// This pattern collapses a `linalg.fill` operation that fills a tensor with a
// single value into a `tensor.expand_shape` operation that expands a tensor
// filled with the same value to a larger shape. This is useful for optimizing
// the performance of tensor operations that involve broadcasting or filling
// large tensors with a single value.
// //
// for example:
// linalg.fill
// before
// ```
//     %13 = linalg.fill ins(%c1_i32 : i32) outs(%12 :
//     tensor<1x1x1x1x1x2x1xi32>) -> tensor<1x1x1x1x1x2x1xi32>
// ```
// after
// ```
//     %17 = tensor.collapse_shape %12 ...
//     %18 = linalg.fill ins(%c1_i32 : i32) outs(%17 : tensor<2xi32>) ->
//     tensor<2xi32> %expanded_6 = tensor.expand_shape %18 [[0, 1, 2, 3, 4, 5,
//     6]] output_shape [1, 1, 1, 1, 1, 2, 1] : tensor<2xi32> into
//     tensor<1x1x1x1x1x2x1xi32>
// ```
struct CollapseFill : public OpRewritePattern<linalg::FillOp> {
  CollapseFill(MLIRContext *context)
      : OpRewritePattern<linalg::FillOp>(context) {}

  LogicalResult collapseMemRef(linalg::FillOp op,
                               PatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto result = op.getOutputs()[0];
    auto resultType = mlir::dyn_cast_or_null<MemRefType>(result.getType());
    if (!resultType)
      return failure();
    auto rank = resultType.getRank();
    if (rank <= 1 ||
        dyn_cast_or_null<StridedLayoutAttr>(resultType.getLayout())) {
      return failure();
    }
    auto reassociationMap = makeLinearReassociation(rank);
    auto elementType = resultType.getElementType();

    auto output = memref::CollapseShapeOp::create(
        rewriter, loc,
        MemRefType::get(llvm::ArrayRef<int64_t>{resultType.getNumElements()},
                        elementType),
        result, reassociationMap);
    op.getOutputsMutable()[0].set(output.getResult());
    return success();
  }

  LogicalResult collapseTensor(linalg::FillOp op,
                               PatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto result = op.getResult(0);
    auto resultType =
        mlir::dyn_cast_or_null<RankedTensorType>(result.getType());
    if (!resultType)
      return failure();
    auto rank = resultType.getRank();
    if (rank <= 1) {
      return failure();
    }
    auto elementType = resultType.getElementType();

    auto reassociationMap = makeLinearReassociation(rank);
    auto init = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get({resultType.getNumElements()}, elementType),
        op.getOutputs()[0], reassociationMap);
    auto fillOp =
        linalg::FillOp::create(rewriter, loc, op.getInputs(), ValueRange{init});

    auto expandOp = tensor::ExpandShapeOp::create(
        rewriter, loc, result.getType(), fillOp.getResult(0), reassociationMap);

    rewriter.replaceOp(op, expandOp.getResult());
    return success();
  }

  LogicalResult matchAndRewrite(linalg::FillOp op,
                                PatternRewriter &rewriter) const override {
    assert(op->getNumResults() <= 1 && "code assumes single result!");
    if (op->getNumResults() == 1) {
      return collapseTensor(op, rewriter);
    } else if (op->getNumResults() == 0) {
      return collapseMemRef(op, rewriter);
    }
    return failure();
  }
};

// This pattern collapses a `linalg.transpose` operation that transposes a
// tensor into a `tensor.collapse_shape` operation that collapses the tensor
// to a smaller shape. This is useful for optimizing the performance of tensor
// operations that involve transposing large tensors.
// //
// for example:
// linalg.transpose
// before
// ```
//     %transposed = linalg.transpose ins(%expanded_2 :
//     tensor<2x2x2x2x2x2x2x2x2x2xi64>) outs(%66 :
//     tensor<2x2x2x2x2x2x2x2x2x2xi64>) permutation = [0, 1, 2, 3, 4, 5, 6, 7,
//     9, 8]
// ```
// after
// ```
//     %collapsed = tensor.collapse_shape %expanded_17 [[0, 1, 2, 3, 4, 5, 6,
//     7], [8], [9]] : tensor<2x2x2x2x2x2x2x2x2x2xi64> into tensor<256x2x2xi64>
//     %77 = tensor.collapse_shape %66 ...:
//     %transposed = linalg.transpose ins(%collapsed : tensor<256x2x2xi64>)
//     outs(%77 : tensor<256x2x2xi64>) permutation = [0, 2, 1] %expanded_22 =
//     tensor.expand_shape  %transposed ...
// ```

struct CollapseTranspose : public OpRewritePattern<linalg::TransposeOp> {
  CollapseTranspose(MLIRContext *context)
      : OpRewritePattern<linalg::TransposeOp>(context) {}
  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getInput();
    auto sourceType = dyn_cast_or_null<RankedTensorType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for source");
    }
    auto sourceRank = sourceType.getRank();
    auto elementType = sourceType.getElementType();
    if (sourceRank <= 3) {
      return rewriter.notifyMatchFailure(
          op, "expected source rank > 3 for transpose collapse");
    }

    SmallVector<int64_t> perm(op.getPermutation());
    SmallVector<int64_t> transposedShape(sourceRank);
    SmallVector<ReassociationIndices> reassociationMap;
    // from {1,1,1,2,2,1,1} to {1,4,1}
    SmallVector<int64_t> collapseShapeInput;
    int dim = 0;
    SmallVector<int64_t> permIdx(sourceRank);
    for (size_t i = 0; i < sourceRank; ++i) {
      permIdx[perm[i]] = i;
    }
    // The original dim corresponds to the dim after collapse
    SmallVector<int64_t> mapDim;
    for (size_t i = 0; i < sourceRank; ++i) {
      auto id = permIdx[i];
      if (i > 0 && (id == 0 || !(perm[id] == perm[id - 1] + 1))) {
        dim++;
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(i);
      mapDim.push_back(dim);
      collapseShapeInput[dim] *= sourceType.getDimSize(i);
    }
    if (collapseShapeInput.size() == sourceRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse broadcast shape");
    }

    SmallVector<int64_t> newPerm;
    for (size_t i = 0; i < sourceRank; ++i) {
      if (i > 0 && newPerm.back() == mapDim[perm[i]]) {
        continue;
      }
      newPerm.push_back(mapDim[perm[i]]);
    }
    perm = newPerm;
    // update transposedShape, based on perm
    transposedShape.clear();
    for (size_t i = 0; i < perm.size(); ++i) {
      transposedShape.push_back(collapseShapeInput[perm[i]]);
    }

    auto loc = op.getLoc();
    sourceType = RankedTensorType::get(collapseShapeInput, elementType);
    source = tensor::CollapseShapeOp::create(rewriter, loc, sourceType, source,
                                             reassociationMap);

    SmallVector<ReassociationIndices> reassociationMapRe(
        reassociationMap.size());
    int idx = 0;
    for (size_t i = 0; i < reassociationMap.size(); ++i) {
      for (size_t j = 0; j < reassociationMap[perm[i]].size(); ++j) {
        reassociationMapRe[i].push_back(idx++);
      }
    }

    Value transposeInit = tensor::CollapseShapeOp::create(
        rewriter, loc, RankedTensorType::get(transposedShape, elementType),
        op.getInit(), reassociationMapRe);

    Value transpose =
        linalg::TransposeOp::create(rewriter, loc, source, transposeInit, perm)
            .getResults()[0];

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResultTypes()[0], transpose, reassociationMapRe);
    return success();
  }
};

// This pattern collapses a `linalg.generic` operation that broadcasts a tensor
// to a larger shape into a `tensor.expand_shape` operation that expands the
// tensor to the desired shape. This is useful for optimizing the performance
// //
// for example:
// linalg.generic with broadcast
// before
// ```
//     %79 = linalg.generic {indexing_maps = [#map7, #map4], iterator_types =
//     ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel",
//     "parallel", "parallel", "parallel", "parallel"]} ins(%76 :
//     tensor<1x1x1x1x1x1x1x1x2x2xi32>) outs(%77 :
//     tensor<2x2x2x2x2x2x2x2x2x2xi32>) attrs =  {broadcastDims = array<i64: 0,
//     1, 2, 3, 4, 5, 6, 7>} { ^bb0(%in: i32, %out: i32):
//       linalg.yield %in : i32
//     } -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
// ```
// after
// ```
//     %collapsed_30 = tensor.collapse_shape %89 [[0, 1, 2, 3, 4, 5, 6, 7], [8,
//     9]] : tensor<1x1x1x1x1x1x1x1x2x2xi32> into tensor<1x4xi32> %92 =
//     tensor.collapse_shape %77 [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]  :
//     tensor<2x2x2x2x2x2x2x2x2x2xi32> into tensor<256x4xi32> %93 =
//     linalg.generic {indexing_maps = [#map2, #map1], iterator_types =
//     ["parallel", "parallel"]} ins(%collapsed_30 : tensor<1x4xi32>) outs(%92 :
//     tensor<256x4xi32>) attrs =  {broadcastDims = array<i64: 0>} { ^bb0(%in:
//     i32, %out: i32):
//       linalg.yield %in : i32
//     } -> tensor<256x4xi32>
//     %expanded_31 = tensor.expand_shape %93 [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9]]
//     output_shape [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] : tensor<256x4xi32> into
//     tensor<2x2x2x2x2x2x2x2x2x2xi32>
// ```
struct CollapseBroadCast : public OpRewritePattern<linalg::GenericOp> {
  CollapseBroadCast(MLIRContext *context)
      : OpRewritePattern<linalg::GenericOp>(context) {}
  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("broadcastDims")) {
      return rewriter.notifyMatchFailure(op,
                                         "expected broadcastDims attribute");
    }
    assert(op->getNumResults() == 1 && "code assumes single result!");
    auto input = op.getInputs()[0];
    auto sourceType = dyn_cast_or_null<RankedTensorType>(input.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for source");
    }
    auto sourceRank = sourceType.getRank();
    if (sourceRank <= 1) {
      return rewriter.notifyMatchFailure(
          op, "expected source rank > 1 for broadcast collapse");
    }

    auto resultType =
        dyn_cast_or_null<RankedTensorType>(op.getResultTypes()[0]);
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for result");
    auto elementType = resultType.getElementType();
    // collapse input tensor from {1,1,1,2,2,1,1} to {1,4,1}
    SmallVector<int64_t, 8> collapseShapeInput;
    SmallVector<int64_t, 8> collapseShapeOutput;
    SmallVector<ReassociationIndices, 8> reassociationMap;
    int dim = 0;
    for (size_t i = 0; i < sourceRank; i++) {
      if (i > 0 && !((sourceType.getDimSize(i) == 1 &&
                      sourceType.getDimSize(i - 1) == 1) ||
                     (sourceType.getDimSize(i) != 1 &&
                      sourceType.getDimSize(i - 1) != 1))) {
        dim++;
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        collapseShapeOutput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(i);
      collapseShapeInput[dim] *= sourceType.getDimSize(i);
      collapseShapeOutput[dim] *= resultType.getDimSize(i);
    }
    if (collapseShapeInput.size() == sourceRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse broadcast shape");
    }

    auto loc = op.getLoc();
    sourceType = RankedTensorType::get(collapseShapeInput, elementType);
    input = tensor::CollapseShapeOp::create(rewriter, loc, sourceType, input,
                                            reassociationMap);
    resultType = RankedTensorType::get(collapseShapeOutput, elementType);
    size_t resultRank = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(op->getNumOperands() + op->getNumResults());
    indexingMaps.push_back(getBroadcastAffineMap(
        op->getContext(), sourceType.getShape(), resultType.getShape()));
    indexingMaps.append(op->getNumResults(),
                        rewriter.getMultiDimIdentityMap(resultRank));

    assert(op->getNumResults() == 1 && "code assumes single result!");

    auto init = tensor::CollapseShapeOp::create(
        rewriter, loc,
        RankedTensorType::get(resultType.getShape(), elementType),
        op.getOutputs()[0], reassociationMap);

    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, init->getResultTypes(), ValueRange{input},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(resultRank));
    rewriter.cloneRegionBefore(op.getRegion(), linalgOp.getRegion(),
                               linalgOp.getRegion().begin());
    linalgOp->setAttr("broadcastDims",
                      rewriter.getDenseI64ArrayAttr(
                          getBroadcastDims(sourceType, resultType)));
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResultTypes()[0], linalgOp->getResult(0), reassociationMap);
    return success();
  }
};

// This pattern collapses a `linalg.reduce` operation that reduces a tensor
// to a smaller shape by summing over specified dimensions into a
// `tensor.expand_shape` operation that expands the reduced tensor to the
// desired shape. This is useful for optimizing the performance of tensor
// operations that involve reductions.
// //
// for example:
// linalg.reduce
// before
// ```
//     %reduced = linalg.reduce ins(%transposed :
//     tensor<2x2x2x2x2x2x2x2x2x2xi64>) outs(%68 :
//     tensor<2x2x2x2x2x2x2x2x2xi64>) dimensions = [8]
//       (%in: i64, %init: i64) {
//         %311 = arith.xori %in, %init : i64
//         linalg.yield %311 : i64
//       }
// ```
// after
// ```
//     %collapsed_20 = tensor.collapse_shape %expanded_19 [[0, 1, 2, 3, 4, 5, 6,
//     7], [8]] : tensor<2x2x2x2x2x2x2x2x2xi64> into tensor<256x2xi64> %reduced
//     = linalg.reduce ins(%transposed : tensor<256x2x2xi64>) outs(%collapsed_20
//     : tensor<256x2xi64>) dimensions = [1]
//       (%in: i64, %init: i64) {
//         %377 = arith.xori %in, %init : i64
//         linalg.yield %377 : i64
//       }
//     %expanded_21 = tensor.expand_shape %reduced [[0, 1, 2, 3, 4, 5, 6, 7],
//     [8]] output_shape [2, 2, 2, 2, 2, 2, 2, 2, 2] : tensor<256x2xi64> into
//     tensor<2x2x2x2x2x2x2x2x2xi64>
// ```
struct CollapseReduce : public OpRewritePattern<linalg::ReduceOp> {
  CollapseReduce(MLIRContext *context)
      : OpRewritePattern<linalg::ReduceOp>(context) {}
  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInputs()[0];
    auto inputType = dyn_cast_or_null<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor type for input");
    }
    auto inputRank = inputType.getRank();
    auto dims = op.getDimensions();
    if (inputRank - dims.size() <= 1) {
      return rewriter.notifyMatchFailure(
          op, "expected input rank - reduction loops > 1 for reduce collapse");
    }
    // from {1,1,1,2,2,1,1} to {1,4,1}
    SmallVector<int64_t> collapseShapeInput;
    SmallVector<int64_t> newDims;
    SmallVector<ReassociationIndices> reassociationMap;
    int dim = 0;
    for (size_t i = 0; i < inputRank; i++) {
      bool reduceAxis = llvm::is_contained(dims, i);
      if (i > 0 && (reduceAxis || llvm::is_contained(dims, i - 1))) {
        // reduce axis
        dim++;
      }
      if (reduceAxis) {
        newDims.push_back(dim);
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(i);
      collapseShapeInput[dim] *= inputType.getDimSize(i);
    }
    if (collapseShapeInput.size() == inputRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse reduce shape");
    }
    SmallVector<int64_t> collapseShapeOutput;
    for (size_t i = 0; i < collapseShapeInput.size(); i++) {
      if (llvm::is_contained(newDims, i)) {
        // reduce axis
        continue;
      }
      collapseShapeOutput.push_back(collapseShapeInput[i]);
    }
    auto elementType = inputType.getElementType();
    auto loc = op.getLoc();
    auto newInputType = RankedTensorType::get(collapseShapeInput, elementType);
    input = tensor::CollapseShapeOp::create(rewriter, loc, newInputType, input,
                                            reassociationMap);

    SmallVector<ReassociationIndices> reassociationMapOutput;
    int idx = 0;
    for (size_t i = 0; i < reassociationMap.size(); ++i) {
      if (llvm::is_contained(newDims, i)) {
        continue; // skip reduce axis
      } else {
        reassociationMapOutput.push_back({});
      }
      for (size_t j = 0; j < reassociationMap[i].size(); ++j) {
        reassociationMapOutput.back().push_back(idx++);
      }
    }
    auto init = tensor::CollapseShapeOp::create(
        rewriter, loc, RankedTensorType::get(collapseShapeOutput, elementType),
        op.getInits()[0], reassociationMapOutput);
    auto newReduce =
        linalg::ReduceOp::create(rewriter, loc, init->getResultTypes(),
                                 ValueRange{input}, ValueRange{init}, newDims);
    rewriter.cloneRegionBefore(op.getRegion(), newReduce.getRegion(),
                               newReduce.getRegion().begin());

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, op.getResultTypes()[0], newReduce->getResult(0),
        reassociationMapOutput);
    return success();
  }
};

// this pattern collapses continuous parallel dimensions in a linalg.generic op
// into one dimension, and generate expand_shape to recover the original shape.
// This is useful for optimizing the performance of tensor operations that
// involve many parallel dimensions.
// for example:
// before
// ```
// %3 = linalg.generic {indexing_maps = [#map, #map1, #map2, #map3,
// #map], iterator_types = ["parallel", "parallel", "parallel",
// "parallel", "parallel", "parallel", "parallel", "parallel",
// "parallel", "parallel"]} ins(%0, %1, %2, %2 :
// tensor<2x2x2x2x2x2x2x2x2x2xi32>, tensor<2x2x2x2x2x2x2x2x2xi32>,
// tensor<2xi32>, tensor<2xi32>) outs(%0 :
// tensor<2x2x2x2x2x2x2x2x2x2xi32>) {
// ^bb0(%in: i32, %in_0: i32, %in_1: i32, %in_2: i32, %out: i32):
//   %4 = arith.xori %in, %in_0 : i32
//   %5 = arith.xori %in_1, %in_2 : i32
//   %6 = arith.cmpi ugt, %in, %4 : i32
//   %7 = arith.extui %6 : i1 to i32
//   %8 = arith.cmpi ne, %7, %5 : i32
//   %9 = arith.select %8, %4, %in : i32
//   linalg.yield %9 : i32
// } -> tensor<2x2x2x2x2x2x2x2x2x2xi32>
// ```
// after
// ```
// %collapsed = tensor.collapse_shape %0 [[0, 1, 2, 3, 4, 5, 6, 7], [8], [9]] :
// tensor<2x2x2x2x2x2x2x2x2x2xi32> into tensor<256x2x2xi32> %collapsed_0 =
// tensor.collapse_shape %1 [[0, 1, 2, 3, 4, 5, 6, 7], [8]] :
// tensor<2x2x2x2x2x2x2x2x2xi32> into tensor<256x2xi32> %collapsed_1 =
// tensor.collapse_shape %0 [[0, 1, 2, 3, 4, 5, 6, 7], [8], [9]] :
// tensor<2x2x2x2x2x2x2x2x2x2xi32> into tensor<256x2x2xi32> %3 = linalg.generic
// {indexing_maps = [#map, #map1, #map2, #map3, #map], iterator_types =
// ["parallel", "parallel", "parallel"]} ins(%collapsed, %collapsed_0, %2, %2 :
// tensor<256x2x2xi32>, tensor<256x2xi32>, tensor<2xi32>, tensor<2xi32>)
// outs(%collapsed_1 : tensor<256x2x2xi32>) { ^bb0(%in: i32, %in_2: i32, %in_3:
// i32, %in_4: i32, %out: i32):
//   %4 = arith.xori %in, %in_2 : i32
//   %5 = arith.xori %in_3, %in_4 : i32
//   %6 = arith.cmpi ugt, %in, %4 : i32
//   %7 = arith.extui %6 : i1 to i32
//   %8 = arith.cmpi ne, %7, %5 : i32
//   %9 = arith.select %8, %4, %in : i32
//   linalg.yield %9 : i32
// } -> tensor<256x2x2xi32>
// %expanded = tensor.expand_shape %3 [[0, 1, 2, 3, 4, 5, 6, 7], [8], [9]]
// output_shape [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] : tensor<256x2x2xi32> into
// tensor<2x2x2x2x2x2x2x2x2x2xi32>
// ```
struct FlattenGeneric final : OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  bool canExtendGroup(int64_t prefix, int64_t newDim,
                      ArrayRef<AffineMap> maps) const {
    SmallVector<bool> preFixFlag(maps.size(), false);
    SmallVector<bool> newDimFlag(maps.size(), false);

    // check newDim in all maps and tensors
    for (unsigned i = 0; i < maps.size(); ++i) {
      auto map = maps[i];

      for (auto expr : map.getResults()) {
        if (isa<AffineConstantExpr>(expr)) {
          continue;
        }
        unsigned pos = cast<AffineDimExpr>(expr).getPosition();

        // Only care about whether this result uses newDim.
        if (pos != newDim) {
          if (pos == prefix)
            preFixFlag[i] = true;
          continue;
        }
        if (newDimFlag[i])
          return false;
        newDimFlag[i] = true;
      }
    }
    if (prefix >= 0 && preFixFlag != newDimFlag)
      return false;
    return true;
  }

  // find continuous parallel axis，generate reassociation
  std::pair<SmallVector<ReassociationIndices>, SmallVector<utils::IteratorType>>
  computeParallelReassociation(ArrayRef<utils::IteratorType> its,
                               ArrayRef<AffineMap> maps) const {
    SmallVector<ReassociationIndices> result;
    SmallVector<utils::IteratorType> newIts;

    int64_t n = its.size();
    for (int64_t i = 0; i < n;) {
      newIts.push_back(its[i]);

      if (its[i] != utils::IteratorType::parallel) {
        result.push_back({i});
        ++i;
        continue;
      }

      ReassociationIndices group = {i};
      ++i;

      while (i < n && its[i] == utils::IteratorType::parallel) {
        if (!canExtendGroup(group.size() ? group.back() : -1, i, maps))
          break;
        group.push_back(i);
        ++i;
      }
      result.push_back(group);
    }
    return {result, newIts};
  }

  SmallVector<Value> computeOriginalIndices(
      const SmallVector<mlir::ReassociationIndices> &reassociationMap,
      llvm::ArrayRef<int64_t> shape, PatternRewriter &rewriter,
      Location loc) const {
    unsigned rank = shape.size();
    SmallVector<Value> newIndex(rank, nullptr);
    // Iterate over each reassociation group (each group corresponds to one
    // flattened dimension)
    for (size_t i = 0; i < reassociationMap.size(); ++i) {
      // The linear index corresponding to this flattened dimension
      Value linearIdx = linalg::IndexOp::create(rewriter, loc, i);

      // Expand each group into its original dimensions
      const auto &group = reassociationMap[i];
      if (group.empty())
        continue;

      Value idx = linearIdx;
      // Compute the original index for each dimension in this group
      for (size_t j = 0; j < group.size(); ++j) {
        size_t dim = group[j];

        // Compute the stride = product of the sizes of all subsequent
        // dimensions in this group
        int64_t strideVal = 1;
        for (size_t k = j + 1; k < group.size(); ++k)
          strideVal *= shape[group[k]];

        Value stride = arith::ConstantIndexOp::create(rewriter, loc, strideVal);
        // Compute the original index for this dimension using integer division
        Value di = arith::DivUIOp::create(rewriter, loc, idx, stride);
        newIndex[dim] = di;

        if (j + 1 < group.size()) {
          // Update the remaining linear index for the next dimension
          idx = arith::RemUIOp::create(rewriter, loc, idx, stride);
        }
      }
    }
    return newIndex;
  }

  AffineMap
  makeExpandMap(const SmallVector<mlir::ReassociationIndices> &reassoc,
                ArrayRef<int64_t> oldShape, unsigned newRank,
                MLIRContext *ctx) const {
    // dims: d0..d(newRank-1)
    SmallVector<AffineExpr> results(oldShape.size());

    for (unsigned newPos = 0; newPos < reassoc.size(); ++newPos) {
      auto &group = reassoc[newPos];
      int64_t stride = 1;

      // linearize from inner to outer.
      for (int i = group.size() - 1; i >= 0; --i) {
        int64_t oldDim = group[i];
        auto d = getAffineDimExpr(newPos, ctx);
        AffineExpr expr = d.floorDiv(stride);
        if (oldShape[oldDim] > 1)
          expr = expr % oldShape[oldDim];
        results[oldDim] = expr;
        stride *= oldShape[oldDim];
      }
    }

    // the result is oldRank expressions.
    return AffineMap::get(newRank, 0, results, ctx);
  }

  AffineMap
  makeCollapseMap(const SmallVector<mlir::ReassociationIndices> &reassoc,
                  ArrayRef<int64_t> oldShape, unsigned oldRank,
                  MLIRContext *ctx) const {
    SmallVector<AffineExpr> results;

    for (unsigned newPos = 0; newPos < reassoc.size(); ++newPos) {
      auto &group = reassoc[newPos];
      AffineExpr expr = getAffineConstantExpr(0, ctx);

      int64_t stride = 1;
      for (int i = group.size() - 1; i >= 0; --i) {
        int64_t oldDim = group[i];
        auto d = getAffineDimExpr(oldDim, ctx);
        expr = expr + d * stride;
        stride *= oldShape[oldDim];
      }

      results.push_back(expr);
    }

    // the result is oldRank expressions
    return AffineMap::get(oldRank, 0, results, ctx);
  }

  // remap old indexing map to new indexing map after collapse parallel dims
  AffineMap remapIndexingMap(AffineMap oldMap,
                             const SmallVector<ReassociationIndices> &reassoc,
                             ArrayRef<int64_t> loopRanges, // old loop extents
                             unsigned newRank, unsigned oldRank,
                             MLIRContext *ctx) const {
    auto expandMap = makeExpandMap(reassoc, loopRanges, newRank, ctx);
    // old indexing_map: (i_old) -> (t_old)
    auto tmp = oldMap.compose(expandMap); // old ∘ expand
    auto collapseMap = makeCollapseMap(reassoc, loopRanges, oldRank, ctx);
    bool dddd = collapseMap.compose(expandMap).isIdentity();
    // new_map: (d_new) -> (t_new)
    auto new_map = collapseMap.compose(tmp); // collapse ∘ (old ∘ expand)
    return new_map;
  }

  AffineMap
  remapIndexingMapPure(AffineMap oldMap,
                       const SmallVector<ReassociationIndices> &reassoc,
                       MLIRContext *ctx) const {
    SmallVector<AffineExpr> newResults;
    newResults.reserve(reassoc.size());

    // old dim -> new dim
    DenseMap<unsigned, unsigned> oldToNew;
    for (unsigned newPos = 0; newPos < reassoc.size(); ++newPos)
      for (int64_t oldPos : reassoc[newPos])
        oldToNew[oldPos] = newPos;

    int pos = -1;
    for (auto it : llvm::enumerate(oldMap.getResults())) {
      auto expr = it.value();
      unsigned oldPos = isa<AffineConstantExpr>(expr)
                            ? it.index()
                            : cast<AffineDimExpr>(expr).getPosition();
      unsigned newPos = oldToNew.lookup(oldPos);
      if (newPos == pos)
        continue;
      pos = newPos;
      newResults.push_back(
          isa<AffineConstantExpr>(expr) ? expr : getAffineDimExpr(newPos, ctx));
    }
    assert(newResults.size() <= reassoc.size() &&
           "pure remap must produce one result per reassoc group");

    return AffineMap::get(reassoc.size(), 0, newResults, ctx);
  }

  bool isPureDimProjection(AffineMap map) const {
    int64_t pos = -1;
    for (auto expr : map.getResults()) {
      auto d = dyn_cast<AffineDimExpr>(expr);
      if (!d) {
        if (auto c = dyn_cast<AffineConstantExpr>(expr)) {
          if (c.getValue() == 0)
            continue;
        }
        return false;
      }
      // bypass reshape,transpose may change the order of dim, but we only care
      // about the relative order of parallel dims, so we allow non-parallel dim
      // in between, but not allow parallel dim out of order
      if (d.getPosition() <= pos)
        return false;
      pos = d.getPosition();
    }
    return true;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // only support static shape for now
    for (auto v : op.getOperands()) {
      auto ty = dyn_cast<ShapedType>(v.getType());
      if (!ty || !ty.hasStaticShape())
        return failure();
    }

    // only support dim projection for now
    for (auto m : op.getIndexingMapsArray()) {
      if (!isPureDimProjection(m))
        return failure();
    }

    auto [reassoc, newIts] = computeParallelReassociation(
        op.getIteratorTypesArray(), op.getIndexingMapsArray());
    bool hasCollapse =
        llvm::any_of(reassoc, [](auto &g) { return g.size() > 1; });
    if (!hasCollapse)
      return failure();

    llvm::MapVector<Value, SmallVector<ReassociationIndices>> operandReassoc;
    for (int operandIdx = 0; operandIdx < op.getNumOperands(); operandIdx++) {
      auto v = op.getOperand(operandIdx);
      auto map = op.getIndexingMapsArray()[operandIdx];
      SmallVector<ReassociationIndices> r;
      for (auto &group : reassoc) {
        ReassociationIndices g;
        for (auto d : group) {
          for (auto it : llvm::enumerate(map.getResults())) {
            if (auto dim = dyn_cast<AffineDimExpr>(it.value()))
              if (dim.getPosition() == d)
                g.push_back(it.index());
            if (isa<AffineConstantExpr>(it.value()))
              if (it.index() == d)
                g.push_back(it.index());
          }
        }
        if (g.size())
          r.push_back(g);
      }
      if (r.size())
        operandReassoc.insert({v, r});
    }

    auto loc = op.getLoc();

    auto collapse =
        [&](Value v,
            llvm::ArrayRef<mlir::ReassociationIndices> reassociation) -> Value {
      if (isa<TensorType>(v.getType()))
        return tensor::CollapseShapeOp::create(rewriter, loc, v, reassociation);
      else
        return memref::CollapseShapeOp::create(rewriter, loc, v, reassociation);
    };

    SmallVector<Value> newIns, newOuts;
    for (auto in : op.getInputs())
      newIns.push_back(collapse(in, operandReassoc[in]));
    for (auto outv : op.getOutputs())
      newOuts.push_back(collapse(outv, operandReassoc[outv]));

    SmallVector<AffineMap> newMaps;
    for (auto &indexMap : op.getIndexingMapsArray()) {
      newMaps.push_back(
          remapIndexingMapPure(indexMap, reassoc, op.getContext()));
    }
    SmallVector<Type> resultTypes;
    for (auto out : newOuts)
      resultTypes.push_back(out.getType());
    auto newOp = linalg::GenericOp::create(rewriter, loc, resultTypes, newIns,
                                           newOuts, newMaps, newIts);

    // inline region
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().begin());

    // remap index
    auto &newBody = newOp.getRegion().front();
    SmallVector<linalg::IndexOp> originalIndices;
    for (auto idx : newBody.getOps<linalg::IndexOp>())
      originalIndices.push_back(idx);

    if (!originalIndices.empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&newBody);
      SmallVector<Value> newIndices = computeOriginalIndices(
          reassoc, op.getStaticLoopRanges(), rewriter, loc);

      for (auto idx : originalIndices) {
        unsigned dim = idx.getDim();
        if (!newIndices[dim])
          return rewriter.notifyMatchFailure(
              idx, "failed to compute original index");
        rewriter.replaceOp(idx, newIndices[dim]);
      }
    }

    auto expand =
        [&](Value v, mlir::Type resultType,
            llvm::ArrayRef<mlir::ReassociationIndices> reassociation) -> Value {
      if (isa<TensorType>(resultType))
        return tensor::ExpandShapeOp::create(rewriter, loc, resultType, v,
                                             reassociation);
      else
        return memref::ExpandShapeOp::create(rewriter, loc, resultType, v,
                                             reassociation);
    };
    SmallVector<Value> expands;
    for (int idx = 0; idx < op.getOutputs().size(); ++idx) {
      auto out = op.getOutputs()[idx];
      expands.push_back(
          expand(newOp.getResult(idx), out.getType(), operandReassoc[out]));
    }
    rewriter.replaceOp(op, expands);
    return success();
  }
};

class CollapseShapePasss
    : public ::impl::CollapseShapeBase<CollapseShapePasss> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<CollapseFill, CollapseBroadCast, CollapseTranspose,
                 CollapseReduce, FlattenGeneric>(&getContext());
    (void)(applyPatternsGreedily(moduleOp, std::move(patterns)));
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createCollapseShapePass() {
  return std::make_unique<CollapseShapePasss>();
}
