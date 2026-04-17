#ifndef LINALG_TO_VECTOR_CONVERSION_PASSES
#define LINALG_TO_VECTOR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToVector : Pass<"linalg-to-vector", "mlir::ModuleOp"> {
  let summary = "Convert linalg to vector";
  let description = [{
      This pass converts the Linalg Dialect into the Vector Dialect.
    }];
  let dependentDialects = ["arith::ArithDialect", "math::MathDialect",
                           "linalg::LinalgDialect", "affine::AffineDialect",
                           "scf::SCFDialect", "tensor::TensorDialect",
                           "memref::MemRefDialect", "vector::VectorDialect"];

  let options =
      [Option<"dumpIntermediateSteps", "dump-intermediate-steps", "bool",
              /*default*/ "false", "Dump intermediate steps">,
       Option<"use1DTiling", "use-1D-tiling", "bool", /*default*/ "false",
              "use 1d tiling, only tiling the innermost dimension. "
              "the innermost dimension will be treated as dynamic">,
       // TODO: in some case, vector<16xi1>/vector<8xi1> causes precision issue.
       Option<"vectorBits", "vector-bits", "uint", /*default*/ "4096",
              "vector bits">,
       Option<"vectorizeNDExtract", "vectorize-nd-extract", "bool",
              /*default*/ "true",
              "Controls whether to vectorize `tensor.extract` when the input "
              "tensor is rank >= 2.">,
       Option<"flatten1DDepthwiseConv", "flatten-1d-depthwiseconv", "bool",
              /*default*/ "false",
              "Controls whether to the channel dimension when vectorising 1D depthwise "
              "convolutions. This should lead to bette vectorization for "
              "tensors with a low number of channel dimensions.">,
       Option<"enableTreeReduction", "enable-tree-reduction", "bool",
              /*default*/ "false",
              "Enable tree-based decomposition of linalg.reduce into "
              "block-level and final-stage reductions for vectorization. "
              "NOTE: May change floating-point semantics due to re-association.">];
}

#endif
