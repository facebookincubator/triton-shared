//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "passes.h"

#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/CollapseShape.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TPtr/IR/TPtrDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Transform/AddLLVMDebugInfo/AddLLVMDebugInfo.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Extensions/AllExtensions.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mlir;

void init_triton_triton_shared(py::module &&m) {
  m.doc() = "Python bindings to the triton-shared backend";

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    // registry.insert<mlir::triton::amdgpu::TritonAMDGPUDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    sparse_tensor::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    cf::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerInferTypeOpInterfaceExternalModels(registry);

    mlir::bufferization::registerAllExtensions(registry);

    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  //===----------------------------------------------------------------------===//
  // Triton / Custom Passes
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_triton_to_linalg_experimental",
                     triton::createTritonToLinalgExperimentalPass);
  ADD_PASS_WRAPPER_1("add_triton_to_structured",
                     triton::createTritonToStructuredPass, bool);
  ADD_PASS_WRAPPER_0("add_triton_to_unstructured",
                     triton::createTritonToUnstructuredPass);
  ADD_PASS_WRAPPER_1("add_triton_arith_to_linalg",
                     triton::createTritonArithToLinalgPass, bool);
  ADD_PASS_WRAPPER_0("add_structured_to_memref",
                     triton::createStructuredToMemrefPass);
  ADD_PASS_WRAPPER_0("add_unstructured_to_memref",
                     triton::createUnstructuredToMemrefPass);
  ADD_PASS_WRAPPER_0("add_triton_ptr_to_memref",
                     triton::createTritonPtrToMemrefPass);
  ADD_PASS_WRAPPER_0("add_triton_to_ptr", triton::createTritonToPtrPass);
  ADD_PASS_WRAPPER_0("add_reconcile_ptr_casts",
                     triton::createReconcilePtrCastsPass);
  ADD_PASS_WRAPPER_0("add_collapse_shape", triton::createCollapseShapePass);

  ADD_PASS_WRAPPER_0("add_llvm_debug_info", triton::createAddLLVMDebugInfoPass);

  //===----------------------------------------------------------------------===//
  // Linalg / Affine / Bufferization
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_convert_linalg_to_affine_loops",
                     mlir::createConvertLinalgToAffineLoopsPass);

  ADD_PASS_WRAPPER_0("add_eliminate_empty_tensors",
                     mlir::bufferization::createEmptyTensorEliminationPass);
  ADD_PASS_WRAPPER_0("add_empty_tensor_to_alloc_tensor",
                     mlir::bufferization::createEmptyTensorToAllocTensorPass);

  m.def("add_one_shot_bufferize", [](mlir::PassManager &pm) {
    mlir::bufferization::OneShotBufferizePassOptions options;
    options.allowReturnAllocsFromLoops = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
  });

  ADD_PASS_WRAPPER_0("add_lower_affine", mlir::createLowerAffinePass);

  ADD_PASS_WRAPPER_0("add_convert_linalg_to_loops",
                     mlir::createConvertLinalgToLoopsPass);

  //===----------------------------------------------------------------------===//
  // SCF / CF
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_convert_scf_to_cf", mlir::createSCFToControlFlowPass);

  //===----------------------------------------------------------------------===//
  // Metadata / MemRef
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_expand_strided_metadata",
                     mlir::memref::createExpandStridedMetadataPass);

  ADD_PASS_WRAPPER_0("add_memref_expand", mlir::memref::createExpandOpsPass);

  ADD_PASS_WRAPPER_0("add_finalize_memref_to_llvm",
                     mlir::createFinalizeMemRefToLLVMConversionPass);

  //===----------------------------------------------------------------------===//
  // LLVM Lowering
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_convert_arith_to_llvm",
                     mlir::createArithToLLVMConversionPass);

  ADD_PASS_WRAPPER_0("add_convert_cf_to_llvm",
                     mlir::createConvertControlFlowToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_math_to_llvm",
                     mlir::createConvertMathToLLVMPass);

  ADD_PASS_WRAPPER_0("add_convert_complex_to_llvm",
                     mlir::createConvertComplexToLLVMPass);

  ADD_PASS_WRAPPER_0("add_convert_vector_to_llvm",
                     mlir::createConvertVectorToLLVMPass);

  ADD_PASS_WRAPPER_0("add_convert_index_to_llvm",
                     mlir::createConvertIndexToLLVMPass);

  ADD_PASS_WRAPPER_0("add_convert_func_to_llvm",
                     mlir::createConvertFuncToLLVMPass);

  //===----------------------------------------------------------------------===//
  // Cleanup
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_reconcile_unrealized_casts",
                     mlir::createReconcileUnrealizedCastsPass);
  ADD_PASS_WRAPPER_0("add_remove_dead_code", mlir::createRemoveDeadValuesPass);
}
