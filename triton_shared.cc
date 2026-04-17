//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "passes.h"

#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TPtrToLLVM/TPtrToLLVM.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
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

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/PtrToLLVM/PtrToLLVM.h"
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
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SparseTensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Extensions/AllExtensions.h"
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

  m.def("load_dialects", [](MLIRContext &context) {
    DialectRegistry registry;
    mlir::tptr::registerConvertTPtrToLLVMInterface(registry);

    // registry.insert<triton::amdgpu::TritonAMDGPUDialect>();
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

    // Register all conversions to LLVM extensions.
    arith::registerConvertArithToLLVMInterface(registry);
    bufferization::registerAllExtensions(registry);
    registerConvertComplexToLLVMInterface(registry);
    cf::registerConvertControlFlowToLLVMInterface(registry);
    func::registerAllExtensions(registry);
    tensor::registerAllExtensions(registry);
    registerConvertFuncToLLVMInterface(registry);
    index::registerConvertIndexToLLVMInterface(registry);
    registerConvertMathToLLVMInterface(registry);
    registerConvertMemRefToLLVMInterface(registry);
    ptr::registerConvertPtrToLLVMInterface(registry);
    ub::registerConvertUBToLLVMInterface(registry);
    vector::registerConvertVectorToLLVMInterface(registry);
    registerConvertX86ToLLVMInterface(registry);

    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  //===----------------------------------------------------------------------===//
  // Triton / Custom Passes
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_triton_to_linalg_experimental",
                     triton::createTritonToLinalgExperimentalPass);

  ADD_PASS_WRAPPER_0("add_llvm_debug_info", triton::createAddLLVMDebugInfoPass);

  //===----------------------------------------------------------------------===//
  // Linalg / Affine / Bufferization
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_convert_linalg_to_affine_loops",
                     createConvertLinalgToAffineLoopsPass);

  ADD_PASS_WRAPPER_0("add_eliminate_empty_tensors",
                     bufferization::createEmptyTensorEliminationPass);
  ADD_PASS_WRAPPER_0("add_empty_tensor_to_alloc_tensor",
                     bufferization::createEmptyTensorToAllocTensorPass);

  m.def("add_one_shot_bufferize", [](PassManager &pm) {
    bufferization::OneShotBufferizePassOptions options;
    options.allowReturnAllocsFromLoops = true;
    pm.addPass(bufferization::createOneShotBufferizePass(options));
  });

  ADD_PASS_WRAPPER_0("add_lower_affine", createLowerAffinePass);

  ADD_PASS_WRAPPER_0("add_convert_linalg_to_loops",
                     createConvertLinalgToLoopsPass);

  //===----------------------------------------------------------------------===//
  // SCF / CF
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_convert_scf_to_cf", createSCFToControlFlowPass);

  //===----------------------------------------------------------------------===//
  // Metadata / MemRef
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_expand_strided_metadata",
                     memref::createExpandStridedMetadataPass);

  ADD_PASS_WRAPPER_0("add_memref_expand", memref::createExpandOpsPass);

  ADD_PASS_WRAPPER_0("add_finalize_memref_to_llvm",
                     createFinalizeMemRefToLLVMConversionPass);

  //===----------------------------------------------------------------------===//
  // LLVM Lowering
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_convert_arith_to_llvm",
                     createArithToLLVMConversionPass);

  ADD_PASS_WRAPPER_0("add_convert_cf_to_llvm",
                     createConvertControlFlowToLLVMPass);
  ADD_PASS_WRAPPER_0("add_convert_math_to_llvm", createConvertMathToLLVMPass);

  ADD_PASS_WRAPPER_0("add_convert_complex_to_llvm",
                     createConvertComplexToLLVMPass);

  ADD_PASS_WRAPPER_0("add_convert_vector_to_llvm",
                     createConvertVectorToLLVMPass);

  ADD_PASS_WRAPPER_0("add_convert_index_to_llvm", createConvertIndexToLLVMPass);

  ADD_PASS_WRAPPER_0("add_convert_func_to_llvm", createConvertFuncToLLVMPass);

  ADD_PASS_WRAPPER_0("add_convert_to_llvm", createConvertToLLVMPass);

  //===----------------------------------------------------------------------===//
  // Cleanup
  //===----------------------------------------------------------------------===//

  ADD_PASS_WRAPPER_0("add_reconcile_unrealized_casts",
                     createReconcileUnrealizedCastsPass);
}
