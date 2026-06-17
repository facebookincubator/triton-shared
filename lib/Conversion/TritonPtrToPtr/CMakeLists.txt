add_triton_library(TritonPtrToPtr
  TritonPtrToPtrPass.cpp

  DEPENDS
  TritonPtrToPtrConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  MLIRReconcileUnrealizedCasts
  TritonIR
)
