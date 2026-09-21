add_triton_library(TensorGatherToLinalg
  TensorGatherToLinalgPass.cpp

  DEPENDS
  TensorGatherToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonSharedUtils

  TritonArithToLinalg
)
