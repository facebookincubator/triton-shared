add_triton_library(TPtrToLLVM
  TPtrToLLVM.cpp

  DEPENDS
  TPtrTableGen

  LINK_LIBS PUBLIC
  TPtrIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
)
