set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name LinalgToVector)
add_public_tablegen_target(LinalgToVectorConversionPassIncGen)
