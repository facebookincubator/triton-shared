#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_triton_library(VerifyTritonCheckpoint
  VerifyTritonCheckpointPass.cpp

  DEPENDS
  VerifyTritonCheckpointTransformPassIncGen

  LINK_LIBS PUBLIC
  TPtrIR
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  TritonStructuredIR
  TritonTilingExtIR
  MLIRIR
  MLIRPass
)
