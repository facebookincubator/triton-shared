//===----------------------------------------------------------------------===//
//
// Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TPtr_DIALECT
#define TPtr_DIALECT

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"
include "mlir/Dialect/Ptr/IR/PtrDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def TPtr_Dialect : Dialect {
  let name = "tptr";

  let cppNamespace = "::mlir::tptr";

  let summary = "Temporary Pointer Dialect";

  let description = [{
    Typed Pointer Dialect.
  }];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let dependentDialects = [
    "mlir::ptr::PtrDialect"
  ];

  let useDefaultAttributePrinterParser = 1;
}

class TPtrTypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TPtr_Dialect, name, traits> {
    // Used by printer/parser
    let mnemonic = _mnemonic;
}

class TPtr_Attr<string name, string _mnemonic,
                     list<Trait> traits = []>
    : AttrDef<TPtr_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

// A shaped value type with value semantics and rank.
class Ptr_ShapedValueType<list<Type> allowedTypes, list<Pred> preds = []> :
  ShapedContainerType<allowedTypes,
    /*containerPred=*/And<[HasValueSemanticsPred] # preds>,
    /*descr=*/[{A shaped type with value semantics and rank.}],
    /*cppType=*/"::mlir::ShapedType">;

// A ptr-like type, either scalar or shaped type with value semantics.
def Ptr_PtrLikeType :
  AnyTypeOf<[Ptr_ShapedValueType<[Ptr_PtrType], [HasRankPred]>, Ptr_PtrType]>;

// An int-like type, either scalar or shaped type with value semantics.
def Ptr_IntLikeType :AnyTypeOf<[
  Ptr_ShapedValueType<[AnySignlessIntegerOrIndex], [HasRankPred]>,
  AnySignlessIntegerOrIndex
]>;

// Memory space attr is required for building Ptr ops
// This acts as default memory space since there is
// no such default implemented upstream
def DefaultMemorySpaceAttr
    : TPtr_Attr<"DefaultMemorySpace", "default_memory_space",
                     [DeclareAttrInterfaceMethods<MemorySpaceAttrInterface>]> {
}

//
// Op Base
//
class TPtr_Op<string mnemonic, list<Trait> traits = []> :
    Op<TPtr_Dialect, mnemonic, traits> {
}

def TPtr_IntToPtrOp : TPtr_Op<"int_to_ptr", [
    Pure
  ]> {
  let summary = "Integer to a pointer operation";
  let description = [{
    The `int_to_ptr` operation casts an int or index value to a pointer.

    Example:
    ```mlir
    %ptr = ptr.int_to_ptr %int : i32 to !ptr.ptr<1 : i32>
    ```
  }];
  let arguments = (ins Ptr_IntLikeType:$arg);
  let results = (outs Ptr_PtrLikeType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

def TPtr_PtrToIntOp : TPtr_Op<"ptr_to_int", [
    Pure
  ]> {
  let summary = "Pointer to an integer operation";
  let description = [{
    The `ptr_to_int` operation casts a pointer value to an int or index.

    Example:
    ```mlir
    %int = ptr.ptr_to_int %ptr : !ptr.ptr<1 : i32> to i32
    ```
  }];
  let arguments = (ins Ptr_PtrLikeType:$arg);
  let results = (outs Ptr_IntLikeType:$res);
  let assemblyFormat = "$arg attr-dict `:` type($arg) `to` type($res)";
}

//===----------------------------------------------------------------------===//
// FromPtrOp
//===----------------------------------------------------------------------===//

// NOTE: this code is copy from mlir/include/mlir/Dialect/Ptr/IR/PtrOps.td, 
// we change this op to-llvm conversion function implementation.
// We allow ptr without metadata. In that case, we will assume the metadata 
// is trivial and only use the pointer value.
// TODO: we should consider to move this code to mlir upstream if it's generally useful.
def TPtr_FromPtrOp : TPtr_Op<"from_ptr", [
    Pure, OptionalTypesMatchWith<"metadata type", "result", "metadata",
            "ptr::PtrMetadataType::get(cast<PtrLikeTypeInterface>($_self))">
  ]> {
  let summary = "Casts a `!ptr.ptr` value to a ptr-like value.";
  let description = [{
    The `from_ptr` operation casts a `ptr` value to a ptr-like object. It's
    important to note that:
    - The ptr-like object cannot be a `!ptr.ptr`.
    - The memory-space of both the `ptr` and ptr-like object must match.
    - The cast is Pure (no UB and side-effect free).

    The optional `metadata` operand exists to provide any ptr-like metadata
    that might be required to perform the cast.

    Example:

    ```mlir
    %typed_ptr = tptr.from_ptr %ptr : !ptr.ptr<#ptr.generic_space> -> !my.ptr<f32, #ptr.generic_space>
    %memref = tptr.from_ptr %ptr metadata %md : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>

    // Cast the `%ptr` to a memref without utilizing metadata.
    %memref = tptr.from_ptr %ptr : !ptr.ptr<#ptr.generic_space> -> memref<f32, #ptr.generic_space>
    ```
  }];

  let arguments = (ins Ptr_PtrType:$ptr, Optional<Ptr_PtrMetadata>:$metadata);
  let results = (outs PtrLikeTypeInterface:$result);
  let assemblyFormat = [{
    $ptr (`metadata` $metadata^)? attr-dict `:` type($ptr) `->` type($result)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}
#endif // TPtr_DIALECT
