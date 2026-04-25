# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
# Licensed under the MIT license.

from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, triton_shared, llvm
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from types import ModuleType
import hashlib
import tempfile
import os
import re
import subprocess
import functools
import triton
from pathlib import Path


def _get_llvm_bin_path(bin_name: str) -> str:
    path = os.getenv("LLVM_BINARY_DIR", "")
    if path == "":
        raise Exception("LLVM_BINARY_DIR is not set.")
    return os.path.join(path, bin_name)


def _get_sanitizer_type():
    # returns "" if not set
    # throws error if set to something other than "asan" or "tsan"
    sanitizer_type = os.getenv("TRITON_SHARED_SANITIZER_TYPE", "")

    if sanitizer_type != "" and sanitizer_type != "asan" and sanitizer_type != "tsan":
        # throw error
        raise Exception(f"TRITON_SHARED_SANITIZER_TYPE {sanitizer_type} is invalid.")

    return sanitizer_type


def _llir_to_bin(llir: str, metadata, options):
    pattern = r"define void @(\w+)\(.+"
    matches = re.findall(pattern, llir)
    assert len(matches) == 1
    metadata["name"] = matches[0]
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "kernel.ll")
        dst_path = os.path.join(tmpdir, "kernel.o")
        Path(src_path).write_text(llir)

        sanitizer_type = _get_sanitizer_type()

        if sanitizer_type != "":
            # using a sanitizer
            # invoke pass to append sanitizer attributes
            instrumented_src_path = os.path.join(tmpdir, "kernel-instrumented.ll")

            opt_path = _get_llvm_bin_path("opt")
            top_level_triton_path = os.path.dirname(triton.__file__)
            sanitizer_attributes_pass_path = str(
                next(Path(top_level_triton_path).rglob("libSanitizerAttributes.so"), None))

            if not sanitizer_attributes_pass_path:
                raise Exception("libSanitizerAttributes.so does not exist.")

            subprocess.check_call([
                opt_path, "-load-pass-plugin", sanitizer_attributes_pass_path, "-passes=sanitizer-attributes",
                f"-sanitizer-type={sanitizer_type}", "-S", src_path, "-o", instrumented_src_path
            ])

            # compile to object file
            clang_path = _get_llvm_bin_path("clang++")

            subprocess_args = [clang_path, "-c", instrumented_src_path, "-o", dst_path]

            if sanitizer_type == "asan":
                subprocess_args.extend(["-g", "-fsanitize=address", "-mllvm", "-asan-stack=0"])
            elif sanitizer_type == "tsan":
                subprocess_args.extend(["-g", "-fsanitize=thread"])

            subprocess.check_call(subprocess_args)
        else:
            llc_path = _get_llvm_bin_path("llc")
            subprocess.check_call([llc_path, src_path, "-filetype=obj", "-relocation-model=pic", "-o", dst_path])

        return Path(dst_path).read_bytes()


@dataclass(frozen=True)
class CPUOptions:
    debug: bool = False
    arch: str = None
    num_warps: int = 0
    num_ctas: int = 0
    num_stages: int = 1
    enable_warp_specialization: bool = False
    enable_fp_fusion: bool = False
    extern_libs = None
    cluster_dims: tuple = (1, 1, 1)
    shared: bool = False
    # Disable FP8 here since this is a sample CPU backend.
    # Target specific backends can eanble it with supported types.
    supported_fp8_dtypes: Tuple[str] = ()
    allow_fp8e4nv: bool = False
    allowed_dot_input_precisions: Tuple[str] = ("ieee", )
    sanitize_overflow: bool = True
    instrumentation_mode: str = ""

    def __post_init__(self):
        pass

    def hash(self):
        key = '_'.join([f'{name}-{val}' for name, val in self.__dict__.items()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CPUBackend(BaseBackend):
    binary_ext = 'obj'

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cpu'

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)

    def parse_options(self, opts) -> Any:
        args = {'arch': self.target.arch}
        args.update({k: opts[k] for k in CPUOptions.__dataclass_fields__.keys() if k in opts})
        return CPUOptions(**args)

    def get_codegen_implementation(self, options):
        codegen_fns = {"min_dot_size": lambda lhsType, rhsType: (1, 1, 1)}
        return codegen_fns

    def pack_metadata(self, metadata):
        # Note: We actually don't need any of these except for the name which is
        # used in the launch function in driver.py. Putting these in so we're
        # consistent with other backends
        return (metadata.num_warps, metadata.num_ctas, metadata.shared, metadata.cluster_dims[0],
                metadata.cluster_dims[1], metadata.cluster_dims[2], metadata.name)

    # Our compilation pipeline isn't in python like nvidia or amd, no need to load
    # dialects. See `triton_shared.cc`
    def load_dialects(self, ctx):
        triton_shared.load_dialects(ctx)
        return

    @staticmethod
    def make_ttir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_combine(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod, 'make_ttir')
        return mod

    @staticmethod
    def make_tt_shared_ir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        if _get_sanitizer_type() != "":
            print("Building with sanitizer support...")
            # has to run before the other passes as operates on the tt dialect
            triton_shared.add_llvm_debug_info(pm)
        triton_shared.add_triton_to_structured(pm, True)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        triton_shared.add_triton_to_unstructured(pm)
        triton_shared.add_triton_arith_to_linalg(pm, True)
        triton_shared.add_structured_to_memref(pm)
        triton_shared.add_unstructured_to_memref(pm)
        triton_shared.add_triton_ptr_to_memref(pm)
        triton_shared.add_triton_to_ptr(pm)
        triton_shared.add_reconcile_unrealized_casts(pm)
        triton_shared.add_reconcile_ptr_casts(pm)
        triton_shared.add_remove_dead_code(pm)
        passes.common.add_cse(pm)
        passes.common.add_canonicalizer(pm)
        if os.getenv("TRITON_SHARED_ENABLE_COLLAPSE_SHAPE", False):
            print("Building with collapse-shape enabled ...")
            triton_shared.add_collapse_shape(pm)
        pm.run(mod, 'make_tt_shared_ir')
        return mod

    @staticmethod
    def make_llir(mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        triton_shared.add_convert_linalg_to_affine_loops(pm)

        # Note: eliminate-empty-tensors fails when there are multiple func.return ops
        # in a single kernel which are the results of early returns.
        # See python/examples/test_early_return.py for examples.
        # We disable this pass for now since performance on CPU isn't the main
        # focus at the moment.
        # triton_shared.add_eliminate_empty_tensors(pm)

        triton_shared.add_empty_tensor_to_alloc_tensor(pm)
        triton_shared.add_one_shot_bufferize(pm)
        triton_shared.add_lower_affine(pm)
        triton_shared.add_convert_linalg_to_loops(pm)
        triton_shared.add_expand_strided_metadata(pm)
        triton_shared.add_convert_scf_to_cf(pm)
        triton_shared.add_convert_arith_to_llvm(pm)
        triton_shared.add_convert_math_to_llvm(pm)
        triton_shared.add_convert_complex_to_llvm(pm)
        triton_shared.add_convert_vector_to_llvm(pm)
        triton_shared.add_convert_index_to_llvm(pm)
        triton_shared.add_memref_expand(pm)
        triton_shared.add_finalize_memref_to_llvm(pm)
        triton_shared.add_convert_func_to_llvm(pm)
        triton_shared.add_convert_cf_to_llvm(pm)
        # Lowering memrefs creates more affine.apply ops.
        # Lowering these affine ops again creates further arith ops,
        # so we have to run these two passes again here.
        triton_shared.add_lower_affine(pm)
        triton_shared.add_convert_arith_to_llvm(pm)
        # Remove all unrealized casts created
        triton_shared.add_reconcile_unrealized_casts(pm)
        pm.run(mod, 'make_llir')

        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()
        llvm_mod = llvm.to_module(mod, context)
        return str(llvm_mod)

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttsharedir"] = lambda src, metadata: self.make_tt_shared_ir(src, metadata, options)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
        stages["obj"] = lambda src, metadata: _llir_to_bin(src, metadata, options)

    @functools.lru_cache()
    def hash(self):
        return self.target

    # The CPU backend does not use any extra python modules, return an empty dictionary
    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}
