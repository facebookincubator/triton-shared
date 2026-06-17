# Copyright (c) Meta Platforms, Inc. and affiliates, Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import os
import tempfile
import torch
import triton
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

if not torch.cuda.is_available():
    torch.cuda.get_device_capability = lambda *args, **kwargs: (0, 0)


def empty_decorator(func=None, *args, **kwargs):
    if func is not None and callable(func):
        return func

    def decorator(func):
        return func

    return decorator


pytest.mark.interpreter = empty_decorator


@pytest.fixture
def device(request):
    return "cpu"


# this fixture is used for test_enable_fp_fusion
@pytest.fixture
def fresh_knobs():
    from triton._internal_testing import _fresh_knobs_impl

    fresh_function, reset_function = _fresh_knobs_impl()
    try:
        yield fresh_function()
    finally:
        reset_function()


# this fixture is used for test_trans_4d && test_trans_reshape
@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())


unsupported_case = {
    # 'INT_MIN % -1' run into Floating point exception, which is not handled in x86 CPU
    "test_bin_op",
    "test_atomic_rmw",
    "test_tensor_atomic_rmw",
    "test_tensor_atomic_add_non_exclusive_offset",
    "test_tensor_atomic_add_shift_1",
    "test_tensor_atomic_add_access_patterns",
    "test_atomic_cas",
    "test_tensor_atomic_cas",
    "test_tensor_atomic_use_result",
    "test_scaled_dot",
    "test_atomic_rmw_predicate",
    "test_tensor_atomic_rmw_block",
    "test_atomic_min_max_neg_zero",
    # do not support launch_cooperative_grid on CPU
    "test_load_scope_sem_coop_grid_cta_not_one",
    "test_load_scope_sem_coop_grid_cta_one",
    # do not support IR CHECK with 'ttgir' on CPU
    "test_optimize_thread_locality",
    "test_cat_nd",
    "test_math_erf_op",
    "test_shapes_as_params",
    "test_no_rematerialization_op",
    "test_generic_reduction",
    "test_assume",
    "test_pointer_arguments",
    "test_num_warps_pow2",
    "test_map_elementwise",
    "test_map_elementwise_multiple_outputs",
    "test_map_elementwise_pack",
    "test_reshape",
    "test_trans_reshape",
    "test_constexpr_if_return",
    "test_tl_range_fuse",
    "test_gather",
    "test_tl_range_fuse_dependent",
    "test_tl_range_option_none",
    "test_disable_licm",
    "test_zero_strided_tensors",
    "test_unroll_attr",
    "test_tensor_member",
    "test_libdevice_rint",
}

annotations_tests_supported = {
    "test_int_annotation",
    "test_unknown_annotation",
}


def _is_float8_dtype(value):
    value = str(value)
    return "float8" in value or "fp8" in value


def pytest_collection_modifyitems(config, items):
    skip_marker = pytest.mark.skip(reason="CPU backend does not support it yet")
    # There is a dependency issue on build machine which breaks bfloat16
    skip_marker_bfloat = pytest.mark.skip(reason="bfloat16 linking issue")
    skip_marker_float16 = pytest.mark.skip(reason="float16 linking issue")
    skip_marker_tf32 = pytest.mark.skip(reason="tf32 is not supported on CPU")
    skip_marker_float8 = pytest.mark.skip(reason="float8 is not supported on CPU")

    for item in items:
        test_func_name = item.originalname if item.originalname else item.name

        test_file = str(item.fspath)
        if test_file.endswith("test_core.py") and (test_func_name in unsupported_case):
            item.add_marker(skip_marker)
            continue

        if test_file.endswith("test_annotations.py") and test_func_name not in annotations_tests_supported:
            item.add_marker(skip_marker)
            continue

        if "parametrize" in item.keywords:
            for param_name, param_value in item.callspec.params.items():
                if ('dtype' in param_name or param_name == "in_type_str"):
                    if param_value == 'bfloat16':
                        item.add_marker(skip_marker_bfloat)
                    if _is_float8_dtype(param_value):
                        item.add_marker(skip_marker_float8)
                    if param_value == 'float16' or param_value == 'fp16':
                        item.add_marker(skip_marker_float16)
                if param_name.startswith('input_precision') and (param_value.startswith('tf32')
                                                                 or param_value.startswith('bf16')):
                    item.add_marker(skip_marker_tf32)
