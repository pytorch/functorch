from functools import wraps, partial
from itertools import product, chain
import itertools
import collections
import copy
import operator
import random
import numbers
import unittest

import torch
import numpy as np
from torch._six import inf
import collections.abc

from typing import Any, List, Sequence, Tuple, Union

from torch.testing import \
    (make_non_contiguous, floating_types, floating_types_and, complex_types,
     floating_and_complex_types, floating_and_complex_types_and,
     all_types_and_complex_and, all_types_and, all_types_and_complex,
     integral_types_and, all_types, empty_types)
# from .._core import _dispatch_dtypes
from torch.testing._internal.common_device_type import \
    (skipIf, skipCUDAIfNoMagma, skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfNoCusolver,
     skipCPUIfNoLapack, skipCPUIfNoFFT, skipCUDAIfRocm, precisionOverride, toleranceOverride, tol,
     onlyCUDA)
from torch.testing._internal.common_cuda import CUDA11OrLater, SM53OrLater, SM60OrLater
from torch.testing._internal.common_utils import \
    (is_iterable_of_tensors,
     random_symmetric_matrix, random_symmetric_psd_matrix,
     make_fullrank_matrices_with_distinct_singular_values,
     random_symmetric_pd_matrix, make_symmetric_matrices,
     make_symmetric_pd_matrices,
     random_fullrank_matrix_distinct_singular_value,
     TEST_WITH_ROCM, IS_WINDOWS, IS_MACOS, make_tensor, TEST_SCIPY,
     torch_to_numpy_dtype_dict, TEST_WITH_ASAN,
     GRADCHECK_NONDET_TOL,)
import torch.testing._internal.opinfo_helper as opinfo_helper
from torch.testing._internal.common_methods_invocations import (
    OpInfo, DecorateInfo, SampleInput, sample_inputs_hardshrink_hardtanh,
    sample_inputs_softmax_variant, S
)

# List of OpInfos that aren't in PyTorch Core yet.
# They are here because we wanted a fast way of writing OpInfos and may not be
# 100% correct (w.r.t. to dtypes and other options).
# TODO: Figure out how to upstream these, delete them when they're upstreamed

additional_op_db = []

# https://github.com/pytorch/pytorch/pull/61971
def sample_inputs_linear(has_bias, self, device, dtype, requires_grad):
    features_options = [[3, 4], [128, 128]]
    batch_options = [
        [], # no batch
        [64],
        [5, 7],
    ]

    sample_inputs = []
    for (in_feat, out_feat), batch_shape in itertools.product(features_options, batch_options):
        input_tensor = make_tensor(batch_shape + [in_feat], device=device,
                                   dtype=dtype, requires_grad=requires_grad,
                                   low=-2, high=2)
        weight = make_tensor([out_feat, in_feat], device=device,
                             dtype=dtype, requires_grad=requires_grad,
                             low=-2, high=2)
        if not has_bias:
            sample_inputs.append(SampleInput(input_tensor, args=(weight,)))
            continue

        bias = make_tensor([out_feat], device=device,
                           dtype=dtype, requires_grad=requires_grad,
                           low=-2, high=2)
        sample_inputs.append(SampleInput(input_tensor, args=(weight, bias)))
    return sample_inputs

additional_op_db.extend([
    OpInfo('nn.functional.linear',
           aten_name='linear',
           variant_test_name='with_bias',
           supports_autograd=True,
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_linear, False),
           supports_out=False),
    OpInfo('nn.functional.linear',
           aten_name='linear',
           variant_test_name='no_bias',
           supports_autograd=True,
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_linear, True),
           supports_out=False),
])

# https://github.com/pytorch/pytorch/pull/61068

def sample_inputs_conv2d(has_bias, self, device, dtype, requires_grad, extra_args=(), groups=1):
    in_ch, out_ch = 6, 4
    inp = make_tensor((2, in_ch * groups, 7, 5), device=device, dtype=dtype,
                      requires_grad=requires_grad, low=-1, high=1)
    weight = make_tensor((out_ch * groups, in_ch, 3, 2), device=device, dtype=dtype,
                         requires_grad=requires_grad, low=-1, high=1)
    bias = None
    if has_bias:
        bias = make_tensor((out_ch * groups,), device=device, dtype=dtype,
                           requires_grad=requires_grad, low=-1, high=1)
    return [SampleInput(inp, args=((weight, bias) + extra_args))]

additional_op_db.extend([
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='no_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, False),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 2))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_no_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, False, extra_args=((2, 2))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_padding_with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 2), (1, 1))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_padding_no_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, False, extra_args=((2, 2), (1, 1))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='strided_padding_dilation_with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 2), (1, 1), (2, 2))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='strided_padding_dilation_no_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 2), (1, 1), (2, 2))),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_groups_with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 3), 0, 1, 2), groups=2),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
    OpInfo('nn.functional.conv2d',
           aten_name="conv2d",
           variant_test_name='stride_depthwise_with_bias',
           supports_autograd=True,
           sample_inputs_func=partial(sample_inputs_conv2d, True, extra_args=((2, 3), 0, 1, 6), groups=6),
           dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           supports_out=False),
])

def sample_inputs_cross_entropy(self, device, dtype, requires_grad, reduction):
    N = 2
    C = 10
    inp = make_tensor((2, C), device=device, dtype=dtype,
                      requires_grad=requires_grad, low=-1, high=1)
    target = torch.randint(0, C, (N,), device=device)
    inp4d = make_tensor((2, C, 4, 5), device=device, dtype=dtype,
                        requires_grad=requires_grad, low=-1, high=1)
    target4d = torch.randint(0, C, (N, 4, 5), device=device)
    weight = make_tensor((C,), device=device, dtype=dtype,
                         low=0.5, high=1)
    sample_inputs = [
        SampleInput(inp, args=(target,), kwargs={'reduction': reduction}),
        SampleInput(inp, args=(target,), kwargs={'ignore_index': 1, 'reduction': reduction}),
        SampleInput(inp, args=(target, weight), kwargs={'ignore_index': 1, 'reduction': reduction}),
    ]
    sample_inputs.extend([
        SampleInput(inp4d, args=(target4d,), kwargs={'reduction': reduction}),
        SampleInput(inp4d, args=(target4d,), kwargs={'ignore_index': 1, 'reduction': reduction}),
        SampleInput(inp4d, args=(target4d, weight), kwargs={'ignore_index': 1, 'reduction': reduction}),
    ])
    return sample_inputs

for reduction in ['mean', 'sum', 'none']:
    additional_op_db.append(
        OpInfo('nn.functional.cross_entropy',
               aten_name="cross_entropy",
               variant_test_name=reduction,
               supports_autograd=True,
               sample_inputs_func=partial(sample_inputs_cross_entropy, reduction=reduction),
               dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
               supports_out=True))

def disablecuDNN(fn):

    @wraps(fn)
    def disable_cudnn(self, *args, **kwargs):
        if self.device_type == 'cuda' and self.has_cudnn():
            with torch.backends.cudnn.flags(enabled=False):
                return fn(self, *args, **kwargs)
        return fn(self, *args, **kwargs)

    return disable_cudnn

def sample_inputs_batch_norm(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_arg_without_requires_grad = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)

    # Ordered as: input shape, kwargs for training, momentum, eps
    cases: Tuple[Tuple[int], dict] = (  # type: ignore[assignment]
        ((S, S, S), {'training': True, 'momentum': 0.5, 'eps': 0.6}),
        ((3, 2, 4), {'training': False, 'momentum': -1.2}),
        ((3, 1), {'training': True, 'momentum': 0.0}),
        ((0,), {'training': True}),
        ((0,), {'training': False}),
        ((3, 2, 3, 4), {'training': True, 'momentum': -1.0, 'eps': 0.5}),
        ((3, 2, 3, 4), {'training': False, 'momentum': -1.0, 'eps': 0.5}),
        ((2, 1), {}),
    )

    def generator():
        for input_shape, kwargs in cases:
            # args: running mean, running var, weight and bias should necessarily be of shape: (channels,)
            channels = input_shape[1] if len(input_shape) > 1 else 0
            weight = make_arg(channels) if channels > 0 else None
            bias = make_arg(channels) if channels > 0 else None
            running_mean = make_arg_without_requires_grad(channels, low=0)
            running_var = make_arg_without_requires_grad(channels, low=0)

            yield SampleInput(
                make_arg(input_shape),
                args=(
                    running_mean,
                    running_var,
                    weight,
                    bias
                ),
                kwargs=kwargs
            )
            # Checking for permutations of weights and biases as `None`
        weights = [channels, None, None]
        biases = [None, channels, None]
        is_training = [True, False, False]

        for weight, bias, training in zip(weights, biases, is_training):
            yield SampleInput(
                make_arg(input_shape),
                args=(
                    running_mean,
                    running_var,
                    make_arg(channels),
                    make_arg(channels)
                ),
                kwargs={'training': training}
            )

        # Test case for no optional kwargs
        # running_mean and running_var are required in evaluation mode (training: False) but not in training mode
        yield SampleInput(make_arg((1, 2, 3)), args=(None, None), kwargs={'training': True})
    return list(generator())

additional_op_db.extend([
    # Multiple variants for batch_norm to test with and without cuDNN disabled
    # See https://github.com/pytorch/pytorch/pull/63218#discussion_r688549391 for more details
    OpInfo('nn.functional.batch_norm',
           aten_name='batch_norm',
           dtypes=floating_types(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           skips=(
               # RuntimeError: deepEquals(input.iValue, deepCopiedInput) INTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":142, please report a bug to PyTorch
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           sample_inputs_func=sample_inputs_batch_norm),
    # This variant tests batch_norm with cuDNN disabled only on CUDA devices
    OpInfo('nn.functional.batch_norm',
           variant_test_name='without_cudnn',
           aten_name='batch_norm',
           dtypesIfCPU=empty_types(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16),
           supports_out=False,
           skips=(
               # RuntimeError: deepEquals(input.iValue, deepCopiedInput) INTERNAL ASSERT FAILED at
               # "../torch/csrc/jit/passes/utils/check_alias_annotation.cpp":142, please report a bug to PyTorch
               DecorateInfo(unittest.skip("Skipped!"), 'TestJit', 'test_variant_consistency_jit'),
           ),
           decorators=[onlyCUDA, disablecuDNN],
           sample_inputs_func=sample_inputs_batch_norm),
])
