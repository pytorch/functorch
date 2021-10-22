import torch
from torch.nn import functional as F
from functorch.compile import memory_efficient_operator_authoring
from torch.testing._internal.common_utils import TestCase, run_tests
import inspect
from typing import Callable


def _num_args(fn: Callable):
    return len(inspect.signature(fn).parameters)


def gelu_bias(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x.mul(torch.tanh(F.softplus(x)))


def hard_sigmoid(x):
    return (x + 3).clamp(min=0, max=6).div(6.0)


def hard_swish(x):
    return x * (x + 3).clamp(min=0, max=6).div(6.0)


def hard_mish(x):
    return 0.5 * x * (x + 2).clamp(min=0, max=2)


def run_and_compare_activation(fn, shape, device, compiler_name):
    if _num_args(fn) == 1:
        ref_a = torch.rand(shape, device=device, requires_grad=True)
        ref_args = (ref_a,)
        res_a = ref_a.clone().detach().requires_grad_(True)
        res_args = (res_a,)
    elif _num_args(fn) == 2:
        ref_a = torch.rand(shape, device=device, requires_grad=True)
        ref_b = torch.rand(shape, device=device, requires_grad=True)
        ref_args = (ref_a, ref_b)
        res_a = ref_a.clone().detach().requires_grad_(True)
        res_b = ref_b.clone().detach().requires_grad_(True)
        res_args = (res_a, res_b)

    ref = fn(*ref_args)
    ref.sum().backward()

    mem_optimized_fn = memory_efficient_operator_authoring(fn, compiler_name)
    res = mem_optimized_fn(*res_args)
    res.sum().backward()

    assert torch.allclose(ref, res)
    for idx in range(_num_args(fn)):
        assert torch.allclose(ref_args[idx].grad, res_args[idx].grad)


class TesMemoryEfficientOpAuthroing(TestCase):
    def test_gelu_bias(self):
        run_and_compare_activation(gelu_bias, 1024, "cpu", "torchscript_nnc")

    def test_mish(self):
        run_and_compare_activation(mish, 1024, "cpu", "torchscript_nnc")

    def test_swish(self):
        run_and_compare_activation(swish, 1024, "cpu", "torchscript_nnc")

    def test_hard_sigmoid(self):
        run_and_compare_activation(hard_sigmoid, 1024, "cpu", "torchscript_nnc")

    def test_hard_swish(self):
        run_and_compare_activation(hard_swish, 1024, "cpu", "torchscript_nnc")

    # TODO - Assertion failure
    # def test_hard_mish(self):
    #     run_and_compare_activation(hard_mish, 1024, "cpu", "torchscript_nnc")


if __name__ == "__main__":
    run_tests()
