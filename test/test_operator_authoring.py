import torch
import unittest

import numpy as np

import torch
from torch import fx
from functorch import pointwise_operator
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.jit_utils import JitTestCase
import unittest

LLVM_ENABLED = torch._C._llvm_enabled()
HAS_SYMPY = False
try:
    import sympy

    HAS_SYMPY = True
except ImportError:
    pass


def pointwise_fn(a, b):
    return (a + b) * 42


nnc_pointwise_fn = pointwise_operator(pointwise_fn)


@pointwise_operator
def custom1(a):
    return a + 1.0


@pointwise_operator
def custom2(a):
    return a + 2.0


class TorchFunctionExample(object):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        assert func in (nnc_pointwise_fn, torch.Tensor.add)
        assert all(issubclass(t, (torch.Tensor, TorchFunctionExample)) for t in types)
        return torch.zeros_like(args[0])


class TestOperatorAuthoringCPU(JitTestCase):
    device = "cpu"

    def rand(self, *args, dtype=torch.float32, **kwargs):
        return torch.randint(0, 100, args, dtype=dtype, device=self.device, **kwargs)

    def check(self, *args):
        result_aten = pointwise_fn(*args)
        result_nnc = nnc_pointwise_fn(*args)
        self.assertEqual(result_nnc.dtype, result_aten.dtype)
        self.assertEqual(result_nnc.size(), result_aten.size())
        self.assertEqual(result_nnc.stride(), result_aten.stride())
        self.assertEqual(result_nnc.requires_grad, result_aten.requires_grad)
        torch.testing.assert_allclose(result_aten, result_nnc)

    def test_broadcast1(self):
        self.check(self.rand(8, 16), self.rand(1))

    def test_broadcast2(self):
        self.check(self.rand(8, 1), self.rand(1, 8))

    def test_transposed1(self):
        self.check(self.rand(7, 3), self.rand(3, 7).transpose(0, 1))

    def test_transposed2(self):
        self.check(self.rand(8, 16).transpose(0, 1), self.rand(8, 16).transpose(0, 1))

    def test_slice1(self):
        self.check(self.rand(20, 20, 2)[:8, :16, 0], self.rand(8, 16))

    def test_slice2(self):
        self.check(self.rand(8, 16, 2)[:, :, 0], self.rand(8, 16, 2)[:, :, 0])

    def test_issue57611(self):
        self.check(self.rand(1, 32, 32, 2), self.rand(2, 1, 1, 2))

    def test_float_double(self):
        self.check(self.rand(8, 16), self.rand(8, 16, dtype=torch.float64))

    def test_int_long(self):
        self.check(
            self.rand(8, 16, dtype=torch.int32), self.rand(1, 1, dtype=torch.int64)
        )

    def test_float_int(self):
        self.check(
            self.rand(8, 16, dtype=torch.float32), self.rand(8, 16, dtype=torch.int32)
        )

    @unittest.skipIf(not HAS_SYMPY, "currently requires sympy")
    def test_requires_grad(self):
        self.check(self.rand(4, 2), self.rand(4, 2, requires_grad=True))

    @unittest.skipIf(not HAS_SYMPY, "currently requires sympy")
    def test_backwards(self):
        def grads(fn):
            a = self.rand(4, 2, requires_grad=True)
            b = self.rand(4, 2, requires_grad=True)
            c = self.rand(4, 2)
            d = self.rand(4, 2)
            fn(fn(a, fn(b, c)), d).sum().backward()
            return a.grad, b.grad

        a1, b1 = grads(pointwise_fn)
        a2, b2 = grads(nnc_pointwise_fn)
        torch.testing.assert_allclose(a1, a2)
        torch.testing.assert_allclose(b1, b2)

    def test_torch_function(self):
        self.check(self.rand(10), TorchFunctionExample())

    def test_fx_trace(self):
        def example(x):
            return custom1(custom2(x))

        graph = fx.symbolic_trace(example)
        self.assertIn("custom1", graph.code)
        self.assertIn("custom2", graph.code)
        x = torch.randn(8, device=self.device)
        torch.testing.assert_allclose(x + 3, graph(x))

    def test_unary_ops(self):
        unary_operators = [
            torch.sin,
            torch.cos,
            torch.tan,
            torch.asin,
            torch.acos,
            torch.atan,
            torch.sinh,
            torch.cosh,
            torch.tanh,
            torch.sigmoid,
            torch.exp,
            torch.expm1,
            torch.log,
            torch.log2,
            torch.log10,
            torch.log1p,
            torch.erf,
            torch.erfc,
            torch.sqrt,
            torch.rsqrt,
            # TODO - Fails backward pass because of missing ops
            # torch.ceil,  # Missing empty_like
            # torch.floor,
            # torch.round,
            # torch.trunc,
            # torch.abs,   # Missing sgn
            # torch.lgamma, # missing digamma
            # TODO - Failure in generating loop nests here for the following ops
            # torch.frac,
            # torch.isnan,
        ]

        for unary_op in unary_operators:
            fn = lambda x: unary_op(x)
            pointwise_fn = pointwise_operator(fn)
            ref_a = torch.rand(2, 3, requires_grad=True)
            res_a = ref_a.clone().detach().requires_grad_(True)

            # Check forward
            ref = fn(ref_a)
            res = pointwise_fn(res_a)
            assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)

            # Check gradients
            ref.sum().backward()
            res.sum().backward()
            assert torch.allclose(ref_a.grad, res_a.grad, atol=1e-3, rtol=1e-3)

    def test_binary_ops(self):
        binary_operators = [
            torch.add,
            torch.sub,
            torch.subtract,
            torch.mul,
            torch.multiply,
            torch.divide,
            torch.div,
            torch.atan2,
            # torch.remainder, #TODO - Fails allclose check
            torch.ops.aten.add,
            torch.ops.aten.sub,
            torch.ops.aten.subtract,
            torch.ops.aten.mul,
            torch.ops.aten.multiply,
            torch.ops.aten.divide,
            torch.ops.aten.div,
            torch.ops.aten.atan2,
            # TODO
            # torch.fmod, Autograd does not have backward for fmod
            # torch.pow, Needs better testign as pow2 accepts int arg
        ]
        for binary_op in binary_operators:
            fn = lambda x, y: binary_op(x, y)
            pointwise_fn = pointwise_operator(fn)
            ref_a = torch.rand(2, 3, requires_grad=True)
            ref_b = torch.rand(2, 3, requires_grad=True)
            res_a = ref_a.clone().detach().requires_grad_(True)
            res_b = ref_b.clone().detach().requires_grad_(True)
            # Check forward
            ref = fn(ref_a, ref_b)
            res = pointwise_fn(res_a, res_b)
            assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)

            # Check gradients
            ref.sum().backward()
            res.sum().backward()
            assert torch.allclose(ref_a.grad, res_a.grad, atol=1e-3, rtol=1e-3)
            assert torch.allclose(ref_b.grad, res_b.grad, atol=1e-3, rtol=1e-3)

    def test_bias_gelu(self):
        def bias_gelu(bias, y):
            x = bias + y
            return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

        pointwise_fn = pointwise_operator(bias_gelu)
        ref_bias = torch.rand(64, 768, requires_grad=True)
        res_bias = ref_bias.clone().detach().requires_grad_(True)
        ref_y = torch.rand(64, 768, requires_grad=True)
        res_y = ref_y.clone().detach().requires_grad_(True)

        # Check forward
        ref = bias_gelu(ref_bias, ref_y)
        res = pointwise_fn(res_bias, res_y)
        assert torch.allclose(ref, res, atol=1e-3, rtol=1e-3)

        # Check gradients
        ref.sum().backward()
        res.sum().backward()
        assert torch.allclose(ref_bias.grad, res_bias.grad, atol=1e-3, rtol=1e-3)
        assert torch.allclose(ref_y.grad, res_y.grad, atol=1e-3, rtol=1e-3)


class TestOperatorAuthoringGPU(TestOperatorAuthoringCPU):
    device = "cuda"


if not LLVM_ENABLED:
    TestOperatorAuthoringCPU = None  # noqa: F811

# TODO: TestOperatorAuthoringGPU is disabled because it fails on CUDAs.
# if not torch.cuda.is_available():
TestOperatorAuthoringGPU = None  # noqa: F811

if __name__ == "__main__":
    run_tests()
