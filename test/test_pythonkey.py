from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import functools
import itertools
import warnings
import math
from typing import Callable, Type
from torch.testing._internal.common_device_type import instantiate_device_type_tests, \
    skipCUDAIfNoMagma, onlyOnCPUAndCUDA, onlyCPU
import types
from functools import partial

import functorch
from functorch import (
    grad, vjp, vmap, jacrev, grad_and_value,
    make_functional, make_functional_with_buffers, make_fx, nnc_jit
)

# NB: numpy is a testing dependency!
import numpy as np

USE_TORCHVISION = False
try:
    import torchvision
    USE_TORCHVISION = True
except ImportError:
    warnings.warn("Couldn't import torchvision. Some of our tests use it, try "
                  "to install it with commands from pytorch.org, post-fixed with "
                  "`--no-deps` to avoid overwriting the pytorch installation",
                  UserWarning)


class TestPythonKey(TestCase):
    def test_make_fx(self, device):
        def f(x):
            return torch.sin(x)
        inp = torch.randn(3)
        fx_f = make_fx(f)(inp)

        new_inp = torch.randn(3)
        self.assertEqual(fx_f(new_inp), f(new_inp))

    def test_nnc_jit(self, device):
        def f(x):
            return torch.sin(x)

        jit_f = nnc_jit(f)

        inp = torch.randn(3)
        self.assertEqual(jit_f(inp), f(inp))

    def test_nnc_pytrees(self, device):
        def f(x):
            return [torch.sin(x[0])]

        jit_f = nnc_jit(f)

        inp = [torch.randn(3)]
        self.assertEqual(jit_f(inp), f(inp))

    def test_external_calls(self, device):
        def f(a, b):
            return torch.mv(a, b)
        jit_f = nnc_jit(f)
        inp = [torch.randn(3, 3), torch.randn(3)]
        self.assertEqual(jit_f(*inp), f(*inp))

    def test_nnc_passthrough(self, device):
        def f(x, y):
            return x + y, y
        inp = (torch.randn(3), torch.randn(3))
        jit_f = nnc_jit(f)
        self.assertEqual(jit_f(*inp), f(*inp))

        def f(x):
            x['a'] = x['a'] * 2
            return x
        inp = ({'a': torch.randn(3), 'b': torch.randn(3)},)
        jit_f = nnc_jit(f)
        self.assertEqual(jit_f(*inp), f(*inp))

    @unittest.skipIf(not USE_TORCHVISION, "test requires torchvision")
    def test_torchvision(self, device):
        model = torchvision.models.resnet18()
        def f(x):
            return model(x)

        inp = torch.randn(1, 3, 200, 200)
        fx_f = make_fx(f)(inp)
        self.assertEqual(fx_f(inp), f(inp))



only_for = ("cpu")
instantiate_device_type_tests(
    TestPythonKey,
    globals(),
    only_for=only_for,
)


if __name__ == '__main__':
    run_tests()
