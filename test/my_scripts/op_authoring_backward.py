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

def pointwise_fn(a, b, c):
    return (a * b * c)


nnc_pointwise_fn = pointwise_operator(pointwise_fn)


a = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, 3, requires_grad=True)
c = torch.randn(2, 3, requires_grad=True)

d = nnc_pointwise_fn(a, b, c)
loss = d.sum()
loss.backward()
print(a.grad)