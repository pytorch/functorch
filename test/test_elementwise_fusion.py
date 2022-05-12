import torch
from torch.nn import functional as F
from functorch.compile import elementwise_fusion
from torch.testing._internal.common_utils import TestCase, run_tests
import unittest

# Set inputs
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float
input_shape = [4, 4]
requires_grad = False

@unittest.skipIf(not torch.cuda.is_available(), "CUDA is unavailable")
class TestElementwiseFusion(TestCase):

    def test_unary_ops(self):
        unary_operators = [
            # torch.addcdiv, # 3 inputs
            # torch.addcmul, # 3 inputs
            # torch.angle, # ?
            # torch.bitwise_not, #need bool input
            # torch.clamp_min, # need kwargs
            # torch.clamp, # need kwargs
            # torch.hardshrink, # need decomp
            # torch.lerp, # 3inputs

            # torch.special.entr,
            # torch.special.erfcx,
            # torch.special.i0e,
            # torch.special.i1,
            # torch.special.i1e,
            # torch.special.ndtri,
            # torch.special.xlog1py,
            # torch.special.zeta,
            F.celu,
            F.elu,
            F.gelu,
            # F.glu,      # missing decomp, there is one in torch inductor
            F.hardsigmoid,
            F.hardswish,
            F.hardtanh,
            F.leaky_relu,
            F.mish,
            F.silu,
            F.softplus,
            F.softshrink,

            torch.abs,
            torch.acos,
            torch.acosh,   # need special input range [1, inf)
            torch.asin,
            torch.asinh,
            torch.atan,
            torch.atanh,
            torch.ceil,
            torch.cos,
            torch.cosh,
            torch.deg2rad,
            # torch.digamma,   # skip, too complicated
            torch.erf,
            torch.erfc,
            torch.erfinv,
            torch.exp,
            torch.exp2,
            torch.expm1,
            torch.floor,
            torch.frac,
            # torch.frexp,    # needs 2 outputs
            # torch.i0,     #skip, too complicated
            # torch.isinf,    # returns bool
            # torch.isnan, # failing return bool
            # torch.isneginf,  # failing due to ::numeric_limits<T>::infinity() not a valid cuda code
            # torch.isposinf,  # failing due to ::numeric_limits<T>::infinity() not a valid cuda code
            torch.lgamma,
            torch.log,
            torch.log10,
            torch.log1p,
            torch.log2,
            torch.logit,
            # torch.nan_to_num,  # failing due to ::numeric_limits<T>::infinity() not a valid cuda code
            torch.neg,
            torch.ops.aten.abs,
            torch.ops.aten.acos,
            torch.ops.aten.asin,
            torch.ops.aten.atan,
            torch.ops.aten.ceil,
            torch.ops.aten.cos,
            torch.ops.aten.cosh,
            torch.ops.aten.erf,
            torch.ops.aten.erfc,
            torch.ops.aten.exp,
            torch.ops.aten.expm1,
            torch.ops.aten.floor,
            torch.ops.aten.frac,
            torch.ops.aten.lgamma,
            # torch.ops.aten.log_sigmoid_forward,   # why we have this op???
            torch.ops.aten.log,
            torch.ops.aten.log10,
            torch.ops.aten.log1p,
            torch.ops.aten.log2,
            torch.ops.aten.round,
            torch.ops.aten.rsqrt,
            torch.ops.aten.sigmoid,
            torch.ops.aten.sin,
            torch.ops.aten.sinh,
            torch.ops.aten.sqrt,
            torch.ops.aten.tan,
            torch.ops.aten.tanh,
            torch.ops.aten.trunc,
            torch.rad2deg,
            torch.reciprocal,
            torch.relu,
            torch.round,
            torch.rsqrt,
            # torch.sgn,   # for complex number
            torch.sigmoid,
            torch.sign,
            # torch.signbit,   # output is bool type
            torch.sin,
            torch.sinc,
            torch.sinh,
            torch.sqrt,
            torch.tan,
            torch.tanh,
            torch.trunc,
        ]

        for unary_op in unary_operators:
            def fn(x):
                return unary_op(x)

            jitted_fn = elementwise_fusion(fn)
            a = torch.rand(input_shape, device=device, requires_grad=requires_grad)
            ref = fn(a)
            res = jitted_fn(a)
            assert torch.allclose(ref, res, equal_nan=True)



    def test_binary_ops(self):
        binary_operators = [
            torch.add,
            torch.sub,
            torch.subtract,
            torch.mul,
            torch.multiply,
            torch.divide,
            torch.div,
            torch.fmod,
            torch.pow,
            torch.atan2,
            # torch.remainder, #TODO - Fails allclose check
            torch.ops.aten.add,
            torch.ops.aten.sub,
            torch.ops.aten.subtract,
            torch.ops.aten.mul,
            torch.ops.aten.multiply,
            torch.ops.aten.divide,
            torch.ops.aten.div,
            torch.ops.aten.fmod,
            torch.ops.aten.pow,
            torch.ops.aten.atan2,

            # torch.gcd,          # input needs to be ints, write a speical sample function for this
            # torch.igamma,       # skip, too long
            # torch.igammac,      # skip, too long
            # torch.lcm,          # input needs to be ints, write a speical sample function for this
            # torch.nn.functional.prelu,   # need speical inputs
            torch.nextafter,
            # torch.polar,        # output is complext type, not the same as input type
            torch.heaviside,
            torch.hypot,
            # torch.polygamma,   # skip, too long
            # torch.nn.functional.threshold,    # 3 inputs

            torch.maximum,
            torch.minimum,
            # torch.mvlgamma,    # needs decomp, needs reduction support

            # torch.copysign,
            # torch.floor_divide,
            # torch.ne,
            # torch.xlogy,
            # torch.rsub,

            # torch.bitwise_or,
            # torch.bitwise_and,
            # torch.bitwise_xor,
            # torch.bitwise_left_shift,
            # torch.bitwise_right_shift,


            # torch.gt,             # returns bool
            # torch.le,           return bool
            # torch.ge,
            # torch.lt,
            # torch.eq,
            # torch.logical_and,
            # torch.logical_not,
            # torch.logical_or,
            # torch.logical_xor,

            # torch.ops.aten.rrelu_with_noise,

        ]

        for binary_op in binary_operators:
            def fn(x, y):
                return binary_op(x, y)

            jitted_fn = elementwise_fusion(fn)
            a = torch.rand(input_shape, device=device, requires_grad=requires_grad)
            b = torch.rand(input_shape, device=device, requires_grad=requires_grad)
            ref = fn(a, b)
            res = jitted_fn(a, b)
            assert torch.allclose(ref, res)


if __name__ == "__main__":
    run_tests()













