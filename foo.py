import torch
import functorch
from torch.utils._pytree import tree_map, tree_flatten
from functools import partial

from functorch import vmap, jacrev

def get_torch_dispatch(subclass):
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # TODO: assumes there are no Tensors in kwargs
        flat_args, _ = tree_flatten(args)
        if not any([isinstance(e, subclass) for e in flat_args]):
            # Redispatch
            return func(*args)

        # Naive batching rule: for-loop + stack
        bdim_size = None
        for e in flat_args:
            if isinstance(e, subclass):
                bdim_size = e.elem.size(e.bdim)

        def get_slice(idx, e):
            return e.elem.select(e.bdim, idx) if isinstance(e, subclass) else e

        results = []
        for i in range(bdim_size):
            sliced_args = tree_map(partial(get_slice, i), args)
            res = func(*sliced_args, **kwargs)
            assert isinstance(res, torch.Tensor)
            results.append(res)

        result = torch.stack(results)
        return subclass(result, 0)

    return __torch_dispatch__

class PythonBatchedTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem', 'bdim']

    @staticmethod
    def __new__(cls, elem, bdim, *args, **kwargs):
        r = torch.Tensor._make_subclass(cls, elem.to('cpu')[0], elem.requires_grad)
        r.elem = elem
        r.bdim = bdim
        r.current_subclass = PythonBatchedTensor
        return r

    def __repr__(self):
        return f"PythonBatchedTensor({self.elem}, {self.bdim})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return get_torch_dispatch(PythonBatchedTensor)(cls, func, types, args, kwargs)

def custom_vmap(f, in_dims):
    def wrapped(*args):
        # Generate a fresh new class on the fly
        class GeneratedPythonBatchedTensor(PythonBatchedTensor):
            @staticmethod
            def __new__(cls, elem, bdim, *args, **kwargs):
                # _make_subclass must take in a normal Tensor?
                dummy_elem = torch.zeros(elem.size(), dtype=elem.dtype, device=elem.device)
                r = torch.Tensor._make_subclass(cls, dummy_elem[0], elem.requires_grad)
                r.elem = elem
                r.bdim = bdim
                return r

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return get_torch_dispatch(GeneratedPythonBatchedTensor)(cls, func, types, args, kwargs)

        subclass = GeneratedPythonBatchedTensor

        def wrap(e, in_dim):
            assert in_dim is None or in_dim == 0
            if in_dim is None:
                return e
            return subclass(e, in_dim)

        batched_args = tuple(wrap(arg, in_dim) for arg, in_dim in zip(args, in_dims))
        functorch._C._enter_custom_transform(subclass)
        try:
            batched_out = f(*batched_args)
        finally:
            functorch._C._exit_custom_transform()
        assert isinstance(batched_out, subclass)
        return batched_out.elem
    return wrapped


x = torch.randn(3, 2, 5, 7)
y = vmap(custom_vmap(vmap(torch.sum), (0,)))(x)
assert torch.allclose(y, x.sum([-1]))

x = torch.randn(3)
y = torch.randn(4)
z = vmap(custom_vmap(torch.mul, (0, None)), (None, 0))(x, y)
assert torch.allclose(z, y.unsqueeze(-1) * x)

x = torch.arange(3)
y = torch.arange(4)
z = custom_vmap(vmap(torch.mul, (0, None)), (None, 0))(x, y)
assert torch.allclose(z, y.unsqueeze(-1) * x)

# The following needed me to hack something inside PyTorch core.
# Up until this point, all of the examples were functorch-only :O.
x = torch.randn(3, 2, 5, 7)
y = custom_vmap(custom_vmap(torch.sum, (0,)), (0,))(x)
assert torch.allclose(y, x.sum([-1, -2]))

x = torch.randn(3, 2, 5, 7)
y = custom_vmap(vmap(custom_vmap(torch.sum, (0,))), (0,))(x)
assert torch.allclose(y, x.sum([-1]))
