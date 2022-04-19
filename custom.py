import torch
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
import functorch
from functorch import grad, grad_and_value, vmap
import functorch._C as _C
from functools import partial
from functorch._src.pytree_hacks import tree_map2


def generate_autograd_function(level, fwd_fn, bwd_fn):
    if level == 0:
        class CustomFuncA(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                result, saved = fwd_fn(*args)
                flat_saved, saved_spec = tree_flatten(saved)
                ctx.saved_spec = saved_spec
                ctx.save_for_backward(*flat_saved)
                return result, saved

            @staticmethod
            def backward(ctx, *args):
                print('custom_vjp A backward')
                saved = tree_unflatten(ctx.saved_tensors, ctx.saved_spec)
                return bwd_fn(*args, saved)
        return CustomFuncA

    class CustomFuncB(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            def unwrap(x):
                return _C._unwrap_for_grad(x, level)

            def wrap(x):
                return _C._wrap_for_grad(x, level)

            unwrapped_args = tree_map(unwrap, args)

            dl = _C.popTop()
            try:
                with torch.enable_grad():
                    result, saved = custom_vjp(fwd_fn, bwd_fn, *unwrapped_args)
            finally:
                _C.pushTop(dl)

            result, saved = tree_map(wrap, (result, saved))

            flat_saved, saved_spec = tree_flatten(saved)
            ctx.saved_spec = saved_spec

            ctx.save_for_backward(*flat_saved)
            return result, saved

        @staticmethod
        def backward(ctx, *args):
            print('custom_vjp B backward')
            saved = tree_unflatten(ctx.saved_tensors, ctx.saved_spec)
            return bwd_fn(*args, saved)

    return CustomFuncB


def custom_vjp(fwd_fn, bwd_fn, *args):
    if not _C.are_transforms_active():
        print('[torch] custom_vjp')
        CustomFunc = generate_autograd_function(0, fwd_fn, bwd_fn)
        return CustomFunc.apply(*args)

    layer, level = _C.top_layer()
    if layer == '':
        print('[torch] custom_vjp')
        CustomFunc = generate_autograd_function(0, fwd_fn, bwd_fn)
        return CustomFunc.apply(*args)
    elif layer == 'vmap':
        assert len(args) == 1
        assert isinstance(args[0], torch.Tensor)
        x, bdim = _C.unpackBatched(args[0], level)
        dl = _C.popTop()

        side_channel = None

        def batch_fwd(f):
            def inner(x):
                new_level = _C.pushTop(dl)
                try:
                    x = _C._add_batch_dim(x, bdim, new_level)
                    results = f(x)
                    flat_results, results_spec = tree_flatten(results)
                    flat_results_and_bdims = [_C.unpackBatched(r, new_level) for r in flat_results]
                    flat_tensors, flat_bdims = zip(*flat_results_and_bdims)
                    nonlocal side_channel
                    side_channel = flat_bdims
                    return tree_unflatten(flat_tensors, results_spec)
                finally:
                    _C.popTop()

            return inner

        def wrap(tensor, bdim):
            return _C._add_batch_dim(tensor, bdim, level)

        def batch_bwd(f):
            def inner(gO, gx, x):
                new_level = _C.pushTop(dl)
                try:
                    x = _C._add_batch_dim(x, bdim, new_level)
                    grads = gO, gx
                    grads = tree_map2(wrap, grads, side_channel)
                    gO, gx = grads
                    gx_new = f(gO, gx, x)
                    gx_new, gx_bdim = _C.unpackBatched(gx_new, new_level)
                    gx_new = gx_new.movedim(gx_bdim, bdim)
                    return gx_new
                finally:
                    _C.popTop()
            return inner

        try:
            results = custom_vjp(batch_fwd(fwd_fn), batch_bwd(bwd_fn), x)
        finally:
            _C.pushTop(dl)

        return tree_map2(wrap, results, side_channel)
    elif layer == 'grad':
        print('[grad] custom_vjp')
        CustomFunc = generate_autograd_function(level, fwd_fn, bwd_fn)
        result = CustomFunc.apply(*args)
        return result
    else:
        raise NotImplementedError('nyi')


def f_fwd(x):
    result = torch.sin(x)
    return result, x


def f_bwd(gO, gx, x):
    # Should be cosine but we're proving a point
    return gO * 2 * x + gx


MySin = lambda *args: custom_vjp(f_fwd, f_bwd, *args)[0]

print('*' * 80)
x = torch.tensor(0.123)
y = MySin(x)
assert torch.allclose(y, x.sin())

print('*' * 80)
x = torch.tensor(0.123, requires_grad=True)
y = MySin(x)
assert torch.allclose(y, x.sin())
gx, = torch.autograd.grad(y, x)
assert torch.allclose(gx, 2 * x)

print('*' * 80)
x = torch.tensor(0.123)
result = grad(MySin)(x)
assert torch.allclose(result, 2 * x)

print('*' * 80)
x = torch.tensor(0.123, requires_grad=True)
gx, y = grad_and_value(MySin)(x)
assert torch.allclose(gx, 2 * x)
ggx, = torch.autograd.grad(gx, x)
assert torch.allclose(ggx, torch.tensor(2.))

print('*' * 80)
x = torch.tensor(0.123)
ggx = grad(grad(MySin))(x)
assert torch.allclose(ggx, torch.tensor(2.))

print('*' * 80)

class NaiveMySinFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        with torch.enable_grad():
            result, saved = f_fwd(*args)
            flat_saved, saved_spec = tree_flatten(saved)
            ctx.saved_spec = saved_spec
            ctx.save_for_backward(*flat_saved)
            return result, saved

    @staticmethod
    def backward(ctx, *args):
        saved = tree_unflatten(ctx.saved_tensors, ctx.saved_spec)
        return f_bwd(*args, saved)

x = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
NaiveMySin = lambda x: NaiveMySinFunc.apply(x)[0]

# Note that after vmap the backward is sin backward...
y = vmap(NaiveMySin)(x)
print(y)

# and that gx is not x * 2
gx = torch.autograd.grad(y.sum(), x)
print(gx)

print('*' * 80)
x = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
y = vmap(MySin)(x)
assert torch.allclose(y, x.sin())
L = y.sum()
gx, = torch.autograd.grad(L, x)
print(gx)
assert torch.allclose(gx, 2 * x)

print('*' * 80)

def g(x):
    return vmap(MySin)(x).sum()

gx = grad(g)(x)
assert torch.allclose(gx, 2 * x)
