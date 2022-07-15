import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils._pytree as pytree

from functorch import grad, vmap
from functorch._src.make_functional import extract_weights, extract_buffers, load_buffers, _del_nested_attr, _set_nested_attr, make_functional_with_buffers, _swap_state
from torch.testing._internal.common_utils import disable_functorch


class PyTreeModule(nn.Module):
    def __init__(self, model, prototype=None):
        super(PyTreeModule, self).__init__()
        self.tree = model

        # TODO: how to make model_prototype not show up?
        if prototype is None:
            tree_prototype = copy.deepcopy(model)
            param_names, _ = safe_two_unzip(tree_prototype.named_parameters())
            buffer_names, _ = safe_two_unzip(tree_prototype.named_buffers())
            attrs = [n.split(".") for n in (param_names + buffer_names)]
            for name in attrs:
                _del_nested_attr(tree_prototype, name)
            self.tree_prototype = tree_prototype
        else:
            self.tree_prototype = prototype

    def forward(self, *args, **kwargs):
        return self.tree(*args, **kwargs)


def safe_two_unzip(lst):
    lst = list(lst)
    if not lst:
        return (), () 
    return zip(*lst)


def pytree_module_flatten(model):
    # TODO: calling .state_dict does weird things :/
    param_names, params = safe_two_unzip(model.tree.named_parameters())
    buffer_names, buffers = safe_two_unzip(model.tree.named_buffers())
    return params + buffers, (model.tree_prototype, param_names, buffer_names)


def load_weights(mod: nn.Module, names, params, as_params=False) -> None:
    for name, p in zip(names, params):
        if as_params and p is not None:
            p = nn.Parameter(p)
        _set_nested_attr(mod, name.split("."), p)


def pytree_module_unflatten(values, context):
    tree_prototype, param_names, buffer_names = context
    model = copy.deepcopy(tree_prototype)
    params = values[:len(param_names)]
    buffers = values[len(param_names):]
    load_weights(model, param_names, params, as_params=True)
    load_buffers(model, buffer_names, buffers)
    result = PyTreeModule(model, tree_prototype)
    return result


pytree._register_pytree_node(
    PyTreeModule,
    pytree_module_flatten,
    pytree_module_unflatten)


# effectively equinox.partition
def partition(tree, filter_fn):
    values, spec = pytree.tree_flatten(tree)
    left = [v if filter_fn(v) else None for v in values]
    right = [v if not filter_fn(v) else None for v in values]
    return pytree.tree_unflatten(left, spec), pytree.tree_unflatten(right, spec)


# effectively equinox.combine
def combine(left, right):
    left_values, spec = pytree.tree_flatten(left)
    right_value, _ = pytree.tree_flatten(right)

    def pick(lvalue, rvalue):
        if lvalue is None:
            return rvalue
        if rvalue is None:
            return lvalue
        assert False

    values = [pick(l, r) for l, r in zip(left_values, right_value)]
    return pytree.tree_unflatten(values, spec)


##############################################
#
# Let's test this out!
#
##############################################

# Case 1: model with no buffers
class ThreeLayerNet(nn.Module):
    def __init__(self):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(1, 40)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(40, 40)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = PyTreeModule(ThreeLayerNet())

def compute_loss(model, x, t):
    y = model(x)
    return F.mse_loss(y, t)

# Standard gradient computation
x = torch.randn(10, 1)
t = torch.randn(10, 1)
grads = grad(compute_loss)(model, x, t)
print(grads.tree.fc1.bias.shape)

# Per-sample gradients
x = torch.randn(10, 1)
t = torch.randn(10, 1)
per_sample_grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(model, x, t)
print(per_sample_grads.tree.fc1.bias.shape)

# Case 2: model with buffers
class ThreeLayerNet(nn.Module):
    def __init__(self):
        super(ThreeLayerNet, self).__init__()
        self.fc1 = nn.Linear(1, 40)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(40, 40)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(40, 1)
        self.register_buffer('foo', torch.randn(3))

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = PyTreeModule(ThreeLayerNet())

def compute_loss(model_params, model_buffers, x, t):
    model = combine(model_params, model_buffers)
    y = model(x)
    return F.mse_loss(y, t)

def requires_grad(x):
    return x.requires_grad

# Standard gradient computation
x = torch.randn(10, 1)
t = torch.randn(10, 1)
model_params, model_buffers = partition(model, requires_grad)
grads = grad(compute_loss)(model_params, model_buffers, x, t)
print(grads.tree.fc1.bias.shape)
print(grads.tree.foo)
