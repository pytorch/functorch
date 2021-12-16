# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils._pytree as _pytree
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from typing import List, Any
import torch
import inspect


for name in dir(torch.return_types):
    if name.startswith('__'):
        continue
    attr = getattr(torch.return_types, name)
    if inspect.isclass(attr):
        return_type_class = attr
        _pytree._register_pytree_node(return_type_class, lambda x: (
            tuple(x), None), lambda x, c, constructor=return_type_class: constructor(x))


# TODO: The following function should only be used with vmap.
def tree_flatten_hack(pytree):
    if _pytree._is_leaf(pytree) and not isinstance(pytree, tuple):
        return [pytree], _pytree.LeafSpec()

    typ = type(pytree)

    flatten_fn = _pytree.SUPPORTED_NODES[typ].flatten_fn
    child_pytrees, context = flatten_fn(pytree)

    # Recursively flatten the children
    result: List[Any] = []
    children_specs: List['TreeSpec'] = []
    for child in child_pytrees:
        flat, child_spec = tree_flatten_hack(child)
        result += flat
        children_specs.append(child_spec)

    return result, _pytree.TreeSpec(typ, context, children_specs)

def tree_map_(fn_, pytree):
    flat_args, _ = tree_flatten(pytree)
    [fn_(arg) for arg in flat_args]
    return pytree

class PlaceHolder():
    def __repr__(self):
        return '*'

def treespec_pprint(spec):
    leafs = [PlaceHolder() for _ in range(spec.num_leaves)]
    result = tree_unflatten(leafs, spec)
    return repr(result)

