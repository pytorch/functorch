# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils._pytree as _pytree
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from typing import List, Any


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
