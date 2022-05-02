import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Tuple
from enum import Enum
from collections import defaultdict
from torch.utils._pytree import tree_map
import torch._decomp

aten = torch.ops.aten
aten.__origin__ = None

decomposition_table = torch._decomp.decomposition_table


def register_decomposition(aten_op, registry=None):
    def decomposition_decorator(f):
        nonlocal registry
        if registry is None:
            registry = decomposition_table

        def add_op_to_table(aten_op):
            # Converts aten.foo to aten.foo.default
            # Done so I can be lazy and not write default on all of these ops
            if not isinstance(aten_op, torch._ops.OpOverload):
                op_overload = aten_op.default
            else:
                op_overload = aten_op
            registry[op_overload] = f
        # To handle allowing multiple aten_ops at once
        tree_map(add_op_to_table, aten_op)
        return f
    return decomposition_decorator


def get_decompositions(aten_ops: List[torch._ops.OpOverload]):
    packets_to_overloads = defaultdict(list)
    for op in decomposition_table:
        packets_to_overloads[op.overloadpacket].append(op)
    decompositions = {}
    for op in aten_ops:
        if op in packets_to_overloads:
            for op_overload in packets_to_overloads[op]:
                decompositions[op_overload] = decomposition_table[op_overload]
        elif op in decomposition_table:
            decompositions[op] = decomposition_table[op]
    return decompositions

# Decompositions have been ported to torch._decomp inside of PyTorch core. Please port contributions there!
