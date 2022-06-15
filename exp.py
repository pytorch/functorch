from re import A
from functorch import make_fx
import torch
from torch._decomp import decomposition_table, get_decompositions
import numpy
import torch
import torch.fx as fx
from functorch import make_fx
from functorch.compile import aot_function, print_compile
from torch.fx import symbolic_trace

import operator
from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.fx.passes.tools_common import NodeList, NodeSet, legalize_graph

import unittest
from torch.testing._internal.common_utils import TestCase, run_tests
from remat_utils import get_users


def f(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + c + e + f

traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))

print("=== traced graph", traced_graph.graph)
node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes }

supported_ops = NvFuserOperatorSupport()
partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
candidates = partitioner.get_candidates()
partitions = partitioner.partition(candidates)
fused_graph = partitioner.fuse_partitions(partitions) # modifed traced in-place
print("=== fused graph", fused_graph.graph)
print("=== fused_0 graph", fused_graph.fused_0.graph)
print("=== fused_1 graph", fused_graph.fused_1.graph)


