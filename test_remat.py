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

def f1(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + e + f


def f2(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return e + f

def get_fused_graph(f):
        traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
        supported_ops = NvFuserOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
        candidates = partitioner.get_candidates()
        partitions = partitioner.partition(candidates)
        fused_graph = partitioner.fuse_partitions(partitions) # modifed traced in-place
        return fused_graph

class GetUsersTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        fused_graph = get_fused_graph(f)
        cls.name_to_node = {node.name:node for node in fused_graph.graph.nodes}
        fused_graph_1 = get_fused_graph(f1)
        cls.name_to_node_1 = {node.name:node for node in fused_graph_1.graph.nodes}
        fused_graph_2 = get_fused_graph(f2)
        cls.name_to_node_2 = {node.name:node for node in fused_graph_2.graph.nodes}
        

    def test_two_getitem_user(self):
        users = get_users(self.name_to_node["fused_1"])
        users_by_name = set([n.name for n in users])
        expected_users = set(["clone", "fused_0"])
        self.assertEqual(users_by_name, expected_users)

    def test_output_not_in_users(self):
        users = get_users(self.name_to_node["fused_0"])
        users_by_name = set([n.name for n in users])
        expected_users = set([])
        self.assertEqual(users_by_name, expected_users)

    def test_one_getitem_user(self):
        users = get_users(self.name_to_node_1["fused_1"])
        users_by_name = set([n.name for n in users])
        expected_users = set(["clone", "fused_0"])
        self.assertEqual(users_by_name, expected_users)
    
    def test_no_getitem_user(self):
        users = get_users(self.name_to_node_2["fused_1"])
        users_by_name = set([n.name for n in users])
        expected_users = set(["clone"])
        self.assertEqual(users_by_name, expected_users)




if __name__ == "__main__":
    run_tests()
