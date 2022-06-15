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
from remat_utils import get_users, get_fused_node_pairs, get_num_changes


def f(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + c + e + f
# === fused graph graph():
#     %a_1 : [#users=1] = placeholder[target=a_1]
#     %fused_1 : [#users=2] = call_module[target=fused_1](args = (%a_1,), kwargs = {})
#     %getitem : [#users=1] = call_function[target=operator.getitem](args = (%fused_1, 0), kwargs = {})
#     %getitem_1 : [#users=2] = call_function[target=operator.getitem](args = (%fused_1, 1), kwargs = {})
#     %clone : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%getitem_1,), kwargs = {})
#     %fused_0 : [#users=1] = call_module[target=fused_0](args = (%clone, %getitem, %getitem_1), kwargs = {})
#     return fused_0

def f1(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + e + f
# === fused graph graph():
#     %a_1 : [#users=1] = placeholder[target=a_1]
#     %fused_1 : [#users=2] = call_module[target=fused_1](args = (%a_1,), kwargs = {})
#     %getitem : [#users=1] = call_function[target=operator.getitem](args = (%fused_1, 0), kwargs = {})
#     %getitem_1 : [#users=1] = call_function[target=operator.getitem](args = (%fused_1, 1), kwargs = {})
#     %clone : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%getitem_1,), kwargs = {})
#     %fused_0 : [#users=1] = call_module[target=fused_0](args = (%clone, %getitem), kwargs = {})
#     return fused_0

def f2(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return e + f
# === fused graph graph():
#     %a_1 : [#users=1] = placeholder[target=a_1]
#     %fused_1 : [#users=2] = call_module[target=fused_1](args = (%a_1,), kwargs = {})
#     %clone : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%fused_1,), kwargs = {})
#     %fused_0 : [#users=1] = call_module[target=fused_0](args = (%clone, %fused_1), kwargs = {})
#     return fused_0

# three fused groups
def f3(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.clone(b)
    h = e + d + b + c
    i = h.clone()
    j = i.relu()
    return j + h

# assignment {add_3: 0, relu_1: 0, add_2: 1, add_1: 1, add: 1, relu: 2, cos: 2}
# === fused graph graph():
#     %a_1 : [#users=1] = placeholder[target=a_1]
#     %fused_2 : [#users=2] = call_module[target=fused_2](args = (%a_1,), kwargs = {})
#     %getitem : [#users=2] = call_function[target=operator.getitem](args = (%fused_2, 0), kwargs = {})
#     %getitem_1 : [#users=2] = call_function[target=operator.getitem](args = (%fused_2, 1), kwargs = {})
#     %clone : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%getitem_1,), kwargs = {})
#     %clone_1 : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%getitem,), kwargs = {})
#     %fused_1 : [#users=2] = call_module[target=fused_1](args = (%clone_1, %clone, %getitem, %getitem_1), kwargs = {})
#     %clone_2 : [#users=1] = call_function[target=torch.ops.aten.clone](args = (%fused_1,), kwargs = {})
#     %fused_0 : [#users=1] = call_module[target=fused_0](args = (%clone_2, %fused_1), kwargs = {})
#     return fused_0

# three fused groups
def f4(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.clone(b)
    h = e + d + b + c
    i = h.clone()
    j = i.relu()
    return j + h + b


# three fused groups
def f5(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    h = d + b + c
    i = h.clone()
    j = i.relu()
    return j + h + b


def f6(a):
    b = a.relu()
    c = a.cos()
    d = b + c
    e = b.relu()
    f = e + b
    g = f.clone()
    h = g + b
    return h + d

def get_fused_graph(f):
    traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
    supported_ops = NvFuserOperatorSupport()
    partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
    candidates = partitioner.get_candidates()
    partitions = partitioner.partition(candidates)
    fused_graph = partitioner.fuse_partitions(partitions) # modifed traced in-place
    return fused_graph

def get_fused_graph_for_num_changes(f):
    traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
    node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes }
    supported_ops = NvFuserOperatorSupport()
    partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
    candidates = partitioner.get_candidates()
    partitions = partitioner.partition(candidates)
    fused_graph = partitioner.fuse_partitions(partitions) # modifed traced in-place
    return node_users_map, fused_graph

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

class GetFusedNodePairsTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.fused_graph = get_fused_graph(f)
        cls.fused_graph_1 = get_fused_graph(f1)
        cls.fused_graph_2 = get_fused_graph(f2)
        cls.fused_graph_3 = get_fused_graph(f3)
        cls.fused_graph_4 = get_fused_graph(f4)

    def test_only_one_pair(self):
        pairs = get_fused_node_pairs(self.fused_graph)
        pair_names = [(pair[0].name, pair[1].name) for pair in pairs]
        expected_pairs = [["fused_1", "fused_0"]]
        self.assertEqual(pair_names, expected_pairs)
    
        pairs = get_fused_node_pairs(self.fused_graph_1)
        pair_names = [(pair[0].name, pair[1].name) for pair in pairs]
        self.assertEqual(pair_names, expected_pairs)

    def test_no_pair(self):
        pairs = get_fused_node_pairs(self.fused_graph_2)
        pair_names = [(pair[0].name, pair[1].name) for pair in pairs]
        expected_pairs = []
        self.assertEqual(pair_names, expected_pairs)

    def test_two_pairs(self):
        pairs = get_fused_node_pairs(self.fused_graph_3)
        pair_names = [(pair[0].name, pair[1].name) for pair in pairs]
        expected_pairs = [["fused_2", "fused_1"], ["fused_1", "fused_0"]]
        self.assertEqual(pair_names, expected_pairs)

    def test_multiple_pairs(self):
        pairs = get_fused_node_pairs(self.fused_graph_4)
        pair_names = [(pair[0].name, pair[1].name) for pair in pairs]
        expected_pairs = [["fused_2", "fused_1"], ["fused_2", "fused_0"], ["fused_1", "fused_0"]]
        self.assertEqual(pair_names, expected_pairs)


class GetNumChangesTestCase(TestCase):

    def test_user_within_origin_module(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f)
        name_to_node = {node.name:node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
        add_num_placeholder, remove_num_placeholder, delta_write \
            = get_num_changes(node_pair, node_users_map, fused_graph)
        self.assertEqual(add_num_placeholder, 1, f"add_num_placeholder is {add_num_placeholder}")
        self.assertEqual(remove_num_placeholder, 2, f"remove_num_placeholder is {remove_num_placeholder}")
        self.assertEqual(delta_write, -1, f"delta_write is {delta_write}")

    def test_multiple_fused_groups(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f3)
        name_to_node = {node.name:node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
        add_num_placeholder, remove_num_placeholder, delta_write \
            = get_num_changes(node_pair, node_users_map, fused_graph)
        self.assertEqual(add_num_placeholder, 4, f"add_num_placeholder is {add_num_placeholder}")
        self.assertEqual(remove_num_placeholder, 1, f"remove_num_placeholder is {remove_num_placeholder}")
        self.assertEqual(delta_write, 0, f"delta_write is {delta_write}")

    def test_share_placeholders(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f4)
        name_to_node = {node.name:node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
        add_num_placeholder, remove_num_placeholder, delta_write \
            = get_num_changes(node_pair, node_users_map, fused_graph)
        self.assertEqual(add_num_placeholder, 4, f"add_num_placeholder is {add_num_placeholder}")
        self.assertEqual(remove_num_placeholder, 2, f"remove_num_placeholder is {remove_num_placeholder}")
        self.assertEqual(delta_write, 0, f"delta_write is {delta_write}")

    def test_write_to_non_fusable_and_other_groups(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f4)
        name_to_node = {node.name:node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_2"], name_to_node["fused_1"])
        add_num_placeholder, remove_num_placeholder, delta_write \
            = get_num_changes(node_pair, node_users_map, fused_graph)
        self.assertEqual(add_num_placeholder, 1, f"add_num_placeholder is {add_num_placeholder}")
        self.assertEqual(remove_num_placeholder, 2, f"remove_num_placeholder is {remove_num_placeholder}")
        self.assertEqual(delta_write, 0, f"delta_write is {delta_write}")

    def test_write_to_other_groups(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f5)
        name_to_node = {node.name:node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_2"], name_to_node["fused_1"])
        add_num_placeholder, remove_num_placeholder, delta_write \
            = get_num_changes(node_pair, node_users_map, fused_graph)
        self.assertEqual(add_num_placeholder, 1, f"add_num_placeholder is {add_num_placeholder}")
        self.assertEqual(remove_num_placeholder, 2, f"remove_num_placeholder is {remove_num_placeholder}")
        self.assertEqual(delta_write, 0, f"delta_write is {delta_write}")

    def test_multiple_users_in_origin_group(self):
        node_users_map, fused_graph = get_fused_graph_for_num_changes(f6)
        name_to_node = {node.name:node for node in fused_graph.graph.nodes}
        node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
        add_num_placeholder, remove_num_placeholder, delta_write \
            = get_num_changes(node_pair, node_users_map, fused_graph)
        self.assertEqual(add_num_placeholder, 1, f"add_num_placeholder is {add_num_placeholder}")
        self.assertEqual(remove_num_placeholder, 2, f"remove_num_placeholder is {remove_num_placeholder}")
        self.assertEqual(delta_write, -1, f"delta_write is {delta_write}")

if __name__ == "__main__":
    run_tests()
