import torch
from functorch import make_fx
from torch.profiler import profile, ProfilerActivity

from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.fx.passes.tools_common import legalize_graph
import operator


def is_fused_node(node):
    return node.op == "call_module" and "fused_" in node.target

def get_users(node):
    # get the users of a node in fused graph
    # the user might use the output of ndoe through getitem
    users = set()
    for user_node in node.users:
        if user_node.target == operator.getitem: #  TODO: any other possible skips?
            users = users.union(set(user_node.users.keys()))
        elif user_node.op != 'output':
            users.add(user_node)
    return users


def get_fused_node_pairs(fused_graph):
    # get pairs of fused node that are (parent, children) relationship in graph
    # the two (parent, children) nodes might have an getitem node between them
    fused_node_pairs = []
    for node in fused_graph.graph.nodes:
        if(is_fused_node(node)):
            users = get_users(node)
            pairs = [(node, user_node) for user_node in users if (is_fused_node(user_node))]
            fused_node_pairs.extend(pairs)
    return fused_node_pairs


def check_remat_orign(node_pair, candidates_names, node_users_map):
    # check whether we should rematerilize node_pair[0] in node_pair[1]
    # candidate names is all node names in original graph that are fusable
    # node_users_map is a map from nodes to their users in the original graph

    node_origin = node_pair[0]
    node_dest = node_pair[1]
    module_origin = getattr(fused_graph, node_origin.name)
    module_dest = getattr(fused_graph, node_dest.name)
        
    # get number of writes reduced if copy all nodes from orig to dest
    # look at the users in traced graph, check how many output args have users in dest, but no
    # un-fusable user
    orig_node_names = set()
    orig_placeholder_node_names = set()
    dest_node_names = set()
    add_num_placeholder = 0
    remove_num_placeholder = 0
    for node in module_origin.graph.nodes:
        if node.op == "placeholder":
            add_num_placeholder += 1
            orig_placeholder_node_names.add(node.name)
        elif node.op != "output":
            orig_node_names.add(node.name)
        
    for node in module_dest.graph.nodes:
        if node.op == "placeholder":
            if node.name in orig_node_names or node.name in orig_placeholder_node_names:
                remove_num_placeholder += 1
        elif node.op != "output":
            dest_node_names.add(node.name)

    delta_read = add_num_placeholder - remove_num_placeholder

    # get the number of writes reduced if we remateriliaze origin
    delta_write = 0
    for name in orig_node_names:
        local_count = 0
        for node in node_users_map[name]:
            if node.name not in candidates_names: # must pass result to a non-fusable operator
                local_count = 0 # cannot reduce this write
                break
            if node.name in dest_node_names:
                local_count += 1
        delta_write -= local_count
    print(delta_write + delta_read)
    return delta_write + delta_read < 0



def f(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + c + e + f

traced = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
node_users_map = {node.name: set(node.users.keys()) for node in traced.graph.nodes }

supported_ops = NvFuserOperatorSupport()
partitioner = CapabilityBasedPartitioner(traced, supported_ops)
candidates = partitioner.get_candidates()
partitions = partitioner.partition(candidates)
fused_graph = partitioner.fuse_partitions(partitions) # modifed traced in-place

candidates_names = set([node.name for node in candidates])


a = torch.rand(5)
expected = f(a)
result = fused_graph(a)
torch.testing.assert_close(expected, result)

fused_node_pairs = get_fused_node_pairs(fused_graph)
print(fused_node_pairs)
node_pair = fused_node_pairs[0]
result = check_remat_orign(node_pair, candidates_names, node_users_map)
print(result)