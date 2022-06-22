from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.backends.nvfuser.operator_support import NvFuserOperatorSupport
from torch.fx.passes.tools_common import legalize_graph
import operator
import math

from .utilities import _size_of, get_cut_nodes_from_partition, draw_nx_graph


num_group_remat = 0  # used for analytical purpose
memory_reduced = 0
# no_weight_nodes = {}

def is_fused_node(node):
    return node.op == "call_module" and "fused_" in node.target


def get_users(node):
    # get the users of a node in fused graph
    # the user might use the output of node through getitem
    users = set()
    for user_node in node.users:
        if user_node.target == operator.getitem:  # TODO: any other possible skips?
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


def check_remat_orign(node_pair, node_users_map, fused_graph):
    return False

def copy_all_nodes(node_pair, fused_graph, name_to_node):
    pass

def get_weight(node):
    weight = 0
    if 'tensor_meta' in node.meta:
        weight = _size_of(node.meta['tensor_meta'])
    return weight


def find_min_cut(node_pair, node_users_map, fused_graph):
    """
        The mincut value is the cost of reading/writing between the two fusion groups
    """

    try:
        import networkx as nx 
    except ImportError:
        raise RuntimeError("Need networkx installed to perform smart recomputation heuristics")
    nx_graph = nx.DiGraph()
    node_origin = node_pair[0]
    node_dest = node_pair[1]
    module_origin = getattr(fused_graph, node_origin.name)
    module_dest = getattr(fused_graph, node_dest.name)

    dest_placeholder_names = set(node.name for node in module_dest.graph.nodes if node.op == "placeholder")

    # used to check if a node has users in dest. The user node in the original graph has the same name as the call_func nodes in dest.
    dest_node_names = set(node.name for node in module_dest.graph.nodes if node.op != "placeholder" and node.op != "output")
    orig_node_names = set(node.name for node in module_origin.graph.nodes if node.op != "placeholder" and node.op != "output")


    for node in module_origin.graph.nodes:
        if node.op == 'output':
            continue

        weight = get_weight(node)

        if node.op == 'placeholder':
            # if need to read this placeholder again
            nx_graph.add_edge("source", node.name+"_out", capacity=weight)  
        elif node.op ==  'call_function':
            # if rematerialize an internal node, need to read and write
            # might not need to add the write cost, because it might be read by other
            # might not need to add the read cost, if already reading it - no need the cost
            # TODO: test case for both
            user_names_set = set({n.name for n in node_users_map[node.name]})
            user_names_outside_set = user_names_set.difference(orig_node_names)
            write_cost = 0 # cost for both read and write because only dest_module is using it

            if weight and user_names_outside_set.issubset(set(dest_node_names)):
                write_cost = weight  
            
            read_cost = weight
            if node.name in dest_placeholder_names:
                nx_graph.add_edge(node.name+"_out", 'sink', capacity=math.inf)

            capacity = write_cost+read_cost
            nx_graph.add_edge(node.name+"_in", node.name+"_out", capacity=capacity)
 
        for user in node.users:
            if user.op != "output":
                nx_graph.add_edge(node.name+"_out", user.name+"_in", capacity=math.inf)

    draw_nx_graph(nx_graph)
    cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")
    print(cut_value, partition)
    cut_nodes = get_cut_nodes_from_partition(partition, nx_graph)
    print(cut_nodes)

    # TODO: check to do nothing if partition has only sink


def get_fused_graph(traced_graph):
    supported_ops = NvFuserOperatorSupport()
    partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
    fused_graph = partitioner.partition_and_fuse()
    return fused_graph


def rematerialize_fused_graph(fused_graph, node_users_map):
    global num_group_remat
    name_to_node = {node.name:node for node in fused_graph.graph.nodes}

    fused_node_pairs = get_fused_node_pairs(fused_graph)
    for node_pair in fused_node_pairs:
        do_remat = check_remat_orign(node_pair, node_users_map, fused_graph)
        if do_remat:
            num_group_remat += 1
            copy_all_nodes(node_pair, fused_graph, name_to_node)
    return fused_graph


def rematerialize(traced_graph):
    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()
    node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes }

    fused_graph = get_fused_graph(traced_graph)
    return rematerialize_fused_graph(fused_graph, node_users_map)

def rematerialize_stat(traced_graph, stat):
    global num_group_remat, memory_reduced
    # global no_weight_nodes
    # no_weight_nodes = {}

    num_group_remat = 0 
    memory_reduced = 0
    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()
    node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes }

    fused_graph = get_fused_graph(traced_graph)
    fused_graph = rematerialize_fused_graph(fused_graph, node_users_map)
    
    stat["num_group_remat"] = num_group_remat
    stat["memory_reduced"] = memory_reduced
    # print(no_weight_nodes)
    return fused_graph