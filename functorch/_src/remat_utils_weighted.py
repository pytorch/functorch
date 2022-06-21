from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport
from torch.fx.passes.tools_common import legalize_graph
import operator
import math

from .utilities import _size_of

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

def get_delta_write(orig_nodes, node_users_map, dest_node_names):
    """"
    get the number of write changes if we rematerializes nodes in [orig_node_names] to compute
    [dest_node_names].

    [node_users_map] is a map from nodes' names to their user in original un-fused graph

    If a node has any write to an unfusable node, this write cannot be reduced.
    """

    # delta_write = 0
    delta_write_weighted = 0
    orig_node_names_set = set(node.name for node in orig_nodes)
    for node in orig_nodes:
        name = node.name
        # local_count = 0
        local_weighted_count = 0
        user_names_set = set({n.name for n in node_users_map[name]})
        user_names_outside_set = user_names_set.difference(orig_node_names_set)
        if len(user_names_outside_set) > 0 and user_names_outside_set.issubset(set(dest_node_names)):
            # local_count += 1
            weight = 0  # TODO: what to do if no tensor_meta exists?
            if 'tensor_meta' in node.meta:
                weight = _size_of(node.meta['tensor_meta'])
            local_weighted_count += weight
        # delta_write -= local_count
        delta_write_weighted -= local_weighted_count
    return delta_write_weighted


def get_num_changes(node_pair, node_users_map, fused_graph):
    # get the number of read/write changes if we rematerilize node_pair[0] in node_pair[1]
    # candidate names is all node names in original graph that are fusable
    # node_users_map is a map from nodes to their users in the original graph
    # assumption: node_pair[0] must be ancestors of node_pair[1]
    # assumption: nodes in node_pair have modules in fused_graph with same name

    node_origin = node_pair[0]
    node_dest = node_pair[1]
    module_origin = getattr(fused_graph, node_origin.name)
    module_dest = getattr(fused_graph, node_dest.name)

    # get number of writes reduced if copy all nodes from orig to dest
    # look at the users in traced graph, check how many output args have users in dest, but no
    # un-fusable user
    orig_nodes = set()
    orig_node_names = set()
    orig_placeholder_node_names = set()
    dest_node_names = set()
    # add_num_placeholder = 0
    # remove_num_placeholder = 0

    add_placeholder_size = 0
    remove_placeholder_size = 0
    for node in module_origin.graph.nodes:
        # might be overcounting some placeholders that are not neccessary to compute
        # nodes in dest
        if node.op == "placeholder":
            # add_num_placeholder += 1
            orig_placeholder_node_names.add(node.name)
            weight = 0  # TODO: what to do if no tensor_meta exists?
            if 'tensor_meta' in node.meta:
                weight = _size_of(node.meta['tensor_meta'])
            add_placeholder_size += weight
        elif node.op != "output":
            orig_nodes.add(node)
            orig_node_names.add(node.name)

    for node in module_dest.graph.nodes:
        if node.op == "placeholder":
            # avoid double counting placeholders that already exists
            if node.name in orig_node_names or node.name in orig_placeholder_node_names:
                # remove_num_placeholder += 1
                weight = 0  # TODO: what to do if no tensor_meta exists?
                if 'tensor_meta' in node.meta:
                    weight = _size_of(node.meta['tensor_meta'])
                remove_placeholder_size +=  weight
        elif node.op != "output":
            dest_node_names.add(node.name)

    # get the number of writes reduced if we remateriliaze origin
    delta_write = get_delta_write(orig_nodes, node_users_map, dest_node_names)
    return add_placeholder_size, remove_placeholder_size, delta_write


def check_remat_orign(node_pair, node_users_map, fused_graph):
    # check whether we should rematerilize node_pair[0] in node_pair[1]
    # candidate names is all node names in original graph that are fusable
    # node_users_map is a map from nodes to their users in the original graph
    add_num_placeholder, remove_num_placeholder, delta_write = get_num_changes(node_pair, node_users_map, fused_graph)
    delta_read = add_num_placeholder - remove_num_placeholder
    return delta_write + delta_read < 0


def copy_all_nodes(node_pair, fused_graph, name_to_node):
    module_origin = getattr(fused_graph, node_pair[0].name)
    module_dest = getattr(fused_graph, node_pair[1].name)

    # map from placeholder names to dest's args in fused graph
    dest_args = []
    for node in fused_graph.graph.nodes:
        if(node.name == module_dest.name):   
            for arg in node.args:
               dest_args.append(arg)
            break
    old_placeholders = []
    for node in module_dest.graph.nodes:
        if node.op == "placeholder":
            old_placeholders.append(node.name)
    dest_arg_map = dict(zip(old_placeholders, dest_args)) 

    name_to_node_dest = {node.name: node for node in module_dest.graph.nodes}
    # create new nodes in dest
    first_node_dest = None
    for node in module_dest.graph.nodes:
        first_node_dest = node
        break

    env = {}  # map from node in origin to node in dest
    origin_placeholder_map = {}  # map from placeholder name in module_origin.graph to node_pair[0].args
    loc = 0 
    for node in module_origin.graph.nodes:
        if node.op == "placeholder":
            origin_placeholder_map[node.name] = node_pair[0].args[loc]
            loc += 1
        elif node.op == "output":
            continue
        with module_dest.graph.inserting_before(first_node_dest):
            new_node = module_dest.graph.node_copy(node, lambda x: env[x])
            new_node.name = node.name # use the same name such that node can be referenced back to original graph
            env[node] = new_node
            # change the args of nodes in dest to use the new node
            if node.name in name_to_node_dest:
                name_to_node_dest[node.name].replace_all_uses_with(new_node)

    # erase old placeholder nodes and record current active placeholders
    active_placeholders = []
    for node in module_dest.graph.nodes:
        if node.op == "placeholder":
            if len(node.users) == 0:
                module_dest.graph.erase_node(node)
            else:
                active_placeholders.append(node.name)

    legalize_graph(module_dest)
    module_dest.graph.eliminate_dead_code()
    module_dest.graph.lint()

    # change the args of dest node in fused_graph
    # use origin_placeholder_map because the active place_holders 
    # might be in another module, and thus need get_item
    for node in fused_graph.graph.nodes:
        if(node.name == module_dest.name):
            new_args = []
            for name in active_placeholders:
                if name in name_to_node: # name is a node in fused graph
                    new_args.append(name_to_node[name])
                elif name in origin_placeholder_map: # name is a palceholder in origin's module
                    new_args.append(origin_placeholder_map[name])
                else: # name is a palceholder in dest's module
                    new_args.append(dest_arg_map[name])
            node.args = tuple(new_args)
            break
    
    fused_graph.recompile()
    # legalize_graph(fused_graph)  # TODO:why this hang sometimes?
    fused_graph.graph.eliminate_dead_code()
    fused_graph.graph.lint()
    module_dest.recompile() 

    # remove the unsed output to write less
    # Use 0 instead of remove entirely because getitem will index into the outputs
    # Assumption: each module node has a single output node
    # Need to do this after fused_graph.graph.eliminate_dead_code() such that
    # extra getitem operators are removed.
    used_inds = set()

    # need to modify the node in fused_graph, not the node passed in pairs
    for node in fused_graph.graph.nodes:
        if(node.name == module_origin.name):
            for node_user in node.users:
                if node_user.target == operator.getitem:
                    used_inds.add(node_user.args[1])
            break
    for node in module_origin.graph.nodes:
        if node.op == "output":
            if (len(used_inds) == 0 and type(node.args[0] is not tuple)): # only has a single output
                break
            new_args = []
            for i in range(len(node.args[0])):
                if i in used_inds:
                    new_args.append(node.args[0][i]) # still useful
                else:
                    new_args.append(0) # no need to write out
            node.args = tuple([tuple(new_args),])
            break
    module_origin.recompile() 
    fused_graph.recompile()

def get_fused_graph(traced_graph):
    supported_ops = NvFuserOperatorSupport()
    partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
    fused_graph = partitioner.partition_and_fuse()
    return fused_graph


def rematerialize_fused_graph(fused_graph, node_users_map):
    name_to_node = {node.name:node for node in fused_graph.graph.nodes}

    fused_node_pairs = get_fused_node_pairs(fused_graph)
    for node_pair in fused_node_pairs:
        do_remat = check_remat_orign(node_pair, node_users_map, fused_graph)
        if do_remat:
            copy_all_nodes(node_pair, fused_graph, name_to_node)

    return fused_graph


def rematerialize(traced_graph):
    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()
    node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes }

    fused_graph = get_fused_graph(traced_graph)
    return rematerialize_fused_graph(fused_graph, node_users_map)

