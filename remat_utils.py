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


def check_remat_orign(node_pair, candidates_names, node_users_map, fused_graph):
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
    return delta_write + delta_read < 0


def copy_all_nodes(node_pair, fused_graph, name_to_node):
    module_origin = getattr(fused_graph, node_pair[0].name)
    module_dest = getattr(fused_graph, node_pair[1].name)

    name_to_node_dest = {node.name:node for node in module_dest.graph.nodes}
    # create new nodes in dest
    first_node_dest = None
    for node in module_dest.graph.nodes:
        first_node_dest = node
        break

    env = {} # map from node in origin to node in dest
    for node in module_origin.graph.nodes:
        if node.op == "output":
            continue
        with module_dest.graph.inserting_before(first_node_dest):
            new_node = module_dest.graph.node_copy(node, lambda x: env[x])
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
    # print("=====module_dest.graph\n", module_dest.graph)

    # change the args of dest node in fused_graph
    for node in fused_graph.graph.nodes:
        if(node.name == module_dest.name):
            node.args = tuple([name_to_node[name] for name in active_placeholders]) 
            break

    legalize_graph(fused_graph)
    fused_graph.graph.eliminate_dead_code()
    fused_graph.graph.lint()
    # print("======= fused_graph.graph\n", fused_graph.graph)
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
            if (len(used_inds) == 0 and len(node.args[0]) == 1): # only has a single output TODO: check
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

def rematerialize(traced_graph):
    node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes }

    supported_ops = NvFuserOperatorSupport()
    partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
    candidates = partitioner.get_candidates()
    partitions = partitioner.partition(candidates)
    fused_graph = partitioner.fuse_partitions(partitions) # modifed traced in-place

    candidates_names = set([node.name for node in candidates])
    name_to_node = {node.name:node for node in fused_graph.graph.nodes}

    fused_node_pairs = get_fused_node_pairs(fused_graph)
    print(fused_node_pairs)
    for node_pair in fused_node_pairs:
        do_remat = check_remat_orign(node_pair, candidates_names, node_users_map, fused_graph)
        print(do_remat)
        if do_remat:
            copy_all_nodes(node_pair, fused_graph, name_to_node)
            fused_graph.recompile()

    return fused_graph
