import torch
from functorch import make_fx
from torch.profiler import profile, ProfilerActivity

from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.fx.passes.tools_common import legalize_graph
import operator

def f(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + c + e + f

traced = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
supported_ops = NvFuserOperatorSupport()
partitioner = CapabilityBasedPartitioner(traced, supported_ops)
candidates = partitioner.get_candidates()
partitions = partitioner.partition(candidates)
fused_graph = partitioner.fuse_partitions(partitions)

a = torch.rand(5)
expected = f(a)
result = fused_graph(a)
torch.testing.assert_close(expected, result)

inp = torch.randn(2**22, device='cuda')

# fused_graph.fused_1 = torch.jit.script(fused_graph.fused_1)
# fused_graph.fused_0 = torch.jit.script(fused_graph.fused_0)
# for _ in range(5):
#     fused_graph(inp)
# with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
#     for _ in range(5):
#         fused_graph(inp)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# script_f = torch.jit.script(fused_graph)
# for _ in range(5):
#     script_f(inp)
# with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
#     for _ in range(5):
#         script_f(inp)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# exit(0)


def copy_all_nodes(module_origin, module_dest, fused_graph, name_to_node):
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
    # print("======= module_origin.graph\n", module_origin.graph)
    # print("======= module_origin.code\n", module_origin.code)

module_origin = fused_graph.fused_1
module_dest = fused_graph.fused_0
name_to_node = {node.name:node for node in fused_graph.graph.nodes}
copy_all_nodes(module_origin, module_dest, fused_graph, name_to_node)
fused_graph.recompile()


result = fused_graph(a)

torch.testing.assert_close(expected, result)

fused_graph.fused_1 = torch.jit.script(fused_graph.fused_1)
fused_graph.fused_0 = torch.jit.script(fused_graph.fused_0)

for _ in range(5):
    fused_graph(inp)
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(5):
        fused_graph(inp)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
