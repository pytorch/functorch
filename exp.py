from functorch import make_fx
import torch

from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport

from torch.profiler import profile, ProfilerActivity
from torch.fx.passes.fuser_utils import fuse_by_partitions
from remat_utils import rematerialize, copy_all_nodes


def f(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.clone(b)
    h = e + d + b + c
    i = h.clone()
    j = i.relu()
    return j + h

traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))

print("=== traced graph", traced_graph.graph)
node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes }

supported_ops = NvFuserOperatorSupport()
partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
candidates = partitioner.get_candidates()
partitions = partitioner.partition(candidates)
fused_graph = partitioner.fuse_partitions(partitions) # modifed traced in-place

# nodes = {node.name:node for node in traced_graph.graph.nodes}
# paritions = [[nodes["cos"], nodes["relu"], nodes["add"]], [nodes["relu_1"], nodes["relu_2"], nodes["add_1"], nodes["add_2"]]]
# fused_graph = fuse_by_partitions(traced_graph, paritions)

name_to_node = {node.name:node for node in fused_graph.graph.nodes}
# print(name_to_node)
node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
copy_all_nodes(node_pair, fused_graph, name_to_node)
fused_graph.recompile()

# fused_graph = rematerialize(traced_graph)
# print("=== after fused graph", fused_graph.graph)
# print("=== after fused_0 graph", fused_graph.fused_0.graph)
# print("=== after fused_1 graph", fused_graph.fused_1.graph)

a = torch.rand(5)
expected = f(a)
result = fused_graph(a)
torch.testing.assert_close(expected, result)

# fused_graph.fused_1 = torch.jit.script(fused_graph.fused_1)
# fused_graph.fused_0 = torch.jit.script(fused_graph.fused_0)

# inp = torch.randn(2**22, device='cuda')
# for _ in range(5):
#     fused_graph(inp)
# with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
#     for _ in range(5):
#         fused_graph(inp)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))