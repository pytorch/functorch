from functorch import make_fx
import torch

from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport

from torch.profiler import profile, ProfilerActivity
from torch.fx.passes.fuser_utils import fuse_by_partitions


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
# nodes = {node.name:node for node in traced_graph.graph.nodes}
# paritions = [[nodes["cos"], nodes["relu"], nodes["add"]], [nodes["relu_1"], nodes["relu_2"], nodes["add_1"], nodes["add_2"]]]
# fused_graph = fuse_by_partitions(traced_graph, paritions)
print("=== fused graph", fused_graph.graph)
print("=== fused_0 graph", fused_graph.fused_0.graph)
print("=== fused_1 graph", fused_graph.fused_1.graph)


# a = torch.rand(5)
# expected = f(a)
# result = fused_graph(a)
# torch.testing.assert_close(expected, result)

# fused_graph.fused_1 = torch.jit.script(fused_graph.fused_1)
# fused_graph.fused_0 = torch.jit.script(fused_graph.fused_0)

# inp = torch.randn(2**22, device='cuda')
# for _ in range(5):
#     fused_graph(inp)
# with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
#     for _ in range(5):
#         fused_graph(inp)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))