from functorch import make_fx
import torch

from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport

from torch.profiler import profile, ProfilerActivity
from torch.fx.passes.fuser_utils import fuse_by_partitions
from functorch._src.remat_utils import rematerialize, copy_all_nodes


# class FxModule(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x_1):
#         clone = torch.ops.aten.clone(x_1)
#         cos = torch.ops.aten.cos(x_1)
#         relu = torch.ops.aten.relu(clone)
#         cos_1 = torch.ops.aten.cos(clone)
#         sin = torch.ops.aten.sin(x_1)
#         relu_1 = torch.ops.aten.relu(cos_1)
#         sin_1 = torch.ops.aten.sin(cos_1)
#         cos_2 = torch.ops.aten.cos(cos_1)
#         relu_2 = torch.ops.aten.relu(sin_1)
#         sin_2 = torch.ops.aten.sin(sin);  sin = None
#         cos_3 = torch.ops.aten.cos(clone)
#         relu_3 = torch.ops.aten.relu(relu)
#         relu_4 = torch.ops.aten.relu(relu);  relu = None
#         clone_1 = torch.ops.aten.clone(cos_1)
#         cos_4 = torch.ops.aten.cos(relu_4)
#         gelu = torch.ops.aten.gelu(cos_2);  cos_2 = None
#         cos_5 = torch.ops.aten.cos(relu_3);  relu_3 = None
#         cos_6 = torch.ops.aten.cos(relu_1)
#         tanh = torch.ops.aten.tanh(gelu)
#         cos_7 = torch.ops.aten.cos(relu_1)
#         sin_3 = torch.ops.aten.sin(tanh);  tanh = None
#         relu_5 = torch.ops.aten.relu(sin_3)
#         clone_2 = torch.ops.aten.clone(gelu);  gelu = None
#         sin_4 = torch.ops.aten.sin(relu_1);  relu_1 = None
#         gelu_1 = torch.ops.aten.gelu(sin_1)
#         tanh_1 = torch.ops.aten.tanh(x_1);  x_1 = None
#         clone_3 = torch.ops.aten.clone(cos_3)
#         relu_6 = torch.ops.aten.relu(sin_1);  sin_1 = None
#         gelu_2 = torch.ops.aten.gelu(sin_4)
#         clone_4 = torch.ops.aten.clone(relu_4);  relu_4 = None
#         relu_7 = torch.ops.aten.relu(cos_4)
#         relu_8 = torch.ops.aten.relu(sin_3);  sin_3 = None
#         gelu_3 = torch.ops.aten.gelu(cos_4)
#         tanh_2 = torch.ops.aten.tanh(relu_7);  relu_7 = None
#         sin_5 = torch.ops.aten.sin(clone_4);  clone_4 = None
#         tanh_3 = torch.ops.aten.tanh(clone);  clone = None
#         clone_5 = torch.ops.aten.clone(cos_6);  cos_6 = None
#         gelu_4 = torch.ops.aten.gelu(tanh_3);  tanh_3 = None
#         clone_6 = torch.ops.aten.clone(relu_6);  relu_6 = None
#         tanh_4 = torch.ops.aten.tanh(cos_4);  cos_4 = None
#         relu_9 = torch.ops.aten.relu(tanh_1)
#         cos_8 = torch.ops.aten.cos(cos);  cos = None
#         relu_10 = torch.ops.aten.relu(gelu_1);  gelu_1 = None
#         clone_7 = torch.ops.aten.clone(tanh_1);  tanh_1 = None
#         cos_9 = torch.ops.aten.cos(cos_3);  cos_3 = None
#         tanh_5 = torch.ops.aten.tanh(cos_1)
#         gelu_5 = torch.ops.aten.gelu(clone_2);  clone_2 = None
#         cos_10 = torch.ops.aten.cos(gelu_5);  gelu_5 = None
#         cos_11 = torch.ops.aten.cos(cos_1);  cos_1 = None
#         tanh_6 = torch.ops.aten.tanh(sin_4);  sin_4 = None
#         return tanh_6


m = FxModule()
# print(m.graph)
traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))

print("=== traced graph", traced_graph.graph)
# node_users_map = {node.name: set(node.users.keys()) for node in traced_graph.graph.nodes }

# supported_ops = NvFuserOperatorSupport()
# partitioner = CapabilityBasedPartitioner(traced_graph, supported_ops)
# candidates = partitioner.get_candidates()
# partitions = partitioner.partition(candidates)
# fused_graph = partitioner.fuse_partitions(partitions) # modifed traced in-place

# nodes = {node.name:node for node in traced_graph.graph.nodes}
# paritions = [[nodes["cos"], nodes["relu"], nodes["add"]], [nodes["relu_1"], nodes["relu_2"], nodes["add_1"], nodes["add_2"]]]
# fused_graph = fuse_by_partitions(traced_graph, paritions)

# name_to_node = {node.name:node for node in fused_graph.graph.nodes}
# # print(name_to_node)
# node_pair = (name_to_node["fused_1"], name_to_node["fused_0"])
# copy_all_nodes(node_pair, fused_graph, name_to_node)
# fused_graph.recompile()

fused_graph = rematerialize(traced_graph)
# print("=== after fused graph", fused_graph.graph)
# print("=== after fused_0 graph", fused_graph.fused_0.graph)
# print("=== after fused_1 graph", fused_graph.fused_1.graph)


# fused_graph.fused_1 = torch.jit.script(fused_graph.fused_1)
# fused_graph.fused_0 = torch.jit.script(fused_graph.fused_0)

# inp = torch.randn(2**22, device='cuda')
# for _ in range(5):
#     fused_graph(inp)
# with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
#     for _ in range(5):
#         fused_graph(inp)
# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))