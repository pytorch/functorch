import torch
from functorch import make_fx
from torch.profiler import profile, ProfilerActivity

from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.fx.passes.tools_common import legalize_graph
import operator

from remat_utils import rematerialize

def f(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + c + e + f

traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))
fused_graph = rematerialize(traced_graph)

a = torch.rand(5)
expected = f(a)
result = fused_graph(a)
torch.testing.assert_close(expected, result)

fused_graph.fused_1 = torch.jit.script(fused_graph.fused_1)
fused_graph.fused_0 = torch.jit.script(fused_graph.fused_0)

inp = torch.randn(2**22, device='cuda')
for _ in range(5):
    fused_graph(inp)
with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(5):
        fused_graph(inp)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))