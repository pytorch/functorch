import torch
from functorch import make_fx
from torch.profiler import profile, ProfilerActivity

from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.fx.passes.tools_common import legalize_graph
import operator

from functorch._src.remat_utils import rematerialize




def benchmark_GPU_time(f, inp, itr = 5):

    inp = torch.randn(2**22, device='cuda')
    for _ in range(5):
        f(inp)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(itr):
            f(inp)

    timing = prof.key_averages()
    cuda_time_total = 0
    for e in timing:
        cuda_time_total = cuda_time_total + e.cuda_time_total
    return cuda_time_total / itr
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

def profile_function(name, f, inp):
    traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(torch.randn(2))

    script_f = torch.jit.script(traced_graph)
    avg_cuda_time_f = benchmark_GPU_time(script_f, inp)

    fused_graph = rematerialize(traced_graph)

    for node in fused_graph.graph.nodes:
        if "fused_" in node.name:
            module = getattr(fused_graph, node.name)
            setattr(fused_graph, node.name, torch.jit.script(module) )

    # fused_graph.fused_1 = torch.jit.script(fused_graph.fused_1)
    # fused_graph.fused_0 = torch.jit.script(fused_graph.fused_0)

    avg_cuda_time_g = benchmark_GPU_time(fused_graph, inp)

    print(f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}")

g_gpu = torch.Generator(device='cuda')
g_gpu.manual_seed(2147483647)
inp = torch.randn(2**20, device='cuda', generator=g_gpu)

def f(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + c + e + f

profile_function("f", f, inp)