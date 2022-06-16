import random

import torch
from functorch import make_fx
from torch.profiler import profile, ProfilerActivity

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

    num_fused_group = 0
    for node in fused_graph.graph.nodes:
        if "fused_" in node.name:
            module = getattr(fused_graph, node.name)
            setattr(fused_graph, node.name, torch.jit.script(module) )
            num_fused_group += 1

    # fused_graph.fused_1 = torch.jit.script(fused_graph.fused_1)
    # fused_graph.fused_0 = torch.jit.script(fused_graph.fused_0)

    avg_cuda_time_g = benchmark_GPU_time(fused_graph, inp)

    print(f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}, {num_fused_group}")

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

def frandom(x):
    vals = [x, x]
    ops = [torch.clone, torch.clone, torch.clone, torch.add, torch.add,
         torch.cos, torch.sin, torch.relu, torch.tanh, torch.nn.functional.gelu]
    for _ in range(100):
        op = random.choice(ops)
        if op == torch.add:
            new_val = op(random.choice(vals), random.choice(vals))
        else:
            new_val = op(random.choice(vals))
        vals.append(new_val)
    return vals[-1]

i = 0
profile_function(f"rand_test_{i}", frandom, inp)
i += 1

profile_function(f"rand_test_{i}", frandom, inp)
i += 1

profile_function(f"rand_test_{i}", frandom, inp)
i += 1

profile_function(f"rand_test_{i}", frandom, inp)
i += 1

profile_function(f"rand_test_{i}", frandom, inp)
i += 1