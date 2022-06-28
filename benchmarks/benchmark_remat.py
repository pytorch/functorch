import random

import torch
from functorch import make_fx
import torch.fx as fx
from functorch.compile import draw_graph, ts_compile
from torch.profiler import profile, ProfilerActivity
from torch.fx._symbolic_trace import symbolic_trace
import pickle
import copy

from functorch._src.remat_utils_mincut import rematerialize, get_fused_graph, rematerialize_fused_graph
from functorch._src.compile_utils import strip_overloads, fx_graph_cse

random.seed(1)


def benchmark_GPU_time(f, inp, list_inp, itr = 5):
    if list_inp:
        for _ in range(5):
            f(*inp)
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(itr):
                f(*inp)

        timing = prof.key_averages()
        cuda_time_total = 0
        for e in timing:
            cuda_time_total = cuda_time_total + e.cuda_time_total
        return cuda_time_total / itr

    for _ in range(5):
        f(inp)
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(itr):
            f(inp)

    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

    timing = prof.key_averages()
    cuda_time_total = 0
    for e in timing:
        cuda_time_total = cuda_time_total + e.cuda_time_total
    return cuda_time_total / itr

def profile_graph(traced_graph, inp, list_inp):
    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()
    script_f = ts_compile(traced_graph, inp)
    avg_cuda_time_f = benchmark_GPU_time(script_f, inp, list_inp)

    return avg_cuda_time_f

def profile_fused_graph(fused_graph, inp, list_inp):
    num_fused_group = 0
    for node in fused_graph.graph.nodes:
        if "fused_" in node.name:
            module = getattr(fused_graph, node.name)
            setattr(fused_graph, node.name, torch.jit.script(module) )
            num_fused_group += 1

    if num_fused_group == 0: # no fused group
        script_f = torch.jit.script(fused_graph)
        return benchmark_GPU_time(script_f, inp, list_inp), 0

    avg_cuda_time_g = benchmark_GPU_time(fused_graph, inp, list_inp)
    return avg_cuda_time_g, num_fused_group


def profile_fused(traced_graph, inp, list_inp):
    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()
    fused_graph = get_fused_graph(traced_graph)
    return profile_fused_graph(fused_graph, inp, list_inp)


def profile_rematerialize(traced_graph, inp, list_inp):
    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()
    fused_graph = rematerialize(traced_graph)
    return profile_fused_graph(fused_graph, inp, list_inp)


# TODO: need to add CSE pass
# def profile_module(name, m, inp):
#     traced_graph = symbolic_trace(m)
#     avg_cuda_time_f = profile_graph(traced_graph, inp, True)

#     traced_graph = symbolic_trace(m)
#     fused_graph = get_fused_graph(traced_graph)
#     avg_cuda_time_g, num_fused_group = profile_fused_graph(fused_graph, inp, True)

#     traced_graph = symbolic_trace(m)
#     fused_graph = rematerialize(traced_graph)
#     avg_cuda_time_h, _ = profile_fused_graph(fused_graph, inp, True)
#     print(f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}, {avg_cuda_time_h}, {num_fused_group}")


def profile_function(name, f, inp, list_inp = False):
    traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(inp)
    strip_overloads(traced_graph)

    avg_cuda_time_f = profile_graph(traced_graph, inp, list_inp)

    csed = fx_graph_cse(traced_graph.graph)
    csed_graph =  fx.GraphModule(traced_graph, csed)
    csed_graph_copy = copy.deepcopy(csed_graph)

    avg_cuda_time_g, num_fused_group = profile_fused(csed_graph, inp, list_inp)
    avg_cuda_time_h, _ = profile_rematerialize(csed_graph_copy, inp, list_inp)
    print(f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}, {avg_cuda_time_h}, {num_fused_group}", flush=True)


def f(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    f = torch.relu(e)
    return b + c + e + f


def frandom(x):
    vals = [x, x]
    ops = [torch.clone] * 4 + [torch.add] * 2 + [torch.relu] * 5
    for _ in range(50):
        op = random.choice(ops)
        if op == torch.add:
            new_val = op(random.choice(vals), random.choice(vals))
        else:
            new_val = op(random.choice(vals))
        vals.append(new_val)
    return sum(vals)

g_gpu = torch.Generator(device='cuda')
g_gpu.manual_seed(214748364)
inp = torch.randn(2**20, device='cuda', generator=g_gpu)

print("name, scripted_cuda_time, fused_cuda_time, remat_cuda_time, num_fused_group")

profile_function("f", f, inp)

for i in range(10):
    profile_function(f"rand_test_{i}", frandom, inp)

def f2(y):
    x = y
    for _ in range(5):
        x = x.cos()
    b = x.sum()
    b.backward()
    
    return y.grad


inp = torch.randn(2**20, device='cuda', generator=g_gpu, requires_grad=True)


# profile_function("joint-f2", f2, inp)