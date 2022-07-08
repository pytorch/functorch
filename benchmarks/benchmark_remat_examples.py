import random

import torch
from functorch import make_fx
import torch.fx as fx
import copy

from functorch._src.remat_utils_mincut import rematerialize, get_fused_graph
from functorch._src.compile_utils import strip_overloads, fx_graph_cse
from benchmark_remat_utils import profile_scripted_graph, profile_fused_graph, benchmark_GPU_time
from functorch.compile import default_decompositions

# Profile rematerialization algorithm on some hand-crafted examples

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


def profile_function(name, f, inp, list_inp = False):
    traced_graph = make_fx(f, decomposition_table=default_decompositions)(inp)

    eager_time = benchmark_GPU_time(traced_graph, inp, list_inp)
    avg_cuda_time_f = profile_scripted_graph(traced_graph, inp, list_inp)

    strip_overloads(traced_graph)
    csed = fx_graph_cse(traced_graph.graph)
    csed_graph =  fx.GraphModule(traced_graph, csed)
    csed_graph_copy = copy.deepcopy(csed_graph)

    avg_cuda_time_g, num_fused_group = profile_fused(csed_graph, inp, list_inp)
    avg_cuda_time_h, _ = profile_rematerialize(csed_graph_copy, inp, list_inp)
    print(f"{name}, {eager_time}, {avg_cuda_time_f}, {avg_cuda_time_g}, {avg_cuda_time_h}, {num_fused_group}", flush=True)


def f(a):
    b = a.cos()
    c = torch.relu(b)
    d = torch.clone(c)
    e = torch.relu(d)
    return b + c + e


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


def f2(y):
    x = y
    for _ in range(5):
        x = x.relu()
    b = x.sum()
    b.backward()
    
    return y.grad


def benchmark_on_small_examples():
    random.seed(1)
    print("name, eager_time, scripted_cuda_time, fused_cuda_time, remat_cuda_time, num_fusion_group", flush=True)
    g_gpu = torch.Generator(device='cuda')
    g_gpu.manual_seed(214748364)
    inp = torch.randn(2**20, device='cuda', generator=g_gpu)

    profile_function("f", f, inp)

    for i in range(10):
        profile_function(f"rand_test_{i}", frandom, inp)

    inp = torch.randn(2**20, device='cuda', generator=g_gpu, requires_grad=True)
    profile_function("joint-f2", f2, inp)


if __name__ == "__main__":
    benchmark_on_small_examples()
