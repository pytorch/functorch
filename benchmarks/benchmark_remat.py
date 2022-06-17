import random

import torch
from functorch import make_fx
from functorch.compile import draw_graph
from torch.profiler import profile, ProfilerActivity
from torch.fx._symbolic_trace import symbolic_trace
import pickle
import copy

from functorch._src.remat_utils import rematerialize, get_fused_graph, rematerialize_fused_graph

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
    script_f = torch.jit.script(traced_graph)
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


def profile_module(name, m, inp):
    traced_graph = symbolic_trace(m)
    avg_cuda_time_f = profile_graph(traced_graph, inp, True)

    traced_graph = symbolic_trace(m)
    fused_graph = get_fused_graph(traced_graph)
    avg_cuda_time_g, num_fused_group = profile_fused_graph(fused_graph, inp, True)

    traced_graph = symbolic_trace(m)
    fused_graph = rematerialize(traced_graph)
    avg_cuda_time_h, _ = profile_fused_graph(fused_graph, inp, True)
    print(f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}, {avg_cuda_time_h}, {num_fused_group}")


def profile_function(name, f, inp, list_inp = False):
    traced_graph = make_fx(f, decomposition_table={torch.ops.aten.detach.default: lambda x: x})(inp)
    avg_cuda_time_f = profile_graph(traced_graph, inp, list_inp)

    traced_graph_copy = copy.deepcopy(traced_graph)

    avg_cuda_time_g, num_fused_group = profile_fused(traced_graph, inp, list_inp)
    avg_cuda_time_h, _ = profile_rematerialize(traced_graph_copy, inp, list_inp)
    print(f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}, {avg_cuda_time_h}, {num_fused_group}")


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
    for _ in range(10):
        op = random.choice(ops)
        if op == torch.add:
            new_val = op(random.choice(vals), random.choice(vals))
        else:
            new_val = op(random.choice(vals))
        vals.append(new_val)
    return sum(vals)



class FxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17):
        convolution = torch.ops.aten.convolution(primals_17, primals_8, primals_7, [4, 4], [2, 2], [1, 1], False, [0, 0], 1);  primals_7 = None
        relu_ = torch.ops.aten.relu_(convolution);  convolution = None
        max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices(relu_, [3, 3], [2, 2])
        getitem = max_pool2d_with_indices[0]
        getitem_1 = max_pool2d_with_indices[1];  max_pool2d_with_indices = None
        convolution_1 = torch.ops.aten.convolution(getitem, primals_12, primals_11, [1, 1], [2, 2], [1, 1], False, [0, 0], 1);  primals_11 = None
        relu__1 = torch.ops.aten.relu_(convolution_1);  convolution_1 = None
        max_pool2d_with_indices_1 = torch.ops.aten.max_pool2d_with_indices(relu__1, [3, 3], [2, 2])
        getitem_2 = max_pool2d_with_indices_1[0]
        getitem_3 = max_pool2d_with_indices_1[1];  max_pool2d_with_indices_1 = None
        convolution_2 = torch.ops.aten.convolution(getitem_2, primals_14, primals_13, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_13 = None
        relu__2 = torch.ops.aten.relu_(convolution_2);  convolution_2 = None
        convolution_3 = torch.ops.aten.convolution(relu__2, primals_16, primals_15, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_15 = None
        relu__3 = torch.ops.aten.relu_(convolution_3);  convolution_3 = None
        convolution_4 = torch.ops.aten.convolution(relu__3, primals_10, primals_9, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_9 = None
        relu__4 = torch.ops.aten.relu_(convolution_4);  convolution_4 = None
        max_pool2d_with_indices_2 = torch.ops.aten.max_pool2d_with_indices(relu__4, [3, 3], [2, 2])
        getitem_4 = max_pool2d_with_indices_2[0]
        getitem_5 = max_pool2d_with_indices_2[1];  max_pool2d_with_indices_2 = None
        _adaptive_avg_pool2d = torch.ops.aten._adaptive_avg_pool2d(getitem_4, [6, 6])
        view = torch.ops.aten.view(_adaptive_avg_pool2d, [128, 9216]);  _adaptive_avg_pool2d = None
        t = torch.ops.aten.t(primals_2);  primals_2 = None
        addmm = torch.ops.aten.addmm(primals_1, view, t);  primals_1 = None
        relu__5 = torch.ops.aten.relu_(addmm);  addmm = None
        t_1 = torch.ops.aten.t(primals_4);  primals_4 = None
        addmm_1 = torch.ops.aten.addmm(primals_3, relu__5, t_1);  primals_3 = None
        relu__6 = torch.ops.aten.relu_(addmm_1);  addmm_1 = None
        t_2 = torch.ops.aten.t(primals_6);  primals_6 = None
        addmm_2 = torch.ops.aten.addmm(primals_5, relu__6, t_2);  primals_5 = None
        return [addmm_2, primals_16, relu__1, primals_17, t_1, relu__3, getitem_5, primals_8, relu__6, relu_, primals_10, t, view, getitem_4, t_2, getitem, primals_12, getitem_3, getitem_2, relu__4, getitem_1, relu__2, relu__5, primals_14]


g_gpu = torch.Generator(device='cuda')
g_gpu.manual_seed(214748364)
inp = torch.randn(2**20, device='cuda', generator=g_gpu)

print("name, scripted_time, fused_time, remat_time, num_fused_group")

profile_function("f", f, inp)

for i in range(10):
    profile_function(f"rand_test_{i}", frandom, inp)

# m = FxModule()
# inp = pickle.load(open("/scratch/shangdiy/work/torchbenchmark/torch_bench_graphs/alexnet/alexnet_forward_0/alexnet_forward_0.exampleinput", "rb"))
# profile_module("alexnet_forward", m, inp)