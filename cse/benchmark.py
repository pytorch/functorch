import torch
import torch.fx as fx
from functorch import make_fx
import random
from torch.profiler import profile, record_function, ProfilerActivity

from cse import modify

# @torch.jit.script
# def f(x):
#     vals = [x]
#     ops = [torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu]
#     for _ in range(100): 
#         new_val = random.choice(ops)(random.choice(vals))
#         vals.append(new_val)
#     return vals[-1]


# def f(x):
#     a = x.sum()
#     b = x.sum()
#     c = x.sum()
#     d = x.sum()
#     return a + b + c + d

def profile_it(f, inp):
    for _ in range(5):
        f(inp)

    itr = 5
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(itr):
            f(inp)

    timing = prof.key_averages()
    timing_table = timing.table(sort_by="cuda_time_total", row_limit=10)
    print(timing_table)
    cuda_time_total = 0
    for e in timing:
        cuda_time_total = cuda_time_total + e.cuda_time_total
    return cuda_time_total / itr

    # print(type(timing)) [FunctionEventAvg]
    

def profile_function(name, f, inp):
    fx_g =  make_fx(f)(inp)
    script_f = torch.jit.script(fx_g)
    new_g = modify(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_g)
    script_g = torch.jit.script(new_g)

    num_node_decrease = len(fx_g.graph.nodes) - len(new_g.graph.nodes)

    avg_cuda_time_f = profile_it(script_f, inp)
    avg_cuda_time_g = profile_it(script_g, inp)

    print(f"{name}, {avg_cuda_time_f}, {avg_cuda_time_g}, {num_node_decrease}, {len(fx_g.graph.nodes)}")

g_gpu = torch.Generator(device='cuda')
g_gpu.manual_seed(2147483647)
inp = torch.randn(2**20, device='cuda', generator=g_gpu)

# def f1(x):
#  return x.cos().cos()

# profile_function("f1", f1, inp)

def f2(x):
    a = x.sum()
    b = x.sum()
    c = x.sum()
    d = x.sum()
    return a + b + c + d

profile_function("f2", f2, inp)



# print(type(fx_g))
# fx_g = fx.symbolic_trace(f)
# fx_g.graph.eliminate_dead_code()
# fx_g.recompile()