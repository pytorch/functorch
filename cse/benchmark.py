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

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        for _ in range(5):
            f(inp)

    timing = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    print(type(timing))
    print(timing)


def f(x):
 return x.cos().cos()

inp = torch.randn(2**20, device='cuda')

fx_g =  make_fx(f)(inp)
script_f = torch.jit.script(fx_g)
new_g = modify(fx_g.graph)
new_g = fx.GraphModule(fx_g, new_g)
script_g = torch.jit.script(new_g)


profile_it(script_f, inp)
profile_it(script_g, inp)


# print(type(fx_g))
# fx_g = fx.symbolic_trace(f)
# fx_g.graph.eliminate_dead_code()
# fx_g.recompile()