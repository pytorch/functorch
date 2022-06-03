import torch
import torch.fx as fx
from functorch import make_fx
import random
from torch.profiler import profile, record_function, ProfilerActivity

from cse import modify

# def f(x):
#     vals = [x]
#     ops = [torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu]
#     for _ in range(100): 
#         new_val = random.choice(ops)(random.choice(vals))
#         vals.append(new_val)
#     return vals[-1]

# @torch.jit.script
# def f(x):
#  return x.cos().cos()

# inp = torch.randn(2**20, device='cuda')
# for _ in range(5):
#  f(inp)

# with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
#     for _ in range(5):
#     f(inp)

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


# new_g = torch.jit.script(new_g)
# new_g(t)


