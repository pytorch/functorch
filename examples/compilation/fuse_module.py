from functorch import compiled_module, tvm_compile
import torch.nn as nn
import torch
from functools import partial

def nop(f, _):
    return f

fw_compiler = partial(tvm_compile, name='fw_keops')
bw_compiler = partial(tvm_compile, name='bw_keops')

def run(mod, input):
    out = mod(input)
    out.sum().backward()
    grads = [p.grad for p in mod.parameters()]
    return out, *grads

class Foo(nn.Module):
    def __init__(self):
        super(Foo, self).__init__()
        self.param = nn.Parameter(torch.randn(2000, 1, 4))

    def forward(self, x):
        return (self.param * x).sum(dim=0)

input = torch.randn(1, 2000, 4)
mod = Foo()
compiled_mod = compiled_module(mod, fw_compiler, bw_compiler)

for a, b in zip(run(mod, input), run(compiled_mod, input)):
    torch.testing.assert_allclose(a, b)
