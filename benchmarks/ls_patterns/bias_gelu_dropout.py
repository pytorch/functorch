import torch
import time
from functorch.compile import memory_efficient_operator_authoring, clear_compile_cache
import benchmark_helper
from xformers.components import Activation, build_activation
from xformers.triton import FusedDropoutBias
from xformers.benchmarks.utils import TestCase, pretty_plot, pretty_print


### ALL comments regarding the patetrns


def bias_gelu_dropout(input, bias):
    a = torch.add(input, bias)
    b = torch.nn.functional.gelu(a)
    c = torch.nn.functional.dropout(b, p=0.7, training=True)
    return c

fn = bias_gelu_dropout
clear_compile_cache()

# Set inputs
device = "cuda"
dtype = torch.float16
batch_size = 1
seq_len = 2048
intermediate_size = 12288
# batch_size = 2
# seq_len = 4
# intermediate_size = 3
# input = torch.randn(
#     batch_size,
#     seq_len,
#     intermediate_size,
#     requires_grad=True,
#     device=device,
#     dtype=dtype,
# )
# bias = torch.randn(intermediate_size, requires_grad=True, device=device, dtype=dtype)


# Get the optimized function
opt_fn = memory_efficient_operator_authoring(
    bias_gelu_dropout, compiler_name="torchscript_nvfuser"
)
shapes = [
    (8, 256, 512),
    (8, 512, 1024),
    (4, 1024, 1024),
    (2, 2048, 2048),
    (2, 4096, 4096),
    (1, 2048, 12288),
]

results = {}
# Profile cuda kernels
for batch_size, seq_len, intermediate_size in shapes:
    input = torch.randn(
        batch_size,
        seq_len,
        intermediate_size,
        requires_grad=True,
        device=device,
        dtype=dtype,
    )
    bias = torch.randn(intermediate_size, requires_grad=True, device=device, dtype=dtype)
    activation = Activation.GeLU
    torch_act = build_activation(activation)
    triton_dropout = FusedDropoutBias(
        0.7, bias_shape=intermediate_size, activation=activation
    )
    def triton_fn(x):
        return triton_dropout(x)
    
    print(batch_size, seq_len, intermediate_size)
    print("Eager")
    print(benchmark_helper.profile_cuda_kernels(fn, (input, bias), "Eager"))
    print("AOTAutograd")
    with torch.jit.fuser("fuser2"):
        print(benchmark_helper.profile_cuda_kernels(opt_fn, (input, bias), "AOTAutograd"))

    print("Triton")
    print(benchmark_helper.profile_cuda_kernels(triton_fn, (input,), "Triton"))
    print()


# # Time it with Torch Timer
# benchmark_helper.time_with_torch_timer(fn, (input, bias), "Eager")
# with torch.jit.fuser("fuser2"):
#     benchmark_helper.time_with_torch_timer(opt_fn, (input, bias), "AOTAutograd")

# # Time it with manual Timer
# benchmark_helper.time_with_manual_timer(fn, (input, bias), "Eager")
# with torch.jit.fuser("fuser2"):
#     benchmark_helper.time_with_manual_timer(opt_fn, (input, bias), "AOTAutograd")
