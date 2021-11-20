import torch
import time
from functorch.compile import memory_efficient_operator_authoring, clear_compile_cache
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.benchmark import Timer

def timeme_with_profile(fn, args, string_id, is_profile=False):
    warmup = 50
    # n_repeats = 1000
    n_repeats = 1
    n_layers = 1
    old_args = args[:]
    for _ in range(0, warmup//n_layers):
        args = list(old_args[:])
        for _ in range(n_layers):
            args[0] = fn(*args)
        ref = args[0]
        loss = ref.sum()
        loss.backward()

    torch.cuda.synchronize()

    def run():
        for _ in range(0, n_repeats//n_layers):
            args = list(old_args[:])
            args[0].grad = None
            args[1].grad = None
            # args[2].grad = 
            for _ in range(n_layers):
                args[0] = fn(*args)
            ref = args[0]

            # loss = ref.sum()

            # torch.cuda.synchronize()
            # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_baseline:
            #     with record_function("baseline"):
            #         loss.backward()
            # print(prof_baseline.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            # torch.cuda.synchronize()

    if is_profile:
        # run()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_baseline:
            with record_function("baseline"):
                run()
        print(prof_baseline.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    else:
        time1 = time.time()
        run()
        time2 = time.time()
        total_time = time2 - time1
        avg_fwd = round(total_time / n_repeats * 10**6, 2)
        print(string_id, avg_fwd)

def timeme(fn, args, string_id):
    warmup = 50
    repeats = 1000
    iters = 1
    old_args = args[:]
    for _ in range(0, warmup//iters):
        args = list(old_args[:])

        args[0].grad = None
        args[1].grad = None
        args[2].grad = None
        ref = fn(*args)
        for _ in range(iters):
            args[0] = fn(*args)
        ref = args[0]
        loss = ref.sum()
        loss.backward()

    torch.cuda.synchronize()

    fwd_times = []
    bwd_times = []
    for _ in range(0, repeats//iters):
        args = list(old_args[:])
        args[0].grad = None
        args[1].grad = None
        args[2].grad = None
        fwd_start = time.time()
        ref = fn(*args)
        for _ in range(iters):
            args[0] = fn(*args)
        ref = args[0]
        torch.cuda.synchronize()
        fwd_end = time.time()

        loss = ref.sum()
        torch.cuda.synchronize()

        bwd_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd_end = time.time()

        fwd_times.append(fwd_end - fwd_start)
        bwd_times.append(bwd_end - bwd_start)
    avg_fwd = round(sum(fwd_times) / repeats * 10**6, 2)
    avg_bwd = round(sum(bwd_times) / repeats * 10**6, 2)
    avg_total = round(avg_fwd + avg_bwd, 2)
    print(string_id, avg_fwd, ",", avg_bwd, ",", avg_total, "us")
    return avg_total

# @brief fused bias, dropout, and residual at the end of Attention and FFN,
# store dropped position in mask, it's not in-place
#
# @param total_count total elements
# @param ratio drop ratio
# @param out [batch_size, seq_len, hidden_size], float and __half
# @param in [batch_size, seq_len, hidden_size], float and __half
# @param mask [batch_size, seq_len, hidden_size], uint8 type
# @param bias [hidden_size], ffn bias
# @param residual [batch_size, seq_len, hidden_size], float and __half
#
#   output4.x = (input4.x + b4.x) * scale * m[0] + res4.x;
#   output4.y = (input4.y + b4.y) * scale * m[1] + res4.y;
#   output4.z = (input4.z + b4.z) * scale * m[2] + res4.z;
#   output4.w = (input4.w + b4.w) * scale * m[3] + res4.w;
# 
#   out4[i] = output4;



# def f(a, b, c):
# 
#     # 5 reads in total = 3 primary input reads + 2 intermediate input reads
#     # 4 writes in total = 1 for each op + 1 saved mask
#     x = a + b
#     y = dropout(x)
#     z = y + c
#     return z
# 
# 
# def f_backward(dz):
#     # 3 reads in total = 1 input read for dz and 1 intermediate saved read for mask + 1 for sum
#     # 3 writes in total = 1 for each op + 1 for the dc
#     dy = dz
#     dc = dz
# 
#     dx = masked_scale(dy, self.saved_mask)
#     
#     da = dx
#     db = dx.sum()
# 
#     return (da, db, dc)
#
#
#
#
# For fused bwd, it is 2 reads + 3 writes. Considering, writes to be not on the critical path. Max we could benefit is 3 reads/2 writes = 1.5x 
#
#  graph(%self : __torch__.torch.fx.graph_module.___torch_mangle_2.GraphModule,
#        %lt.1 : Tensor,
#        %tangents_1.1 : Tensor):
#    %4 : int[] = prim::Constant[value=[1024]]()
#    %37 : bool = prim::CudaFusionGuard[types=[Float(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0), Bool(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0)]](%tangents_1.1, %lt.1)
#    %35 : Tensor, %36 : Tensor = prim::If(%37)
#      block0():
#        %sum_1.4 : Tensor, %mul_1.4 : Tensor = prim::CudaFusionGroup_0(%tangents_1.1, %lt.1)
#        -> (%sum_1.4, %mul_1.4)
#      block1():
#        %sum_1.1 : Tensor, %mul_1.1 : Tensor = prim::FallbackGraph_1(%tangents_1.1, %lt.1)
#        -> (%sum_1.1, %mul_1.1)
#    %view.1 : Tensor = aten::view(%35, %4) # <eval_with_key>.10:10:11
#    %23 : Tensor[] = prim::ListConstruct(%36, %view.1, %tangents_1.1)
#    return (%23)
#  with prim::CudaFusionGroup_0 = graph(%8 : Float(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0),
#        %11 : Bool(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0)):
#    %3 : NoneType = prim::Constant()
#    %2 : bool = prim::Constant[value=1]() # <eval_with_key>.10:9:46
#    %1 : int[] = prim::Constant[value=[0, 1]]()
#    %6 : float = prim::Constant[value=3.333333333333333]()
#    %type_as_1.1 : Float(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0) = aten::type_as(%11, %8) # <eval_with_key>.10:5:16
#    %mul.1 : Float(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0) = aten::mul(%8, %type_as_1.1) # <eval_with_key>.10:6:10
#    %mul_1.1 : Float(32, 196, 1024, strides=[200704, 1024, 1], requires_grad=0, device=cuda:0) = aten::mul(%mul.1, %6) # <eval_with_key>.10:8:12
#    %sum_1.1 : Float(1, 1, 1024, strides=[1024, 1024, 1], requires_grad=0, device=cuda:0) = aten::sum(%mul_1.1, %1, %2, %3) # <eval_with_key>.10:9:12
#    return (%sum_1.1, %mul_1.1)
#
#





def dropout_res_bias(input, bias, residual):
    a = torch.add(input, bias)
    b = torch.nn.functional.dropout(a, p=0.7, training=True)
    c = b + residual
    return c

def dropout_bias(input, bias):
    a = torch.add(input, bias)
    b = torch.nn.functional.dropout(a, p=0.7, training=True)
    return b


def dropout_act_gelu(input, bias):
    a = torch.add(input, bias)
    # b = torch.nn.functional.gelu(a)
    b = a * 0.5 * (1.0 + torch.tanh(0.79788456 * a * (1 + 0.044715 * a * a)))
    c = torch.nn.functional.dropout(b, p=0.6, training=True)
    return c


is_gelu = False
is_just_bias = False
if is_gelu:
    fn = dropout_act_gelu
elif is_just_bias:
    fn = dropout_bias
else:
    fn = dropout_res_bias

clear_compile_cache()

# Set inputs
device = 'cuda'
dtype = torch.float16
batch_size = 32
seq_len = 196
hidden_size = 1024
# batch_size = 2
# seq_len = 4
# hidden_size = 3
input = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True, device=device, dtype=dtype)
bias = torch.randn(hidden_size, requires_grad=True, device=device, dtype=dtype)
# bias = torch.randn(batch_size, seq_len, hidden_size, requires_grad=True, device=device, dtype=dtype)
residual = torch.randn(batch_size, seq_len, hidden_size, requires_grad=False, device=device, dtype=dtype)

# Clone inputs to help with accuracy testing
clone = lambda x : x.clone().detach().requires_grad_(True)
input_clone = clone(input)
bias_clone = clone(bias)
residual_clone = residual.clone().detach().requires_grad_(False)



compiler = "nvfuser"
if is_gelu:
    if compiler == "nvfuser":
        def eager_fn(input, bias):
            a = torch.add(input, bias)
            b = torch.nn.functional.gelu(a)
            c = torch.nn.functional.dropout(b, p=0.6, training=True)
            return c


        opt_fn = memory_efficient_operator_authoring(fn, compiler_name="torchscript_nvfuser")
    
        baseline = timeme(eager_fn, (input, bias), "Eager time =")
        with torch.jit.fuser("fuser2"):
            # optimized_ts_nvfuser = timeme(opt_fn, (input_clone, bias_clone, residual_clone), "AOT Autograd NVFuser =")
            optimized_ts_nvfuser = timeme(opt_fn, (input, bias), "AOT Autograd NVFuser =")
        print("speedup with nvfuser", baseline/optimized_ts_nvfuser)
else:
    if compiler == "nvfuser":
        opt_fn = memory_efficient_operator_authoring(fn, compiler_name="torchscript_nvfuser")

        # with torch.jit.fuser("fuser2"):
        #     ref = fn(input, bias, residual)
        #     ref.sum().backward()

        #     res = fn(input_clone, bias_clone, residual_clone)
        #     res.sum().backward()
        #     print(ref)
        #     print(res)
        #     print("################")
        #     print(input.grad)
        #     print(input_clone.grad)
        #     print("############")
        #     print(bias.grad)
        #     print(bias_clone.grad)
        #     print("############")
        #     print(residual.grad)
        #     print(residual_clone.grad)
        #baseline = timeme(fn, (input, bias, residual), "Eager time =")
        baseline = timeme(fn, (input, bias, residual), "Eager time =")
        timeme_with_profile(fn, (input, bias, residual), "Eager time =", is_profile=True)
        with torch.jit.fuser("fuser2"):
            # optimized_ts_nvfuser = timeme(opt_fn, (input_clone, bias_clone, residual_clone), "AOT Autograd NVFuser =")
            optimized_ts_nvfuser = timeme(opt_fn, (input_clone, bias_clone, residual_clone), "AOT Autograd NVFuser =")
            # timeme_with_profile(opt_fn, (input_clone, bias_clone, residual_clone), "AOT Autograd NVFuser =", is_profile=True)
            timeme_with_profile(opt_fn, (input_clone, bias_clone, residual_clone), "AOT Autograd NVFuser =", is_profile=True)
        print("speedup with nvfuser", baseline/optimized_ts_nvfuser)



        t = Timer(stmt = "fn(input, bias, residual)", globals = globals())
        timing_res = t.blocked_autorange()
        print(timing_res)

        t = Timer(stmt = "opt_fn(input, bias, residual)", globals = globals())
        timing_res = t.blocked_autorange()
        print(timing_res)
    elif compiler == "nnc":
        opt_fn = memory_efficient_operator_authoring(fn, compiler_name="torchscript_nnc")

        baseline = timeme(fn, (input, bias, residual), "Eager time =")
        optimized_ts_nnc = timeme(opt_fn, (input_clone, bias_clone, residual_clone), "AOT Autograd NNC =")
        print("speedup with nnc", baseline/optimized_ts_nnc)


