import os
import copy
import importlib
import pickle

import torch
from torch.fx._symbolic_trace import symbolic_trace
from functorch._src.remat_utils import rematerialize, get_fused_graph, rematerialize_fused_graph
from torch.profiler import profile, ProfilerActivity

graphs_dir = "/scratch/shangdiy/work/torchbenchmark/"
os.chdir(graphs_dir)

# models = []
# for x in os.listdir(graphs_dir):
#     models.append(x)

# model = "hf_Bart"
# graphs = []
# for x in os.listdir(graphs_dir+model):
#     input_dir = os.path.join(graphs_dir, model, x, f"{model}_forward_{graph_index}.input")
#     graphs.append(input_dir)

# def generate_random_input(inp_descs):
#     torch.manual_seed(1337)
#     inp = []
#     for inp_desc in inp_descs:
#         inp_type = inp_desc[0]
#         sizes = inp_desc[1]
#         stride = inp_desc[2]
#         dtype = inp_desc[3]
#         inp.append(inp_type(torch.rand(sizes, dtype=dtype, device="cuda")))
#     inp = tuple(inp)
#     return inp

test_cases = [
    "torch_bench_graphs/resnext50_32x4d/resnext50_32x4d_forward_0",
    "torch_bench_graphs/resnext50_32x4d/resnext50_32x4d_backward_0",
    "torch_bench_graphs/nvidia_deeprecommender/nvidia_deeprecommender_backward_0",
    "torch_bench_graphs/nvidia_deeprecommender/nvidia_deeprecommender_forward_0",
    "torch_bench_graphs/moco/moco_forward_4",
    "torch_bench_graphs/moco/moco_backward_0",
    "torch_bench_graphs/moco/moco_backward_7",
    "torch_bench_graphs/moco/moco_forward_9",
    "torch_bench_graphs/moco/moco_forward_3",
    "torch_bench_graphs/moco/moco_backward_10",
    "torch_bench_graphs/moco/moco_forward_7",
    "torch_bench_graphs/moco/moco_backward_9",
    "torch_bench_graphs/moco/moco_backward_3",
    "torch_bench_graphs/moco/moco_forward_10",
    "torch_bench_graphs/moco/moco_backward_4",
    "torch_bench_graphs/moco/moco_forward_0",


    "torch_bench_graphs/moco/moco_forward_2",
    "torch_bench_graphs/moco/moco_forward_8",
    "torch_bench_graphs/moco/moco_backward_11",

    "torch_bench_graphs/moco/moco_backward_1",
    "torch_bench_graphs/moco/moco_backward_5",
    "torch_bench_graphs/moco/moco_forward_1",
    "torch_bench_graphs/moco/moco_forward_6",

    "torch_bench_graphs/moco/moco_backward_8",
    "torch_bench_graphs/moco/moco_forward_11",
    "torch_bench_graphs/resnet18/resnet18_backward_0",
    "torch_bench_graphs/mnasnet1_0/mnasnet1_0_backward_0",
    "torch_bench_graphs/BERT_pytorch/BERT_pytorch_forward_0",
    "torch_bench_graphs/BERT_pytorch/BERT_pytorch_backward_0",
    "torch_bench_graphs/resnet50/resnet50_forward_0",
    "torch_bench_graphs/resnet50/resnet50_backward_0",
    "torch_bench_graphs/hf_DistilBert/hf_DistilBert_backward_0",
    "torch_bench_graphs/hf_DistilBert/hf_DistilBert_forward_1",
    "torch_bench_graphs/hf_DistilBert/hf_DistilBert_forward_0",
    "torch_bench_graphs/hf_DistilBert/hf_DistilBert_backward_1",
    "torch_bench_graphs/hf_Albert/hf_Albert_backward_1",
    "torch_bench_graphs/hf_Albert/hf_Albert_forward_3",
    "torch_bench_graphs/hf_Albert/hf_Albert_backward_2",
    "torch_bench_graphs/hf_Albert/hf_Albert_forward_0",
    "torch_bench_graphs/hf_Albert/hf_Albert_forward_2",
    "torch_bench_graphs/hf_Albert/hf_Albert_backward_0",
    "torch_bench_graphs/hf_Albert/hf_Albert_forward_1",
    "torch_bench_graphs/hf_Albert/hf_Albert_backward_3",
    "torch_bench_graphs/dlrm/dlrm_backward_0",
    "torch_bench_graphs/dlrm/dlrm_forward_0",
    "torch_bench_graphs/drq/drq_backward_0",
    "torch_bench_graphs/drq/drq_forward_1",
    "torch_bench_graphs/drq/drq_backward_1",
    "torch_bench_graphs/drq/drq_forward_0",
    "torch_bench_graphs/pytorch_struct/pytorch_struct_backward_0",
    "torch_bench_graphs/pytorch_struct/pytorch_struct_forward_0",
    "torch_bench_graphs/Background_Matting/Background_Matting_backward_0",
    "torch_bench_graphs/Background_Matting/Background_Matting_forward_0",
    "torch_bench_graphs/timm_regnet/timm_regnet_forward_0",
    "torch_bench_graphs/timm_regnet/timm_regnet_backward_0",

    "torch_bench_graphs/hf_Bert/hf_Bert_forward_1",     #passing
    "torch_bench_graphs/hf_Bert/hf_Bert_backward_2",      # passing
    "torch_bench_graphs/hf_Bert/hf_Bert_forward_2",     # passing
    "torch_bench_graphs/hf_Bert/hf_Bert_forward_0",       # passing
    "torch_bench_graphs/hf_Bert/hf_Bert_backward_0",        # passing

    "torch_bench_graphs/densenet121/densenet121_backward_0",
    "torch_bench_graphs/densenet121/densenet121_forward_0",
    "torch_bench_graphs/timm_nfnet/timm_nfnet_backward_0",
    "torch_bench_graphs/timm_nfnet/timm_nfnet_forward_0",
    "torch_bench_graphs/squeezenet1_1/squeezenet1_1_forward_0",         # passed
    "torch_bench_graphs/squeezenet1_1/squeezenet1_1_backward_0",        # passed
    "torch_bench_graphs/alexnet/alexnet_forward_0",     # passed
    "torch_bench_graphs/alexnet/alexnet_backward_0",    # passed
    "torch_bench_graphs/Super_SloMo/Super_SloMo_forward_0",
    "torch_bench_graphs/Super_SloMo/Super_SloMo_backward_0",
    "torch_bench_graphs/timm_vision_transformer/timm_vision_transformer_backward_0",
    "torch_bench_graphs/timm_vision_transformer/timm_vision_transformer_forward_0",
    "torch_bench_graphs/maml_omniglot/maml_omniglot_backward_0",
    "torch_bench_graphs/maml_omniglot/maml_omniglot_forward_0",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_1",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_13",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_0",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_7",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_6",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_11",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_9",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_3",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_10",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_2",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_8",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_12",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_5",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_4",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_6",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_10",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_7",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_12",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_0",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_1",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_4",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_13",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_5",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_2",
    "torch_bench_graphs/hf_Bart/hf_Bart_backward_8",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_9",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_3",
    "torch_bench_graphs/hf_Bart/hf_Bart_forward_11",
    "torch_bench_graphs/timm_resnest/timm_resnest_forward_0",
    "torch_bench_graphs/timm_resnest/timm_resnest_backward_0",
    "torch_bench_graphs/mobilenet_v2/mobilenet_v2_backward_0",
    "torch_bench_graphs/mobilenet_v2/mobilenet_v2_forward_0",
    "torch_bench_graphs/timm_efficientnet/timm_efficientnet_forward_0",
    "torch_bench_graphs/timm_efficientnet/timm_efficientnet_backward_0",
    "torch_bench_graphs/soft_actor_critic/soft_actor_critic_backward_1",
    "torch_bench_graphs/soft_actor_critic/soft_actor_critic_forward_1",
    "torch_bench_graphs/soft_actor_critic/soft_actor_critic_backward_0",
    "torch_bench_graphs/soft_actor_critic/soft_actor_critic_forward_0",
    "torch_bench_graphs/mobilenet_v2_quantized_qat/mobilenet_v2_quantized_qat_backward_0",
    "torch_bench_graphs/mobilenet_v2_quantized_qat/mobilenet_v2_quantized_qat_forward_0",
    "torch_bench_graphs/LearningToPaint/LearningToPaint_backward_0",
    "torch_bench_graphs/LearningToPaint/LearningToPaint_forward_0",
    "torch_bench_graphs/vgg16/vgg16_forward_0",
    "torch_bench_graphs/vgg16/vgg16_backward_0",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_1",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_6",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_1",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_6",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_11",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_8",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_2",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_5",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_8",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_2",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_5",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_10",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_7",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_0",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_7",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_0",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_4",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_11",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_3",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_9",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_4",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_10",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_3",
    "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_9",
    "torch_bench_graphs/pytorch_unet/pytorch_unet_backward_0",
    "torch_bench_graphs/pytorch_unet/pytorch_unet_forward_0",
    "torch_bench_graphs/dcgan/dcgan_backward_0",
    "torch_bench_graphs/dcgan/dcgan_forward_0",
    "torch_bench_graphs/timm_vovnet/timm_vovnet_forward_0",
    "torch_bench_graphs/timm_vovnet/timm_vovnet_backward_0",
    "torch_bench_graphs/hf_T5/hf_T5_forward_7",
    "torch_bench_graphs/hf_T5/hf_T5_forward_13",
    "torch_bench_graphs/hf_T5/hf_T5_backward_0",
    "torch_bench_graphs/hf_T5/hf_T5_backward_11",
    "torch_bench_graphs/hf_T5/hf_T5_backward_7",
    "torch_bench_graphs/hf_T5/hf_T5_forward_0",
    "torch_bench_graphs/hf_T5/hf_T5_forward_14",
    "torch_bench_graphs/hf_T5/hf_T5_backward_9",
    "torch_bench_graphs/hf_T5/hf_T5_backward_3",
    "torch_bench_graphs/hf_T5/hf_T5_forward_10",
    "torch_bench_graphs/hf_T5/hf_T5_forward_4",
    "torch_bench_graphs/hf_T5/hf_T5_backward_12",
    "torch_bench_graphs/hf_T5/hf_T5_forward_9",
    "torch_bench_graphs/hf_T5/hf_T5_forward_3",
    "torch_bench_graphs/hf_T5/hf_T5_backward_4",
    "torch_bench_graphs/hf_T5/hf_T5_backward_6",
    "torch_bench_graphs/hf_T5/hf_T5_forward_1",
    "torch_bench_graphs/hf_T5/hf_T5_backward_10",
    "torch_bench_graphs/hf_T5/hf_T5_forward_12",
    "torch_bench_graphs/hf_T5/hf_T5_forward_6",
    "torch_bench_graphs/hf_T5/hf_T5_backward_1",
    "torch_bench_graphs/hf_T5/hf_T5_forward_2",
    "torch_bench_graphs/hf_T5/hf_T5_forward_8",
    "torch_bench_graphs/hf_T5/hf_T5_backward_5",
    "torch_bench_graphs/hf_T5/hf_T5_backward_13",
    "torch_bench_graphs/hf_T5/hf_T5_backward_14",
    "torch_bench_graphs/hf_T5/hf_T5_backward_2",
    "torch_bench_graphs/hf_T5/hf_T5_backward_8",
    "torch_bench_graphs/hf_T5/hf_T5_forward_5",
    "torch_bench_graphs/hf_T5/hf_T5_forward_11",
    "torch_bench_graphs/shufflenet_v2_x1_0/shufflenet_v2_x1_0_backward_0",      # passing with nan

    # "torch_bench_graphs/hf_Bert/hf_Bert_backward_1",  #failing due to value mismatch
    # "torch_bench_graphs/shufflenet_v2_x1_0/shufflenet_v2_x1_0_forward_0",         # failing due to current build didn't include cudnn
    # "torch_bench_graphs/moco/moco_backward_6",      # failing due to input bad input meta
    # "torch_bench_graphs/moco/moco_forward_5",  #cudnn_batch_norm: ATen not compiled with cuDNN support
    # "torch_bench_graphs/moco/moco_backward_2",  # ??

    # "torch_bench_graphs/resnet18/resnet18_forward_0",  #cudnn_batch_norm: ATen not compiled with cuDNN support
    # "torch_bench_graphs/mnasnet1_0/mnasnet1_0_forward_0",  #cudnn_batch_norm: ATen not compiled with cuDNN support
]

# def benchmark_GPU_time(f, inp, list_inp, itr = 5):
#     if list_inp:
#         for _ in range(5):
#             f(*inp)
#         with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
#             for _ in range(itr):
#                 f(*inp)

#         timing = prof.key_averages()
#         cuda_time_total = 0
#         for e in timing:
#             cuda_time_total = cuda_time_total + e.cuda_time_total
#         return cuda_time_total / itr

#     for _ in range(5):
#         f(inp)
#     with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
#         for _ in range(itr):
#             f(inp)

#     # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))

#     timing = prof.key_averages()
#     cuda_time_total = 0
#     for e in timing:
#         cuda_time_total = cuda_time_total + e.cuda_time_total
#     return cuda_time_total / itr

# def profile_graph(traced_graph, inp, list_inp):
#     traced_graph.graph.eliminate_dead_code()
#     traced_graph.recompile()
#     script_f = torch.jit.script(traced_graph)
#     avg_cuda_time_f = benchmark_GPU_time(script_f, inp, list_inp)

#     return avg_cuda_time_f

# def profile_fused_graph(fused_graph, inp, list_inp):
#     num_fused_group = 0
#     for node in fused_graph.graph.nodes:
#         if "fused_" in node.name:
#             module = getattr(fused_graph, node.name)
#             setattr(fused_graph, node.name, torch.jit.script(module) )
#             num_fused_group += 1

#     if num_fused_group == 0: # no fused group
#         script_f = torch.jit.script(fused_graph)
#         return benchmark_GPU_time(script_f, inp, list_inp), 0

#     avg_cuda_time_g = benchmark_GPU_time(fused_graph, inp, list_inp)
#     return avg_cuda_time_g, num_fused_group


# def profile_fused(traced_graph, inp, list_inp):
#     traced_graph.graph.eliminate_dead_code()
#     traced_graph.recompile()
#     fused_graph = get_fused_graph(traced_graph)
#     return profile_fused_graph(fused_graph, inp, list_inp)


# def profile_rematerialize(traced_graph, inp, list_inp):
#     traced_graph.graph.eliminate_dead_code()
#     traced_graph.recompile()
#     fused_graph = rematerialize(traced_graph)
#     return profile_fused_graph(fused_graph, inp, list_inp)


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
    
device = 'cuda'

for dir in test_cases:
    path = dir.split('/')
    model_name = path[-1]
    module_path = '.'.join(path)
    input_data_path = f'{dir}/{model_name}.input'

    print(f"====== {model_name} ======")
    print(os.getcwd())
    print(module_path)
    module = importlib.import_module(module_path)

    m = module.FxModule()
    traced = symbolic_trace(m)
    node_users_map = {node.name: set(node.users.keys()) for node in traced.graph.nodes }
    try:
        print("Generating testing data...")
        with (open(input_data_path, 'rb')) as f:
            inputs_meta = pickle.load(f)
            # print(len(inputs_meta))
            # print(inputs_meta)

            inputs = []
            for meta in inputs_meta:
                type, shape, stride, dtype = meta

                if dtype in {torch.int, torch.int32, torch.int64, torch.bool, torch.int, torch.uint8}:
                    input = torch.randint(0, 1, shape, dtype=dtype, device=device)
                else:
                    input = torch.rand(shape, dtype=dtype, device=device)

                inputs.append(input)

        m.to(device)
        fused_graph.to(device)

        print("Running original model...")
        expected = m(*inputs)

        fused_graph = get_fused_graph(traced)
        print("Running fused model...")
        result = fused_graph(*inputs)   
        torch.testing.assert_close(expected, result, equal_nan=True, rtol=1e-5, atol=1e-5)

        fused_graph = rematerialize_fused_graph(fused_graph, node_users_map)
        result = fused_graph(*inputs)   
        torch.testing.assert_close(expected, result, equal_nan=True, rtol=1e-5, atol=1e-5)
        print("Passed!")

    except Exception as e:
        print(f"{model_name} failed!", e)