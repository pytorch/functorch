import os
import copy
import importlib
import pickle

import torch
import torch.fx as fx
from functorch import make_fx
from torch.fx._symbolic_trace import symbolic_trace
from functorch._src.remat_utils_mincut import rematerialize, rematerialize_stat, get_fused_graph, rematerialize_fused_graph, is_fused_node
from torch.profiler import profile, ProfilerActivity
from functorch._src.compile_utils import strip_overloads, fx_graph_cse
from benchmarks.benchmark_remat_utils import get_test_cases, get_skip_cases, check_num_remat_gm, profile_module

test_cases = get_test_cases()
SKIP_CASES = get_skip_cases()

# graphs_dir = "/scratch/shangdiy/work/torchbenchmark/"
# os.chdir(graphs_dir)


device = 'cuda'

print("name, eager_time, scripted_cuda_time, fused_cuda_time, remat_cuda_time, num_fusion_group, num_remat_group, memory_reduced")

for dir in test_cases:
    path = dir.split('/')
    model_name = path[-1]
    module_path = '.'.join(path)
    input_data_path = f'{dir}/{model_name}.input'
    if model_name in SKIP_CASES:
        continue
    
    # if model_name not in non_zero_mincut_memory_group:
    #     continue
    # print(f"====== {model_name} ======")
    module = importlib.import_module(module_path)

    m = module.FxModule()
    try:
        inputs = []
        with (open(input_data_path, 'rb')) as f:
            
            inputs_meta = pickle.load(f)
            for meta in inputs_meta:
                type, shape, stride, dtype = meta
                if dtype in {torch.int, torch.int32, torch.int64, torch.bool, torch.int, torch.uint8}:
                    input = torch.randint(0, 1, shape, dtype=dtype, device=device)
                else:
                    input = torch.rand(shape, dtype=dtype, device=device)
                inputs.append(input)
        m.to(device)
        # profile_module(model_name, m, inputs)
        check_num_remat_gm(model_name, m, inputs)

    except Exception as e:
        # pass
        print(f"{model_name}, failed,")
        print(e)