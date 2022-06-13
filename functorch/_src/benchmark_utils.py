
import os
import sys
import logging
import copy
import time

import torch
from torch.profiler import profile, ProfilerActivity
import torchdynamo


current_name = ""
graph_index = 0
folder_name = ""

def _save_fx(gm, example_inputs):
    from functorch.compile import aot_module_simplified

    def graph_saver_forward(gm, _):
        global current_name
        global graph_index
        global folder_name
        isExist = os.path.exists(f"{folder_name}/{current_name}")
        if not isExist:
            os.makedirs(f"{folder_name}/{current_name}")
        gm.to_folder(f"{folder_name}/{current_name}/{current_name}_forward_{graph_index}")
        return gm

    def graph_saver_backward(gm, _):
        global current_name
        global graph_index
        gm.to_folder(f"{folder_name}/{current_name}/{current_name}_backward_{graph_index}")
        graph_index = graph_index + 1
        return gm

    return aot_module_simplified(gm, fw_compiler=graph_saver_forward, bw_compiler=graph_saver_backward)



def save_fx_graph(f, input, graph_name, _folder_name, manual_seed = 1337):
    """
    The forward/backward computation graph of f will be stored in 
    {folder_name}/{current_name}/{current_name}_forward_{graph_index} and 
    {folder_name}/{current_name}/{current_name}_backward_{graph_index} respectively.

    Since each f might produce multiple graphs, the graph_index is used to distinguish difference graphs
    """
    global current_name
    global folder_name
    current_name = graph_name
    folder_name = _folder_name
    torch.enable_grad()
    torch.manual_seed(manual_seed)
    optimize_ctx = torchdynamo.optimize(    
            _save_fx 
        )
    try:
        with optimize_ctx:
            f(input)
    except Exception:
        logging.exception("unhandled error")
        print("ERROR")
        return sys.exit(-1)


def dump_chrome_trace(f, input, trace_filename, optimize_ctx, num_runs = 1, randomize_input = False, 
                            devices = ["cuda"], activities=[ProfilerActivity.CUDA], kwargs_for_f={}, kwargs_for_profiler={}):
    """
    Output the chrome trace of running f(input, **kwargs_for_f) with [optimize_ctx] [num_runs] times to [trace_filename].
    [activities] are the activities that the profiler will record
    Return total runtime without the profiler

    Outputs to trace_filename
    """

    if devices != ["cpu"] and torch.cuda.is_available():
        synchronize = torch.cuda.synchronize
    else:
        synchronize =  lambda: None

    inputs = (
        randomize_input(copy.deepcopy(input))
        if randomize_input
        else inputs
    )
    
    
    with optimize_ctx:
        torch.manual_seed(1337)
        for _ in range(5): # warmup runs
            f(inputs, **kwargs_for_f)
            synchronize()
        torch.manual_seed(1337)
        t0 = time.perf_counter()
        for _ in range(num_runs):
            f(input, **kwargs_for_f)
            synchronize()
        t1 = time.perf_counter()
    timing = t1 - t0
    
    with profile(activities=activities, **kwargs_for_profiler) as prof:
        with optimize_ctx:
            synchronize()
            torch.manual_seed(1337)
            for _ in range(num_runs):
                f(input, **kwargs_for_f)
                synchronize()
    prof.export_chrome_trace(trace_filename)

    return  f"{timing:.3f}"