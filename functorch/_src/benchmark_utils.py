
import os
import sys
import logging
import copy
import time

import torch
from torch.profiler import profile, ProfilerActivity


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
