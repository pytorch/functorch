import copy
import time

import torch
from torch.profiler import profile, ProfilerActivity


def dump_chrome_trace(f, input, trace_filename, optimize_ctx, num_runs=1, randomize_input=False,
                            devices = ["cuda"], activities=[ProfilerActivity.CUDA], kwargs_for_f=None, kwargs_for_profiler=None):
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

    if kwargs_for_f is None:
        kwargs_for_f = {}
    if kwargs_for_profiler is None:
        kwargs_for_profile = {}
    
    
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