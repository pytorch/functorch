import os
from os.path import exists, abspath
import sys
import logging
import argparse
import subprocess
import itertools
import copy
import warnings
from shutil import rmtree

import torch
import torchdynamo
from torchbench_utils_dumpgraph import *
from torchdynamo.testing import same
import pickle
from functorch._src.compilers import get_save_fx_default_func


# current_name = ""
graph_index = 0
folder_name = "torch_bench_graphs"
current_dir = os.getcwd()
torch.backends.cuda.matmul.allow_tf32 = True


os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
for torchbench_dir in (
    "../torchbenchmark",
    "../torchbench",
    "../benchmark",
):
    if exists(torchbench_dir):
        break
assert exists(torchbench_dir), "../torchbenchmark does not exist"
torchbench_dir = abspath(torchbench_dir)
os.chdir(torchbench_dir)
sys.path.append(torchbench_dir)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude", "-x", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument("--devices", "-d", action="append", help="cpu or cuda")
    parser.add_argument("--only", help="used by --isolate to run just one model")
    parser.add_argument(
        "--isolate", action="store_true", help="run each model in its own process"
    )
    parser.add_argument(
        "--training",
        action="store_true",
        help="Performs training",
    )
    parser.add_argument(
        "--use-eval-mode",
        action="store_true",
        help="sets model.eval() to reduce randomness",
    )

    args = parser.parse_args()
    args.devices = args.devices or ["cpu"]
    args.filter = args.filter or [r"."]
    args.exclude = args.exclude or [r"^$"]

    # if args.nvfuser:
    #     torch._C._jit_override_can_fuse_on_cpu(False)
    #     torch._C._jit_override_can_fuse_on_gpu(False)
    #     torch._C._jit_set_texpr_fuser_enabled(False)
    #     torch._C._jit_set_nvfuser_enabled(True)
    # else:
    torch._C._jit_override_can_fuse_on_cpu(True)
    torch._C._jit_override_can_fuse_on_gpu(True)
    torch._C._jit_set_texpr_fuser_enabled(True)
    if torch.cuda.is_available():
        torch._C._jit_set_nvfuser_enabled(False)


    if args.training:
        model_iter_fn = forward_and_backward_pass
    else:
        model_iter_fn = forward_pass

    if args.only:
        for device in args.devices:
            try:
                device, name, model, example_inputs = load_model(
                    device, args.only, args.training, args.use_eval_mode
                )
                global current_name, current_device
                current_device = device
                current_name = name
            except NotImplementedError:
                continue  # bad benchmark implementation

            save_fx_func = get_save_fx_default_func(current_name, folder_name, dump_example_input = False)
            optimize_ctx = torchdynamo.optimize(    
                save_fx_func
            )
            with pick_grad(name, args.training):
                mode = "train" if args.training else "eval"
                sys.stdout.write(f"{current_device:4} {mode:5} {current_name:34}\n ")
                sys.stdout.flush()
                for submod in itertools.chain([model], model.modules()):
                    assert not torchdynamo.utils.is_jit_model(submod)

                torch.manual_seed(1337)
                correct_result = model_iter_fn(copy.deepcopy(model), example_inputs)

                torch.manual_seed(1337)
                torchdynamo.reset()
                try:
                    with optimize_ctx:
                        new_result = model_iter_fn(model, example_inputs)
                    
                    try:
                        if not same(correct_result, new_result):
                            # if os.path.exists(f"{folder_name}/{current_name}"):
                            #     rmtree(f"{folder_name}/{current_name}")
                            print("INCORRECT")
                            # return sys.exit(-1)
                    except Exception:
                        print("INCORRECT")
                except Exception:
                    if os.path.exists(f"{folder_name}/{current_name}"):
                        rmtree(f"{folder_name}/{current_name}")
                    logging.exception("unhandled error")
                    print("ERROR")
                    return sys.exit(-1)

    elif args.isolate:
        os.chdir(current_dir)
        for name in iter_model_names(args):
            if "detectron2" in name:
                continue
            try:
                subprocess.check_call([sys.executable] + sys.argv + [f"--only={name}"])
            except subprocess.SubprocessError:
                print("ERROR")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()