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
from torchbench_utils import *
from torchdynamo.testing import same
import pickle


current_name = ""
graph_index = 0
folder_name = "torch_bench_graphs"
current_dir = os.getcwd()
torch.backends.cuda.matmul.allow_tf32 = True


os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
for torchbench_dir in (
    "../../torchbenchmark",
    "../../torchbench",
    "../../benchmark",
):
    if exists(torchbench_dir):
        break
assert exists(torchbench_dir), "../torchbenchmark does not exist"
torchbench_dir = abspath(torchbench_dir)
os.chdir(torchbench_dir)
sys.path.append(torchbench_dir)
log = logging.getLogger(__name__)


def save_fx(gm, example_inputs):
    from functorch.compile import aot_module_simplified

    def graph_saver_forward(gm, fw_args):
        input_meta = []
        for arg in fw_args:
            if(type(arg) == int or type(arg) == float):
                input_meta.append((type(arg),))
            else:
                input_meta.append((type(arg), arg.shape, arg.stride(), arg.dtype))
        global current_name
        global graph_index
        global folder_name
        isExist = os.path.exists(f"{folder_name}/{current_name}")
        if not isExist:
            os.makedirs(f"{folder_name}/{current_name}")
        gm.to_folder(f"{folder_name}/{current_name}/{current_name}_forward_{graph_index}")
        pickle.dump(input_meta, open( f"{folder_name}/{current_name}/{current_name}_forward_{graph_index}/{current_name}_forward_{graph_index}.input", "wb" ))
        return gm

    def graph_saver_backward(gm, bw_args):
        input_meta = []
        for arg in bw_args:
            if(type(arg) == int or type(arg) == float):
                input_meta.append((type(arg),))
            else:
                input_meta.append((type(arg), arg.shape, arg.stride(), arg.dtype))
        global current_name
        global graph_index
        gm.to_folder(f"{folder_name}/{current_name}/{current_name}_backward_{graph_index}")
        pickle.dump(input_meta, open( f"{folder_name}/{current_name}/{current_name}_backward_{graph_index}/{current_name}_backward_{graph_index}.input", "wb" ))
        graph_index = graph_index + 1
        return gm

    return aot_module_simplified(gm, fw_compiler=graph_saver_forward, bw_compiler=graph_saver_backward)



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

    optimize_ctx = torchdynamo.optimize(    
            save_fx # aot_module(gm, fw_compiler=graph_saver_forward, bw_compiler=graph_saver_backward)
        )
    
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

            # if args.float32:
            #     model, example_inputs = cast_to_fp32(model, example_inputs)
            # elif args.float16:
            #     model, example_inputs = cast_to_fp16(model, example_inputs)

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
                    if not same(correct_result, new_result):
                        if os.path.exists(f"{folder_name}/{current_name}"):
                            rmtree(f"{folder_name}/{current_name}")
                        print("INCORRECT")
                        return sys.exit(-1)
                except Exception:
                    if os.path.exists(f"{folder_name}/{current_name}"):
                        rmtree(f"{folder_name}/{current_name}")
                    logging.exception("unhandled error")
                    print("ERROR")
                    return sys.exit(-1)

    elif args.isolate:
        os.chdir(current_dir)
        for name in iter_model_names(args):
            try:
                subprocess.check_call([sys.executable] + sys.argv + [f"--only={name}"])
            except subprocess.SubprocessError:
                print("ERROR")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main()
