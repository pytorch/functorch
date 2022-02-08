import copy
import torch.nn as nn


def batch_norm_without_running_stats(module: nn.Module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats:
        module.running_mean = None
        module.running_var = None
        module.num_batches_tracked = None
        module.track_running_stats = False


def replace_all_batch_norm_modules(root: nn.Module) -> nn.Module:
    # base case
    batch_norm_without_running_stats(root)

    for obj in root.modules():
        batch_norm_without_running_stats(obj)
    return root


def copy_and_replace_all_batch_norm_modules(root: nn.Module) -> nn.Module:
    replaced = copy.deepcopy(root)
    return replace_all_batch_norm_modules(replaced)
