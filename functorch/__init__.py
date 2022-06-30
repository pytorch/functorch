# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

try:
    from .version import __version__  # noqa: F401
    from .version import pytorch_cuda_restrictions  # noqa: F401

    if pytorch_cuda_restrictions is not None:
        if torch.version.cuda is None:
            torch_cuda_version = 'cpu'
            verbose_torch_cuda_version = f'cpuonly'
        else:
            torch_cuda_version = torch.version.cuda
            verbose_torch_cuda_version = f'CUDA {torch.version.cuda}'

        if torch_cuda_version not in pytorch_cuda_restrictions:
            raise RuntimeError(
                f"We've detected an installation of PyTorch 1.12 with {verbose_torch_cuda_version} support. "
                "This functorch 0.2.0 binary is not compatible with the PyTorch installation. "
                "Please see our install page for suggestions on how to resolve this: "
                "https://pytorch.org/functorch/stable/install.html")

        # don't leak variables
        del torch_cuda_version
        del verbose_torch_cuda_version
    del pytorch_cuda_restrictions
except ImportError:
    pass

from . import _C

# Monkey patch PyTorch. This is a hack, we should try to upstream
# these pieces.
from ._src import monkey_patching as _monkey_patching

# Top-level APIs. Please think carefully before adding something to the
# top-level namespace:
# - private helper functions should go into functorch._src
# - very experimental things should go into functorch.experimental
# - compilation related things should go into functorch.compile

# functorch transforms
from ._src.vmap import vmap
from ._src.eager_transforms import (
    grad, grad_and_value, vjp, jacrev, jvp, jacfwd, hessian,
)
from ._src.python_key import make_fx

# utilities. Maybe these should go in their own namespace in the future?
from ._src.make_functional import (
    make_functional_with_buffers,
    make_functional,
    combine_state_for_ensemble,
    FunctionalModule,
    FunctionalModuleWithBuffers,
)

