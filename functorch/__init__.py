# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

if torch.version.cuda == '10.2':
    raise RuntimeError(
        "We've detected an installation of PyTorch 1.12 with CUDA 10.2 support. "
        "The official functorch 0.2.0 binaries are not compatible with the "
        "PyTorch 1.12 CUDA 10.2 binaries. "
        "Please install PyTorch with support for a different version of CUDA "
        "(either cpu-only, 11.3, or 11.6; see pytorch.org for instructions) or "
        "file an issue on GitHub to discuss more.")


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

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass
