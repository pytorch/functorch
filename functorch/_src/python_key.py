# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import functools
from typing import Any, Dict, Optional, Tuple, Callable, Union
import torch
from torch._C import _disabled_torch_function_impl
import torch.utils._pytree as pytree
from torch.fx import Tracer, GraphModule
import torch.fx as fx
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from contextlib import contextmanager

from torch.fx.experimental.proxy_tensor import make_fx, ProxyTensor, dispatch_trace, PythonKeyTracer
