from functorch import make_fx
import torch

from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport

from torch.profiler import profile, ProfilerActivity
from torch.fx.passes.fuser_utils import fuse_by_partitions
from functorch._src.remat_utils import rematerialize, copy_all_nodes
from torch.fx._symbolic_trace import symbolic_trace
import pickle


class FxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_1):
        relu = torch.ops.aten.relu(x_1)
        relu_1 = torch.ops.aten.relu(x_1)
        clone = torch.ops.aten.clone(x_1)
        clone_1 = torch.ops.aten.clone(relu_1)
        relu_2 = torch.ops.aten.relu(clone)
        clone_2 = torch.ops.aten.clone(x_1)
        relu_3 = torch.ops.aten.relu(clone)
        clone_3 = torch.ops.aten.clone(relu_2)
        clone_4 = torch.ops.aten.clone(clone)
        relu_4 = torch.ops.aten.relu(clone)
        relu_5 = torch.ops.aten.relu(relu)
        clone_5 = torch.ops.aten.clone(x_1)
        clone_6 = torch.ops.aten.clone(relu_3)
        relu_6 = torch.ops.aten.relu(clone_5)
        relu_7 = torch.ops.aten.relu(relu_3)
        add = torch.ops.aten.add(relu_6, clone_6)
        relu_8 = torch.ops.aten.relu(clone_4)
        relu_9 = torch.ops.aten.relu(relu_4)
        relu_10 = torch.ops.aten.relu(clone_1)
        relu_11 = torch.ops.aten.relu(x_1)
        clone_7 = torch.ops.aten.clone(relu_1)
        add_1 = torch.ops.aten.add(relu_7, add)
        add_2 = torch.ops.aten.add(clone_3, clone_3)
        add_3 = torch.ops.aten.add(relu_9, relu_1)
        add_4 = torch.ops.aten.add(relu_1, relu_3)
        clone_8 = torch.ops.aten.clone(relu_10)
        clone_9 = torch.ops.aten.clone(relu_8)
        relu_12 = torch.ops.aten.relu(add_1)
        clone_10 = torch.ops.aten.clone(relu_2)
        clone_11 = torch.ops.aten.clone(relu_4)
        relu_13 = torch.ops.aten.relu(clone_6)
        relu_14 = torch.ops.aten.relu(relu_1)
        clone_12 = torch.ops.aten.clone(add_4)
        relu_15 = torch.ops.aten.relu(clone_5)
        clone_13 = torch.ops.aten.clone(clone_5)
        relu_16 = torch.ops.aten.relu(add)
        add_5 = torch.ops.aten.add(clone_3, relu_12)
        add_6 = torch.ops.aten.add(clone_1, clone_10)
        clone_14 = torch.ops.aten.clone(relu)
        clone_15 = torch.ops.aten.clone(relu_13)
        relu_17 = torch.ops.aten.relu(relu_2)
        relu_18 = torch.ops.aten.relu(relu_12)
        relu_19 = torch.ops.aten.relu(relu_18)
        clone_16 = torch.ops.aten.clone(relu_3)
        relu_20 = torch.ops.aten.relu(clone_2)
        clone_17 = torch.ops.aten.clone(clone_5)
        relu_21 = torch.ops.aten.relu(clone_16)
        clone_18 = torch.ops.aten.clone(relu_20)
        relu_22 = torch.ops.aten.relu(clone_15)
        add_7 = torch.ops.aten.add(clone_13, clone_14)
        add_8 = torch.ops.aten.add(x_1, 0)
        add_9 = torch.ops.aten.add(add_8, x_1);  add_8 = x_1 = None
        add_10 = torch.ops.aten.add(add_9, relu);  add_9 = relu = None
        add_11 = torch.ops.aten.add(add_10, relu_1);  add_10 = relu_1 = None
        add_12 = torch.ops.aten.add(add_11, clone);  add_11 = clone = None
        add_13 = torch.ops.aten.add(add_12, clone_1);  add_12 = clone_1 = None
        add_14 = torch.ops.aten.add(add_13, relu_2);  add_13 = relu_2 = None
        add_15 = torch.ops.aten.add(add_14, clone_2);  add_14 = clone_2 = None
        add_16 = torch.ops.aten.add(add_15, relu_3);  add_15 = relu_3 = None
        add_17 = torch.ops.aten.add(add_16, clone_3);  add_16 = clone_3 = None
        add_18 = torch.ops.aten.add(add_17, clone_4);  add_17 = clone_4 = None
        add_19 = torch.ops.aten.add(add_18, relu_4);  add_18 = relu_4 = None
        add_20 = torch.ops.aten.add(add_19, relu_5);  add_19 = relu_5 = None
        add_21 = torch.ops.aten.add(add_20, clone_5);  add_20 = clone_5 = None
        add_22 = torch.ops.aten.add(add_21, clone_6);  add_21 = clone_6 = None
        add_23 = torch.ops.aten.add(add_22, relu_6);  add_22 = relu_6 = None
        add_24 = torch.ops.aten.add(add_23, relu_7);  add_23 = relu_7 = None
        add_25 = torch.ops.aten.add(add_24, add);  add_24 = add = None
        add_26 = torch.ops.aten.add(add_25, relu_8);  add_25 = relu_8 = None
        add_27 = torch.ops.aten.add(add_26, relu_9);  add_26 = relu_9 = None
        add_28 = torch.ops.aten.add(add_27, relu_10);  add_27 = relu_10 = None
        add_29 = torch.ops.aten.add(add_28, relu_11);  add_28 = relu_11 = None
        add_30 = torch.ops.aten.add(add_29, clone_7);  add_29 = clone_7 = None
        add_31 = torch.ops.aten.add(add_30, add_1);  add_30 = add_1 = None
        add_32 = torch.ops.aten.add(add_31, add_2);  add_31 = add_2 = None
        add_33 = torch.ops.aten.add(add_32, add_3);  add_32 = add_3 = None
        add_34 = torch.ops.aten.add(add_33, add_4);  add_33 = add_4 = None
        add_35 = torch.ops.aten.add(add_34, clone_8);  add_34 = clone_8 = None
        add_36 = torch.ops.aten.add(add_35, clone_9);  add_35 = clone_9 = None
        add_37 = torch.ops.aten.add(add_36, relu_12);  add_36 = relu_12 = None
        add_38 = torch.ops.aten.add(add_37, clone_10);  add_37 = clone_10 = None
        add_39 = torch.ops.aten.add(add_38, clone_11);  add_38 = clone_11 = None
        add_40 = torch.ops.aten.add(add_39, relu_13);  add_39 = relu_13 = None
        add_41 = torch.ops.aten.add(add_40, relu_14);  add_40 = relu_14 = None
        add_42 = torch.ops.aten.add(add_41, clone_12);  add_41 = clone_12 = None
        add_43 = torch.ops.aten.add(add_42, relu_15);  add_42 = relu_15 = None
        add_44 = torch.ops.aten.add(add_43, clone_13);  add_43 = clone_13 = None
        add_45 = torch.ops.aten.add(add_44, relu_16);  add_44 = relu_16 = None
        add_46 = torch.ops.aten.add(add_45, add_5);  add_45 = add_5 = None
        add_47 = torch.ops.aten.add(add_46, add_6);  add_46 = add_6 = None
        add_48 = torch.ops.aten.add(add_47, clone_14);  add_47 = clone_14 = None
        add_49 = torch.ops.aten.add(add_48, clone_15);  add_48 = clone_15 = None
        add_50 = torch.ops.aten.add(add_49, relu_17);  add_49 = relu_17 = None
        add_51 = torch.ops.aten.add(add_50, relu_18);  add_50 = relu_18 = None
        add_52 = torch.ops.aten.add(add_51, relu_19);  add_51 = relu_19 = None
        add_53 = torch.ops.aten.add(add_52, clone_16);  add_52 = clone_16 = None
        add_54 = torch.ops.aten.add(add_53, relu_20);  add_53 = relu_20 = None
        add_55 = torch.ops.aten.add(add_54, clone_17);  add_54 = clone_17 = None
        add_56 = torch.ops.aten.add(add_55, relu_21);  add_55 = relu_21 = None
        add_57 = torch.ops.aten.add(add_56, clone_18);  add_56 = clone_18 = None
        add_58 = torch.ops.aten.add(add_57, relu_22);  add_57 = relu_22 = None
        add_59 = torch.ops.aten.add(add_58, add_7);  add_58 = add_7 = None
        return add_59
m = FxModule()
traced_graph = symbolic_trace(m)
fused_graph = rematerialize(traced_graph)
fused_graph(torch.rand(2))