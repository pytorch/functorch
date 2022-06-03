import torch
import torch.fx as fx
from functorch import make_fx
import random

from functorch._src.cse import fx_graph_cse
import unittest

# check if the CSE modified graph of f has delta less nodes, and do not reduce the number of nodes further on a second pass.
# delta is an integer >= -1. If delta = -1, only check if the new graph
#   has less or equal number of nodes
#  
def check(f, t, delta, check_val = True, graph_input = False):
    if graph_input:
        fx_g = f
    else:
        fx_g = make_fx(f)(t)
    new_graph = fx_graph_cse(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_graph)

    # the number of nodes decrease/ or stay the same
    old_num_nodes = len(fx_g.graph.nodes)
    new_num_nodes = len(new_graph.nodes)
    if delta == -1:
        assert  old_num_nodes >= new_num_nodes, (
            f"number of nodes increased {old_num_nodes}, {new_num_nodes}")
    else:
        assert  old_num_nodes == new_num_nodes + delta, (
            f"number of nodes not the same {old_num_nodes - delta}, {new_num_nodes}\n {fx_g.graph} \n {new_graph}")

    # a second pass should not reduce more nodes
    pass_2_graph = fx_graph_cse(new_graph)
    pass_2_num_nodes = len(pass_2_graph.nodes)
    assert pass_2_num_nodes == new_num_nodes, (
        f"second pass graph has less node {pass_2_num_nodes}, {new_num_nodes}\n {new_graph} \n {pass_2_graph}")

    # check correctness
    if check_val:
        true_result = fx_g(t)
        our_result = new_g(t)
        if true_result is None: # both return None
            assert our_result is None, f"true result is None, CSE result is {our_result}"
        else: # results returned are the same
            assert torch.all( true_result == our_result ), (
                f"results are different {true_result}, {our_result}") #check results are the same


class NoChangeTestCase(unittest.TestCase):

    def test_nochange(self):
        def f(x):
            a = x + 1
            b = x + a
            a = x
            d = x + a
            return b + d
        t = torch.randn(2, 2)
        check(f, t, 0)

    def test_empty(self):
        def f(x):
            pass
        t = torch.randn(2, 2)
        check(f, t, 0)

    def test_rand_like(self):
        def f(x):
            a = torch.rand_like(x)
            b = torch.rand_like(x)
            return a + b
        t = torch.randn(2, 2)
        check(f, t, 0, check_val=False)

    def test_rand_n(self):
        def f(x):
            g_cpu = torch.Generator()
            g_cpu.manual_seed(2147483647)
            a = torch.randn(4, generator=g_cpu)
            b = torch.randn(4, generator=g_cpu)
            return a + b
        t = torch.randn(2, 2)
        check(f, t, 0)

class ReduceTestCase(unittest.TestCase):

    def test_immutable_list_type(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1)
            c = x.sum()
            d = x.sum()
            return a + b + c + d
        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_immutable_list_multiple_entries(self):
        def f(x):
            a = x.sum(dim=[0, 1])
            b = x.sum(dim=[0, 1])
            c = x.sum(dim=1)
            d = x.sum(dim=1)
            return a + b + c + d
        t = torch.randn(2, 2)
        check(f, t, 2)


    def test_simple(self):
        def f(x):
            a = x.cos()
            b = x.cos()
            c = a + a
            d = b + b
            return c + d
        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple_2(self):
        def f(x):
            a = x.cos().sin()
            b = x.cos().sin()
            c = a + a
            d = b + b
            return c + d
        t = torch.randn(1)
        check(f, t, 3)

    def test_two_args_default(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=False)
            c = x.sum(dim=1, keepdim=False)
            d = x.sum(dim=1)
            return a + b + c + d
        t = torch.randn(2, 2)
        check(f, t, 3)

    def test_two_args(self):
        def f(x):
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=True)
            c = x.sum(dim=1, keepdim=True)
            d = x.sum(dim=1)
            return a + b + c + d
        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple_multiple_same_ops(self):
        def f(x):
            a = x.sum()
            b = x.sum()
            c = x.sum()
            d = x.sum()
            return a + b + c + d
        t = torch.randn(2, 2)
        check(f, t, 3)

    def test_nested_immutable_list_type(self):
        def f(x):
            a = torch.cat((x, x))
            b = torch.cat((x, x))
            return a + b
        t = torch.randn(2, 2)
        check(f, t, 1)

    def test_kwarg(self):
        def f(x):
            a = torch.ones_like(x)
            b = torch.ones_like(x)
            return a + b
        t = torch.randn(2, 2)
        check(f, t, 1)


class RandomOpTestCase(unittest.TestCase):
  def test_random(self):
    def f(x):
      vals = [x]
      ops = [torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu]
      for _ in range(100): 
        new_val = random.choice(ops)(random.choice(vals))
        vals.append(new_val)
      return vals[-1]

    fx_g = fx.symbolic_trace(f)
    fx_g.graph.eliminate_dead_code()
    fx_g.recompile()
    t = torch.randn(2, 2)

    for _ in range(30):
        check(fx_g, t, -1, graph_input=True)

if __name__ == '__main__':
    unittest.main()
