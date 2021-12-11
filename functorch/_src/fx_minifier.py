import torch.fx as fx
import copy
import torch
import math


class ConcreteProp(torch.fx.Interpreter):
    def run_node(self, n):
        result = super().run_node(n)

        found_tensor = False

        def extract_tensor_meta(obj):
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return obj
            else:
                return obj

        from torch.fx.node import map_aggregate
        concrete_value = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta['concrete_value'] = concrete_value

        return result

    def propagate(self, *args):
        return super().run(*args)

def get_placeholders(graph):
  return list(filter(lambda x: x.op == 'placeholder', graph.nodes))


# inplace modifies node/inps
def convert_node_to_placeholder(node, inps):
  node.op = 'placeholder'
  node.args = ()
  node.target = node.name
  concrete_val = node.meta['concrete_value']
  if isinstance(concrete_val, torch.Tensor):
    inps.append(concrete_val)
  else:
    inps.append(torch.zeros(()))



def minimizer(failing_f: fx.GraphModule, inps, pass_checker):
  failing_graph = failing_f.graph
  cur_size = len(failing_graph.nodes)

  def graph_passes(graph, inps):
    graph.lint()
    mod = fx.GraphModule(failing_f, graph)
    return pass_checker(mod, inps)

  ConcreteProp(failing_f).propagate(*inps)
  if graph_passes(failing_graph, inps):
    raise RuntimeError("Input graph did not fail the tester")
  print(f"Started off with {cur_size} nodes")

  def remove_suffix(cur_graph, inps):
    print("Strategy: Remove suffix")
    assert not graph_passes(cur_graph, inps)
    gap = 2**math.floor(math.log2(len(cur_graph.nodes)))
    tested = set()
    while gap >= 1:
      print(f"search gap: {gap}: ", end='')
      new_graph = fx.Graph()
      env = {}
      for idx, node in enumerate(cur_graph.nodes):
        new_node = new_graph.node_copy(node, lambda x: env[x])
        if node.op not in ['placeholder', 'output']:
          if idx % gap == 0 and idx not in tested:
            print(f"{idx}", end=',')
            output_node = new_graph.output(([new_node],))
            if not graph_passes(new_graph, inps) and len(new_graph.nodes) < len(cur_graph.nodes):
              print()
              print(f"SUCCESS: Found failing case with first {idx} nodes")
              return (new_graph, inps), True
            else:
              tested.add(idx)
              new_graph.erase_node(output_node)
        env[node] = new_node
      gap //= 2
      print()
    print("FAIL: Could not remove suffix")
    return (new_graph, inps), False


  def remove_unused_inputs(cur_graph, inps):
    print("Strategy: Remove unused inputs")
    assert not graph_passes(cur_graph, inps)
    ph_nodes = get_placeholders(cur_graph)
    if len(ph_nodes) != len(inps):
      print(cur_graph)
      print(len(inps))
    assert len(ph_nodes) == len(inps)

    new_inps = []
    for idx in range(len(ph_nodes)):
      if len(ph_nodes[idx].users) == 0:
        cur_graph.erase_node(ph_nodes[idx])
      else:
        new_inps.append(inps[idx])

    if len(new_inps) < len(inps):
      print(f"SUCCESS: Went from {len(inps)} inputs to {len(new_inps)} inputs")
      return (cur_graph, new_inps), True
    else:
      print("FAIL: Could not remove inputs")
      return (cur_graph, new_inps), False

  def consolidate_placeholders(cur_graph):
    new_graph = fx.Graph()
    env = {}
    for node in cur_graph.nodes:
      if node.op == 'placeholder':
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node

    for node in cur_graph.nodes:
      if node.op != 'placeholder':
        new_node = new_graph.node_copy(node, lambda x: env[x])
        env[node] = new_node
    return new_graph
      



  def remove_middle(cur_graph: fx.Graph, inps):
    print("Strategy: Delta Debugging")
    assert not graph_passes(cur_graph, inps)
    starting_placeholders = len(get_placeholders(cur_graph))
    num_nodes = len(cur_graph.nodes)
    gap = int(2**math.floor(math.log2(num_nodes)))
    while gap >= 1:
      print(f"Searching with gap of {gap}")
      for start_range in range(0, num_nodes, gap):
        is_removing = False
        new_graph = copy.deepcopy(cur_graph)
        new_inps = inps[:]
        for idx in range(start_range, min(num_nodes, start_range + gap)):
          new_node = list(new_graph.nodes)[idx]
          if new_node.op not in ['placeholder', 'output']:
            is_removing = True
            convert_node_to_placeholder(new_node, new_inps)
        if not is_removing:
          continue 
        new_graph = consolidate_placeholders(new_graph)
        if not graph_passes(new_graph, new_inps):
          print(f"SUCCESS: Went from {starting_placeholders} placeholders to {len(get_placeholders(new_graph))}")
          return (new_graph, new_inps), True
      gap //= 2

    print("FAIL: Could not remove prefix")
    return (cur_graph, inps), False


  while True:
    any_succeeded = False
    for strategy in [remove_suffix, remove_unused_inputs, remove_middle, remove_unused_inputs]:
      print(f"###################")
      print(f"Current size: {len(failing_graph.nodes)}")
      print(f"###################")
      out = strategy(copy.deepcopy(failing_graph), inps[:])
      (cur_graph, cur_inps), succeeded = out
      if succeeded:
        failing_graph = cur_graph
        failing_graph.eliminate_dead_code()
        inps = cur_inps
        any_succeeded = True
      print()

    if not any_succeeded:
      break

  print(fx.GraphModule(failing_f, failing_graph).code, [i.shape for i in inps])
  return