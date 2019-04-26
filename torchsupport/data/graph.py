import torch
from torch.utils.data import Dataset, DataLoader
from torchsupport.modules.structured.nodegraph import PartitionedNodeGraphTensor as PNGT
import torchsupport.modules.structured.nodegraph as ng

class LazyGraph(object):
  def __init__(self, adjacency_source, feature_source):
    self.adjacency_source = adjacency_source
    self.feature_source = feature_source

class AttributeListGraphIndicatorData(Dataset):
  def __init__(self, indicator, edge_list,
               node_attributes=None, edge_attributes=None,
               node_labels=None, edge_labels=None, graph_labels=None):
    super(AttributeListGraphIndicatorData, self).__init__()
    if node_attributes is None:
      node_attributes = (1.0 for _ in indicator)
    self.graphs = self._graphs_from_attributes(indicator, edge_list, node_attributes, edge_attributes)
    self.labels = self._graphs_from_attributes(indicator, edge_list, node_labels, edge_labels)
    self.graph_labels = graph_labels

  def data(self, idx):
    assert idx < len(self.graphs)
    return self.graphs[idx]

  def label(self, idx):
    assert idx < len(self.labels)
    return self.labels[idx]

  def graph_label(self, idx):
    assert idx < len(self.graph_labels)
    return self.graph_labels[idx]

  def __getitem__(self, idx):
    return {
      "data": self.data(idx),
      "label": self.label(idx),
      "graph_label": self.graph_label(idx)
    }

  def __len__(self):
    return len(self.graphs)

  def _graphs_from_attributes(self, indicator, edge_list, node_attributes, edge_attributes):
    """
    Creates graphs from attributes.
    """
    graphs = []
    current_graph = None
    previous_idg = 0
    node_offset = 0

    # construct edges:
    edges = [[] for _ in indicator]
    for source, target in edge_list:
      edges[source].append(target)
      edges[target].append(source)

    for idx, (idg, node_attributes) in enumerate(zip(indicator, node_attributes)):
      if previous_idg != idg:
        # finalize graph edges:
        for node in len(current_graph.adjacency):
          for edge in edges[node + node_offset]:
            if edge_attributes != None:
              edge_node = current_graph.add_node(torch.tensor(node_attributes), kind="edge")
              current_graph.add_edge(node, edge_node)
              current_graph.add_edge(edge_node, edge)
            else:
              current_graph.add_edge(node, edge)

        # append graph:
        graphs.append(current_graph)
        current_graph = PNGT()
        current_graph.add_kind("node")
        if edge_attributes != None:
          current_graph.add_kind("edge")

        # reset counters:
        previous_idg = idg
        node_offset = idx

      node = current_graph.add_node(torch.tensor(node_attributes), kind="node")
    
    return graphs

class SubgraphData(Dataset):
  def __init__(self, graph_data, patches=10, depth=5, keep_depth=1):
    self.data = graph_data
    self.patches = patches
    self.depth = depth
    self.keep_depth = keep_depth
    # TODO

def GraphDataLoader(
  dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
  num_workers=0, collate_fn=ng.cat, pin_memory=False, drop_last=False,
  timeout=0, worker_init_fn=None
):
  return DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
    num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
    timeout=timeout, worker_init_fn=worker_init_fn
  )
