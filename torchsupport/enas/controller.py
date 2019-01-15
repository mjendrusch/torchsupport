import torch
import torch.nn
import torch.nn.functional as func

class DummyController(object):
  def __init__(self):
    pass

  def __call__(self, input, state):
    state[input.name].append(input)
    return state, input

class NamedList(object):
  def __init__(self, name, value):
    self.name = name
    self.value = value

  def __len__(self):
    return len(self.value)

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      return NamedList(self.name, self.value[idx])
    else:
      return self.value[idx]

class _Manifestation(nn.Module):
  def __init__(self, lstm, op_embedding,
               link_embedding_in, link_embedding_out,
               link_attention, choice, incoming, operations):
    super(_Manifestation, self).__init__()
    self.lstm = lstm
    self.op_embedding = op_embedding
    self.link_embedding_in = link_embedding_in
    self.link_embedding_out = link_embedding_out
    self.link_attention = link_attention
    self.choice = choice
    self.incoming = incoming
    self.operations = operations

  def forward(self, input, hidden, prev_attention):
    trace = {
      "incoming": [],
      "operations": [],
      "incoming_logits": [],
      "operations_logits": []
    }
    
    for idx, indices in enumerate(self.incoming):
      history, hidden = self.lstm(input, hidden)
      embedding_out = self.link_embedding_out(hidden[-1])
      prev_attention.append(embedding_out)
      embedding_in = self.link_embedding_in(input)
      sum_attention = torch.tanh(
        embedding_in + torch.cat([self.prev_attention[index] for index in indices], dim=0)
      )
      logits = self.link_attention[idx](sum_attention)
      choice = torch.multinomial(logits)
      trace["incoming"].append(choice)
      trace["incoming_logits"].append(logits)
      input = history[choice]

    for idx, indices in enumerate(self.operations):
      history, hidden = self.lstm(input, hidden)
      logits = self.choice[idx](hidden[-1])
      choice = torch.multinomial(logits)
      trace["operations"].append(choice)
      trace["operations_logits"].append(logits)
      input = self.op_embedding[idx][choice]

    return history, hidden, trace

class SearchNode(object):
  def __init__(self, incoming=[], outgoing=[], operations=[], n_inputs=1):
    self.incoming = NamedList("incoming", incoming)
    self.outgoing = outgoing
    self._operations = operations
    self.operations = NamedList("operations", [
      idx for idx, _ in enumerate(operations)
    ])
    self.n_inputs = n_inputs

  def choose(self, controller):
    # Example:
    # controller(self.incoming)
    # controller(self.operations[0:2])
    # controller(self.operations[2:])
    raise NotImplementedError("Abstract method.")

  def forward(self, operations, incoming):
    raise NotImplementedError("Abstract method.")

  def manifest_choice(self, hidden_size=100):
    controller = DummyController()
    state = self.choose(controller)
    link_embedding_in = nn.Linear(hidden_size, hidden_size)
    link_embedding_out = nn.Linear(hidden_size, hidden_size)

    last_slice = None
    att = None
    link_attention = []
    for sl in state["incoming"]:
      if sl != last_slice:
        last_slice = sl
        att = nn.Linear(hidden_size, len(sl))
      link_attention.append(att)
    link_attention = nn.ModuleList(link_attention)

    last_slice = None
    chc = None
    choice = []
    for sl in state["operations"]:
      if sl != last_slice:
        last_slice = sl
        chc = nn.Linear(hidden_size, len(sl))
      choice.append(chc)
    choice = nn.ModuleList(choice)

    op_embedding = torch.randn((self.operations, hidden_size), requires_grad=True)

    def _manifestation(lstm):
      return _Manifestation(
        lstm, op_embedding, link_embedding_in, link_embedding_out,
        link_attention, choice, state["incoming"], state["operations"]
      )

    return _manifestation

  def manifest_module(self, operations, incoming):
    return self.forward(operations, incoming)

class _SearchSpaceManifestation(nn.Module):
  def __init__(self, node_manifestations, hidden=100):
    super(_SearchSpaceManifestation, self).__init__()
    self.lstm = nn.LSTMCell(hidden, hidden)
    self.prev_attention = []
    self.node_manifestations = node_manifestations

  def forward(self, input, hidden):
    full_trace = []
    for manifestation in self.node_manifestations:
      history, hidden, trace = manifestation(input, hidden, self.prev_attention)
      full_trace.append(trace)
    return history, hidden, full_trace

class SearchSpace(object):
  def __init__(self, nodes=[], edges=[]):
    self.nodes = nodes
    self.edges = edges

  def add_node(self, node : SearchNode):
    self.nodes.append(node)
    return len(self.nodes)

  def add_edge(self, source, target):
    self.edges.append((source, target))
    self.nodes[source].outgoing.append(self.nodes[target])
    self.nodes[target].incoming.append(self.nodes[source])

  def manifest_choice(self):
    ...
