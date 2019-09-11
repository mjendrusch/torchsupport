class State:
  def __init__(self, name):
    self.name = name

  def read_action(self, training, data):
    setattr(training, self.name, data[self.name])

  def write_action(self, training, data):
    data[self.name] = getattr(training, self.name)

class PathState(State):
  def __init__(self, path):
    self.path = path[:-1]
    self.last = path[-1]
    self.name = "/".join(path)

  def walk(self, training):
    walk = training
    for step in self.path:
      walk = getattr(walk, step)
    return walk

  def read_action(self, training, data):
    setattr(self.walk(training), self.last, data[self.name])

  def write_action(self, training, data):
    data[self.name] = getattr(self.walk(training), self.last)

class NetState(State):
  def read_action(self, training, data):
    getattr(training, self.name).load_state_dict(data[self.name])

  def write_action(self, training, data):
    data[self.name] = getattr(training, self.name).state_dict()

class NetNameListState(NetState):
  def read_action(self, training, data):
    for key in data[self.name]:
      getattr(training, key).load_state_dict(data[self.name][key])

  def write_action(self, training, data):
    net_dict = {}
    for key in getattr(training, self.name):
      net_dict[key] = getattr(training, key).state_dict()
    data[self.name] = net_dict

class TrainingState(State):
  training_parameters = ["epoch_id", "step_id"]
  def __init__(self):
    pass

  def read_action(self, training, data):
    for name in self.training_parameters:
      setattr(training, name, data[name])

  def write_action(self, training, data):
    for name in self.training_parameters:
      data[name] = getattr(training, name)
