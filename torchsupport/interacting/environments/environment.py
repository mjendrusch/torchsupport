from torchsupport.data.namedtuple import namedtuple

class Environment:
  data_type = namedtuple("Data", [
    "state", "action", "rewards", "done"
  ])
  def reset(self):
    raise NotImplementedError

  def push_changes(self):
    pass

  def pull_changes(self):
    pass

  def action_space(self):
    raise NotImplementedError

  def observation_space(self):
    raise NotImplementedError

  def is_done(self):
    raise NotImplementedError

  def observe(self):
    raise NotImplementedError

  def act(self, action):
    raise NotImplementedError

  def schema(self):
    raise NotImplementedError
