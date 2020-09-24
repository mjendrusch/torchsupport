from torchsupport.data.io import netwrite

class Checkpoint:
  def __init__(self, path, **networks):
    self.path = path
    self.networks = networks

  def __call__(self, context):
    print(
      "CHECKPOINT",
      self.path,
      context.context
    )
