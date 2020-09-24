import torch

from torchsupport.new_training.tasklets.tasklet import Tasklet, tasklet
from torchsupport.new_training.tasklets.encoder import KLEncoder

class Generator(Tasklet):
  def __init__(self, generator):
    self.generator = generator

  def run(self, inputs, args) -> "fake_sample":
    fake_sample = self.generator(inputs, args)
    return fake_sample

  def loss(self, fake_sample) -> []:
    pass

class Decoder(Generator):
  def __init__(self, generator, match):
    super().__init__(generator)
    self.run = self.run.require(
      inputs="code"
    ).provide(
      fake_sample="reconstruction"
    )
    self.match = match.func

  def loss(self, inputs, reconstruction) -> "reconstruction_loss":
    return self.match(inputs, reconstruction)
