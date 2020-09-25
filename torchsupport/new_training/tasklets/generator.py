import torch

from torchsupport.new_training.tasklets.tasklet import Tasklet
from torchsupport.new_training.tasklets.encoder import KLEncoder

class Generator(Tasklet):
  def __init__(self, generator):
    super().__init__()
    self.generator = generator

  def run(self, inputs, args):
    fake_sample = self.generator(inputs, args)
    self.store(fake_sample=fake_sample)
    return fake_sample

  def loss(self, fake_sample) -> []:
    pass

class Decoder(Generator):
  def __init__(self, generator, match):
    super().__init__(generator)
    self.match = match.func

  def loss(self, target, reconstruction):
    result = self.match(target, reconstruction)
    self.store(target=target)
    self.store(reconstruction_loss=result)
    return result

  def step(self, inputs, target, args):
    transform = self.run(inputs, args)
    loss = self.loss(target, transform)
    return loss
