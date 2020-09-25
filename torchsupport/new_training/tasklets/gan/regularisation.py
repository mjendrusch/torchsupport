from functools import partial

import torch
import torch.nn.functional as func

from torchsupport.data.io import make_differentiable

from torchsupport.new_training.tasklets.tasklet import tasklet, Tasklet
from torchsupport.new_training.tasklets.generator import Generator
from torchsupport.new_training.tasklets.matching import match_l2, match_kl
from torchsupport.new_training.tasklets.loss import join
from torchsupport.new_training.tasklets.gan.gan import GANGenerator, GANDiscriminator

class GeneratorRegularisation(Tasklet):
  def run(self, inputs, args, fake, decision):
    pass

  def loss(self, inputs, args, fake, decision):
    return 0.0

class DiscriminatorRegularisation(Tasklet):
  def run(self, inputs, args, fake_inputs, fake_args):
    pass

  def loss(self, inputs, args, fake_inputs, fake_args,
           real_decision, fake_decision):
    return 0.0

# class DiscriminatorGradientPenalty(Tasklet):
#   def loss(self, inputs, decision) -> (
#     "gradient_penalty"
#   ):
#     gradient = ag.grad(decision, inputs, )

class RegularisedGANGenerator(GANGenerator):
  pass # TODO

class RegularisedGANDiscriminator(GANDiscriminator):
  pass # TODO

