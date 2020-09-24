import torch
import torch.nn.functional as func

from torchsupport.data.io import make_differentiable

from torchsupport.new_training.tasklets.tasklet import Tasklet
from torchsupport.new_training.tasklets.gan.gan import GANGenerator, Discriminator
from torchsupport.new_training.tasklets.matching import match_l1

class CrossModalGenerator(GANGenerator):
  def __init__(self, generator, discriminator, match=match_l1,
               discriminator_type=Discriminator):
    super().__init__(
      generator, discriminator,
      discriminator_type=discriminator_type
    )
    self.match = match.func

  def run(self, latent, inputs, args):
    return super().run(latent, (inputs, args))

  def loss(self, inputs, fake_sample, decision, realness) -> (
    "cross_modal_loss", "generator_loss", "reconstruction_loss"
  ):
    gan_loss = super().loss(decision, realness)
    reconstruction_loss = self.match(inputs, fake_sample)
    total_loss = gan_loss + reconstruction_loss
    return total_loss, gan_loss, reconstruction_loss


