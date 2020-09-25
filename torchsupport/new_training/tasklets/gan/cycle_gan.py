import torch
import torch.nn.functional as func

from torchsupport.data.io import make_differentiable

from torchsupport.new_training.tasklets.tasklet import Tasklet
from torchsupport.new_training.tasklets.gan.gan import GANGenerator, Discriminator
from torchsupport.new_training.tasklets.matching import match_l1

class CrossModalGenerator(GANGenerator):
  def __init__(self, generator, discriminator, match=match_l1,
               run=None, loss=None):
    super().__init__(
      generator, discriminator,
      run=run, loss=loss
    )
    self.match = match

  def loss(self, decision, fake, target):
    gan_loss = super().loss(decision)
    reconstruction_loss = self.match(target, fake)
    total_loss = gan_loss + reconstruction_loss
    self.store(
      reconstruction_loss=reconstruction_loss,
      gan_loss=gan_loss
    )
    return total_loss
