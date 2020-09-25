from typing import List

from functools import partial
from collections import namedtuple

from torchsupport.new_training.tasklets.tasklet import Tasklet
from torchsupport.new_training.tasklets.encoder import KLEncoder, Encoder
from torchsupport.new_training.tasklets.generator import Decoder
from torchsupport.new_training.tasklets.matching import match

CycleMode = namedtuple("CycleMode", ["encoder", "decoder", "beta"])
class CycleVAE(Tasklet):
  @staticmethod
  def mode(prior, encoder, decoder, match, beta=1.0):
    return CycleMode(
      encoder=KLEncoder(encoder, prior),
      decoder=Decoder(decoder, match),
      beta=beta
    )

  def __init__(self, modes: List[CycleMode]):
    self.modes = modes

  def run(self, inputs, args):
    table = {}
    for idx, (data, mode) in enumerate(zip(inputs, self.modes)):
      code = mode.encoder.run(mode, args)
      for idy, (other_data, other) in enumerate(zip(inputs, self.modes)):
        reconstruction = other.decoder.run(code, args)
        recode = other.encoder.run(reconstruction, args)
        cycle = mode.decoder.run(recode, args)
        table[idx, idy] = (reconstruction, other_data, code, recode)
    return table

  def loss(self, recoding_map, args):
    cycle_losses = []
    match_losses = []
    recode_losses = []
    total_loss = 0.0
    for origin, other in recoding_map:
      rec, other, code, recode = recoding_map[origin, other]
      code_loss = self.modes[origin].encoder.loss(code, args)
      recode_loss = match(recode, code)
      match_loss = self.modes[other].decoder.loss(rec, other)
      cycle_loss = self.modes[origin].decoder.loss(rec, recode)
      total_loss += code_loss + recode_loss + match_loss + cycle_loss
      recode_losses.append(recode_loss.detach())
      match_losses.append(match_loss.detach())
      cycle_losses.append(cycle_loss.detach())
    self.store(
      recode_losses=recode_losses,
      match_losses=match_losses,
      cycle_losses=cycle_losses
    )
    return total_loss
