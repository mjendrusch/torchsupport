from torchsupport.new_training.tasklets.tasklet import Tasklet
from torchsupport.new_training.tasklets.encoder import Encoder, KLEncoder, LpEncoder, VQEncoder
from torchsupport.new_training.tasklets.generator import Decoder
from torchsupport.new_training.tasklets.matching import match_l1

class AE(Tasklet):
  def __init__(self, encoder, decoder,
               decoder_kind=Decoder, decoder_kwargs=None,
               encoder_kind=Encoder, encoder_kwargs=None,
               encoder_weight=1.0, decoder_weight=1.0):
    decoder_kwargs = decoder_kwargs or dict(match=match_l1)
    encoder_kwargs = encoder_kwargs or dict()
    self.encoder = encoder_kind(encoder, **encoder_kwargs).func
    self.decoder = decoder_kind(decoder, **decoder_kwargs).func
    self.encoder_weight = encoder_weight
    self.decoder_weight = decoder_weight

  def run(self, inputs, args) -> ("code", "reconstruction"):
    code = self.encoder.run(inputs, args)
    reconstruction = self.decoder.run(code, args)
    return code, reconstruction

  def loss(self, inputs, args, code, reconstruction) -> (
    "ae_loss", "encoder_loss", "decoder_loss"
  ):
    encoder_loss = self.encoder.loss(code, args)
    decoder_loss = self.decoder.loss(inputs, reconstruction)
    return self.decoder_weight * decoder_loss + self.encoder_weight * encoder_loss

class LpAE(AE):
  def __init__(self, encoder, decoder, match, p=2, beta=1.0):
    super().__init__(
      encoder, decoder,
      decoder_kind=Decoder, decoder_kwargs=dict(match=match),
      encoder_kind=LpEncoder, encoder_kwargs=dict(p=p),
      encoder_weight=beta, decoder_weight=1.0
    )

class VAE(AE):
  def __init__(self, prior, encoder, decoder, match, beta=1.0):
    super().__init__(
      encoder, decoder,
      decoder_kind=Decoder, decoder_kwargs=dict(match=match),
      encoder_kind=KLEncoder, encoder_kwargs=dict(prior=prior),
      encoder_weight=beta, decoder_weight=1.0
    )

class VQVAE(AE):
  def __init__(self, encoder, codebook, decoder, match):
    super().__init__(
      encoder, decoder,
      decoder_kind=Decoder, decoder_kwargs=dict(match=match),
      encoder_kind=VQEncoder, encoder_kwargs=dict(codebook=codebook),
      encoder_weight=1.0, decoder_weight=1.0
    )

  def run(self, inputs, args) -> ("code", "reconstruction"):
    code = self.encoder.run(inputs, args)
    reconstruction = self.decoder.run(code.code, args)
    return code, reconstruction

  def loss(self, inputs, code, reconstruction) -> (
    "ae_loss", "encoder_loss", "decoder_loss"
  ):
    _, value, assignment = code
    encoder_loss = self.encoder.loss(value, assignment)
    decoder_loss = self.decoder.loss(inputs, reconstruction)
    total_loss = encoder_loss + decoder_loss
    return total_loss, encoder_loss, decoder_loss
