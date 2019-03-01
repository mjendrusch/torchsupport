
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torchsupport.training.training import Training
from torchsupport.data.io import netwrite
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabaz_score

from matplotlib import pyplot as plt

from tensorboardX import SummaryWriter

class VAETraining(Training):
  def __init__(self, encoder, decoder, data,
               optimizer=torch.optim.Adam,
               loss=nn.CrossEntropyLoss(),
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network"):
    super(VAETraining, self).__init__()

    self.checkpoint_path = network_name

    self.encoder = encoder.to(device)
    self.decoder = decoder.to(device)

    self.data = data
    self.train_data = None
    self.loss = loss
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.device = device

    self.network_name = network_name
    self.writer = SummaryWriter(network_name)

    self.epoch_id = 0
    self.step_id = 0

    self.optimizer = optimizer(
      list(self.encoder.parameters()) +
      list(self.decoder.parameters()),
      lr=5e-4
    )

  def vae_loss(self, mean, logvar, reconstruction, target, beta=20, c=0.5):
    mse = func.binary_cross_entropy(reconstruction, target, reduction="sum")
    mse /= target.size(0)
    kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp(), dim=0)
    kld = kld.sum()
    kld_c = beta * torch.norm(kld - c, 1)

    self.writer.add_scalar("mse loss", float(mse), self.step_id)
    self.writer.add_scalar("kld loss", float(kld), self.step_id)
    self.writer.add_scalar("kld-c loss", float(kld_c), self.step_id)

    return mse + kld_c

  def step(self, data):
    self.optimizer.zero_grad()
    data = data.to(self.device)
    features, mean, logvar = self.encoder(data)
    std = torch.exp(0.5 * logvar)
    sample = torch.randn_like(std).mul(std).add_(mean)

    reconstruction = self.decoder(sample)

    with torch.no_grad():
      vector = sample.view(sample.size(0), -1)

      weirdness = (sample[0] + sample[1]).unsqueeze(0) / 2
      self.decoder.eval()
      weird_reconstruction = self.decoder(weirdness)
      self.decoder.train()
      im_vs_rec = torch.cat(
        (
          data[0].cpu(),
          data[1].cpu(),
          reconstruction[0].cpu(),
          reconstruction[1].cpu(),
          weird_reconstruction[0].cpu()
        ),
        dim=2
      ).numpy()
      im_vs_rec = im_vs_rec - im_vs_rec.min()
      im_vs_rec = im_vs_rec / im_vs_rec.max()
      self.writer.add_image("im vs rec", im_vs_rec, self.step_id)

    loss_val = self.vae_loss(
      mean, logvar, reconstruction, data,
      beta=1000, c=0.5 + self.step_id * (50 - 0.5) * 0.00001
    )
    self.writer.add_scalar("reconstruction loss", float(loss_val), self.step_id)

    loss_val.backward()
    self.writer.add_scalar("total loss", float(loss_val), self.step_id)
    self.optimizer.step()
    self.each_step()

  def checkpoint(self):
    netwrite(
      self.encoder,
      f"{self.checkpoint_path}-encoder-epoch-{self.epoch_id}-step-{self.step_id}.torch"
    )
    netwrite(
      self.decoder,
      f"{self.checkpoint_path}-decoder-epoch-{self.epoch_id}-step-{self.step_id}.torch"
    )
    self.each_checkpoint()

  def train(self):
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=8,
        shuffle=True
      )
      for data, *_ in self.train_data:
        self.step(data)
        self.step_id += 1
      self.checkpoint()

    return self.encoder, self.decoder

class JointVAETraining(VAETraining):
  def __init__(self, encoder, decoder, data,
               n_classes=3,
               optimizer=torch.optim.Adam,
               loss=nn.CrossEntropyLoss(),
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               ctarget=50,
               dtarget=5,
               gamma=1000):
    super(JointVAETraining, self).__init__(
      encoder, decoder, data,
      optimizer=optimizer,
      loss=loss,
      max_epochs=max_epochs,
      batch_size=batch_size,
      device=device,
      network_name=network_name
    )
    self.n_classes = n_classes
    self.temperature = 0.67
    self.ctarget = ctarget
    self.dtarget = dtarget
    self.gamma = gamma

  def sample_gumbel(self, shape, eps=1e-20):
    U = torch.rand(shape).to(self.device)
    return -torch.log(-torch.log(U + eps) + eps)

  def gumbel_softmax_sample(self, probabilities, temperature):
    y = torch.log(probabilities + 1e-20) + self.sample_gumbel(probabilities.size())
    return func.softmax(y / temperature, dim=1)

  def gumbel_prior(self, probabilities):
    return 1 / probabilities.size(1)

  def gumbel_softmax(self, probabilities, temperature):
    """
    ST-gumbel-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = self.gumbel_softmax_sample(probabilities, temperature)
    # shape = y.size()
    # _, ind = y.max(dim=-1)
    # y_hard = torch.zeros_like(y).view(-1, shape[-1])
    # y_hard.scatter_(1, ind.view(-1, 1), 1)
    # y_hard = y_hard.view(*shape)
    # y_hard = (y_hard - y).detach() + y
    return y#y_hard.view(*shape)

  def gumbel_kl_loss(self, category, beta=1000, c=0.5):
    kld = torch.sum(category * torch.log(category + 1e-20), dim=1)
    kld = kld.mean(dim=0) + np.log(category.size(-1))

    kld_c = beta * torch.norm(kld - c, 1)

    self.writer.add_scalar("cat kld loss", float(kld), self.step_id)
    self.writer.add_scalar("cat kld-c loss", float(kld_c), self.step_id)

    return kld_c

  def step(self, data):
    self.optimizer.zero_grad()
    data = data.to(self.device)
    features, mean, logvar, logits = self.encoder(data)
    probabilities = logits
    std = torch.exp(0.5 * logvar)
    sample = torch.randn_like(std).mul(std).add_(mean)
    category = self.gumbel_softmax(
      probabilities,
      self.temperature
    )

    reconstruction = self.decoder(sample, category)

    with torch.no_grad():
      weirdness = (sample[0] + sample[1]).unsqueeze(0) / 2
      self.decoder.eval()
      cat_sample = torch.zeros_like(category[0]).unsqueeze(0)
      cat_sample[0, random.choice(range(category.size(1)))] = 1
      weird_reconstruction = self.decoder(
        weirdness, cat_sample.to(self.device)
      )
      self.decoder.train()
      im_vs_rec = torch.cat(
        (
          data[0].cpu(),
          reconstruction[0].cpu(),
          weird_reconstruction[0].cpu()
        ),
        dim=2
      ).numpy()
      im_vs_rec = im_vs_rec - im_vs_rec.min()
      im_vs_rec = im_vs_rec / im_vs_rec.max()
      self.writer.add_image("im vs rec", im_vs_rec, self.step_id)

    vae_loss = self.vae_loss(
      mean, logvar, reconstruction, data,
      beta=self.gamma, c=self.step_id * (self.ctarget) * 0.00001
    )
    kl_loss = self.gumbel_kl_loss(
      probabilities,
      beta=self.gamma, c=min(self.step_id * (self.dtarget) * 0.00001, np.log(category.size(-1)))
    )
    loss_val = (vae_loss + kl_loss) / (data.size(2) * data.size(3))
    self.writer.add_scalar("reconstruction loss", float(loss_val), self.step_id)

    loss_val.backward()
    self.writer.add_scalar("total loss", float(loss_val), self.step_id)
    self.optimizer.step()
    self.each_step()
