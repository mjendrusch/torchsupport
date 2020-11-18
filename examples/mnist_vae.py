import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset
from torch.distributions import Normal, Categorical, Distribution

from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torchsupport.distributions.standard import StandardNormal
from torchsupport.data.match import MatchTensor, match_l2, match_bce
from torchsupport.training.vae import VAETraining, Prior

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

class VAEDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label = self.data[index]
    return (MatchTensor(data, match=match_bce),)

  def __len__(self):
    return len(self.data)

class Encoder(nn.Module):
  def __init__(self, z=32):
    super(Encoder, self).__init__()
    self.z = z
    self.encoder = nn.Sequential(
      nn.Linear(28 * 28, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, z),
      nn.ReLU()
    )
    self.mean = nn.Linear(z, z)
    self.logvar = nn.Linear(z, z)

  def forward(self, inputs):
    inputs = inputs.view(inputs.size(0), -1)
    features = self.encoder(inputs)
    mean = self.mean(features)
    logvar = self.logvar(features)
    return (Normal(mean, logvar.exp()), )

class GMM(Distribution):
  def __init__(self, logits, locs, scales):
    self.logits = logits.clone()
    self.locs = locs.clone()
    self.scales = scales.clone()

  def log_prob(self, x):
    normal = Normal(self.locs, self.scales)
    prob = self.logits.log_softmax(dim=0)[None, :, None]
    normal_logprob = normal.log_prob(x[:, None])
    normal_logprob = normal_logprob.view(*normal_logprob.shape[:2], -1)
    total = (prob + normal_logprob).logsumexp(dim=1)
    return total

  def rsample(self, sample_shape=torch.Size()):
    cat = Categorical(logits=self.logits).sample(sample_shape=sample_shape)
    loc = self.locs[cat]
    scale = self.scales[cat]
    dist = Normal(loc, scale)
    return dist.sample()

  def sample(self, sample_shape=torch.Size()):
    with torch.no_grad():
      return self.rsample(sample_shape=sample_shape)

class CustomPrior(nn.Module):
  def __init__(self, z=32, mixture=10):
    super().__init__()
    self.logits = nn.Parameter(torch.zeros(mixture))
    self.mean = nn.Parameter(torch.zeros(mixture, z))
    self.logvar = nn.Parameter(torch.zeros(mixture, z))

  def sample(self, size, *args, **kwargs):
    result = self.forward(*args, **kwargs)
    return (result.sample(sample_shape=(size,)), [])

  def forward(self, *args, **kwargs):
    return GMM(self.logits, self.mean, self.logvar.exp())

class Decoder(nn.Module):
  def __init__(self, z=32):
    super(Decoder, self).__init__()
    self.z = z  
    self.decoder = nn.Sequential(
      nn.Linear(z, 128),
      nn.ReLU(),
      nn.Linear(128, 256),
      nn.ReLU(),
      nn.Linear(256, 28 * 28)
    )

  def display(self, data):
    return data.sigmoid()

  def forward(self, sample):
    return self.decoder(sample).view(-1, 1, 28, 28)

if __name__ == "__main__":
  mnist = MNIST("examples/", download=True, transform=ToTensor())
  data = VAEDataset(mnist)

  z = 32

  encoder = Encoder(z=z)
  decoder = Decoder(z=z)

  training = VAETraining(
    encoder, decoder,
    CustomPrior(z, mixture=100), data,
    network_name="mnist-vae-new/prior-gmm-11",
    device="cpu",
    batch_size=64,
    max_epochs=1000,
    prior_mu=0.0,
    verbose=True
  )

  training.train()
