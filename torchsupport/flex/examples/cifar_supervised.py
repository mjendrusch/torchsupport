import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from torchsupport.utils.argparse import parse_options
from torchsupport.flex.log.log_types import LogImage
from torchsupport.flex.context.context import TrainingContext
from torchsupport.flex.data_distributions.data_distribution import DataDistribution
from torchsupport.flex.tasks.likelihood.maximum_likelihood import SupervisedArgs
from torchsupport.flex.training.supervised import supervised_training

def valid_callback(args: SupervisedArgs, ctx: TrainingContext=None):
  ctx.log(images=LogImage(args.sample))
  labels = args.prediction.argmax(dim=1)
  for idx in range(10):
    positive = args.sample[labels == idx]
    if positive.size(0) != 0:
      ctx.log(**{f"classified {idx}": LogImage(positive)})

class CIFAR10Dataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __getitem__(self, index):
    data, label = self.data[index]
    return data, label

  def __len__(self):
    return len(self.data)

class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(3, 16, 3),
      nn.MaxPool2d(2),
      nn.Conv2d(16, 32, 3),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, 3)
    )
    self.out = nn.Linear(64, 10)

  def forward(self, inputs):
    features = self.conv(inputs)
    features = func.adaptive_avg_pool2d(features, 1)
    return self.out(features.view(features.size(0), -1))

if __name__ == "__main__":
  opt = parse_options(
    "CIFAR10 classifier training using flex.",
    path="flexamples/cifar10-classifier",
    device="cpu",
    batch_size=64,
    max_epochs=1000,
    report_interval=10
  )

  cifar10 = CIFAR10("examples/", download=False, transform=ToTensor())
  data = CIFAR10Dataset(cifar10)
  data = DataDistribution(
    data, batch_size=opt.batch_size,
    device=opt.device
  )

  net = Classifier()

  training = supervised_training(
    net, data, valid_data=data,
    losses=[nn.CrossEntropyLoss()],
    path=opt.path,
    device=opt.device,
    batch_size=opt.batch_size,
    max_epochs=opt.max_epochs,
    report_interval=opt.report_interval
  )
  training.get_step("valid_step").extend(valid_callback)

  training.train()
