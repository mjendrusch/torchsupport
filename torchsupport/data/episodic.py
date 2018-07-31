import torch
import random
import pandas as pd
from io import imread
from torch.utils.data import Dataset, DataLoader, Sampler, SubsetRandomSampler
import os

class SubDataset(Dataset):
  def __init__(self, dataset, indices):
    """The subset of a Dataset, defined by a set of indices.
    
    Arguments
    ---------
    dataset : a :class:`Dataset` from which a subset is chosen.

    indices : a :class:`list` of indices making up the chosen subset.
    """
    self.dataset = dataset
    self.indices = indices
  
  def __len__(self):
    return len(self.indices)

  def __getitem__(self, index):
    idx = self.indices[index]
    img_name = os.path.join(self.dataset.annotation.iloc[idx, 0])
    image = imread(img_name)
    labelname = self.dataset.annotation.iloc[idx, 1]
    label = torch.LongTensor([[self.dataset.classmap[labelname]]])

    if self.dataset.transform != None:
      image = self.dataset.transform(image)

    sample = {
      "image": image,
      "label": label
    }

    return sample

class LabelPartitionedDataset(Dataset):
  def __init__(self, dataset):
    """A :class:`Dataset` partitioned by its labels.
    
    Arguments
    ---------
    dataset : a :class:`Dataset` to be partitioned.
    """
    self.dataset = dataset
    self.labelindices = {}
    self.labels = []
    for idx in range(len(self.dataset)):
      labelindex = self.dataset[idx]["label"][0][0]
      label = self.dataset.classmap_inv[labelindex]
      if label not in self.labels:
        self.labelindices[label] = []
        self.labels.append(label)
      self.labelindices[label].append(idx)
  
  def __getitem__(self, label):
    return SubDataset(self.dataset, self.labelindices[label])

class UnionSampler(Sampler):
  def __init__(self, samplers):
    """Samples randomly from a union of samplers.
    
    Arguments
    ---------
    samplers : a :class:`list` of :class:`Sampler`s to be unified.
    """
    self.samplers = samplers
  
  def __len__(self):
    result = 0
    for sampler in self.samplers:
      result += len(sampler)
    return result
  
  def __iter__(self):
    iters = [iter(sampler) for sampler in self.samplers]
    def iterator():
      for _ in range(len(self)):
        next_val = None
        while next_val == None:
          it = random.choice(iters)
          try:
            next_val = next(it)
          except StopIteration:
            next_val = None
            iters.remove(it)
        yield next_val
    return iterator()

class LabelPartitionedSampler(object):
  def __init__(self, dataset):
    """Partition of a :class:`Dataset` into multiple class:`Sampler`s, by label.
    
    Arguments
    ---------
    dataset : a :class:`Dataset` to be partitioned by label.
    """
    self.dataset = dataset
    self.labelindices = {}
    self.labels = []
    self.samplers = {}
    for idx in range(len(self.dataset)):
      labelindex = self.dataset[idx]["label"][0][0]
      label = self.dataset.classmap_inv[labelindex]
      if label not in self.labels:
        self.labelindices[label] = []
        self.labels.append(label)
      self.labelindices[label].append(idx)
    for label in self.labels:
      self.samplers[label] = SubsetRandomSampler(
        self.labelindices[label]
      )
  
  def __getitem__(self, label):
    result = None
    if isinstance(label, list):
      samplers = []
      for elem in label:
        samplers.append(self.samplers[elem])
      result = UnionSampler(samplers)
    else:
      result = self.samplers[label]
    return result

class EpisodicSampler(Sampler):
  def __init__(self, dataset,
               batch_size=128, label_size=2,
               shot_size=1, max_episodes=1000):
    """Samples episodes from a dataset.
      
    Arguments
    ---------
    dataset : a :class:`Dataset` to be packed into episodes.

    batch_size : the batch size for each episode.

    label_size : the maximum number of labels per episode.

    shot_size : the maximum size of the support set for each given class.

    max_episodes : the number of episodes per epoch.

    """
    self.dataset = dataset
    self.labelsampler = LabelPartitionedSampler(dataset)
    self.batch_size = batch_size
    self.label_size = label_size
    self.shot_size = shot_size
    self.max_episodes = max_episodes

  def __iter__(self):
    num_episode_labels = random.randrange(2, self.label_size + 1)
    episode_labels = random.sample(self.labelsampler.labels, num_episode_labels)
    sampler = iter(self.labelsampler[episode_labels])
    batchindices = []
    supportindices = []
    for idx in range(self.batch_size):
      batchindices.append(sampler[idx])
    for label in episode_labels:
      labelsampler = self.labelsampler[label]
      for idx in range(self.shot_size):
        supportindices.append(next(labelsampler))
    return iter(batchindices + supportindices)

  def __len__(self):
    return self.max_episodes

class _EpisodicOverlay(object):
  def __init__(self, loader, batch_size):
    """Wraps a DataLoader to separate its batches into batch and support."""
    self.loader = loader
    self.batch_size = batch_size

  def __len__(self):
    return len(self.loader)

  def __iter__(self):
    def iterator():
      for elem in self.loader:
        data, labels = elem.values()
        batch = data[:self.batch_size, :, :, :]
        support = data[self.batch_size:, :, :, :]
        batchlabels = labels[:self.batch_size, :, :]
        supportlabels = labels[self.batch_size:, :, :]
        yield {
          "batch": batch,
          "batchlabels": batchlabels,
          "support": support,
          "supportlabels": supportlabels
        }
    return iterator()

def EpisodicLoader(dataset, batch_size=128, label_size=2,
                   shot_size=1, max_episodes=1000, num_workers=None):
  """Creates a loader for episodic training.
  
  Arguments
  ---------
  dataset : a :class:`Dataset` to be packed into episodes.

  batch_size : the batch size for each episode.

  label_size : the maximum number of labels per episode.

  shot_size : the maximum size of the support set for each given class.

  max_episodes : the number of episodes per epoch.

  num_workers : the number of processes to use for data loading.

  """
  sampler = EpisodicSampler(
    dataset,
    batch_size=batch_size,
    label_size=label_size,
    shot_size=shot_size,
    max_episodes=max_episodes
  )
  loader = DataLoader(
    dataset, batch_sampler=sampler,
    num_workers=num_workers
  )
  return _EpisodicOverlay(loader, batch_size)