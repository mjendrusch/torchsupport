import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset
import openslide
import random
import numpy as np

class SlideImage(object):
  def __init__(self, path):
    self.slide = openslide.OpenSlide(path)

  def _tile_at_impl(self, position, level, size, origin):
    to_cat = []
    for lv in level:
      size_off = (
        size[0] * self.slide.level_downsample[lv],
        size[1] * self.slide.level_downsample[lv]
      )
      start = (
        int(position[0] - origin[0] * size_off[0]),
        int(position[1] - origin[1] * size_off[1])
      )
      image = self.slide.read_region(start, level, size)
      image = np.array(image).astype(type)
      image = np.transpose(image,(0,1,2))
      image = torch.from_numpy(image)
      to_cat.append(image.unsqueeze(0))
    result = torch.cat(to_cat, dim=0)
    return result

  def tile_at(self, position, level=0, size=(224, 224), origin=(0.5, 0.5)):
    if isinstance(level, list) or isinstance(level, tuple):
      return self._tile_at_impl(position, level, size, origin)
    else:
      return self._tile_at_impl(position, [level], size, origin)

  def regular_tiling(self, level=0, size=(224, 224)):
    dimensions = self.slide.dimensions
    n_tiles = (dimensions[0] // size[0], dimensions[1] // size[1])
    for idx in range(n_tiles[0]):
      x_pos = idx * size[0]
      for idy in range(n_tiles[1]):
        y_pos = idy * size[1]
        yield self.tile_at((x_pos, y_pos), level=level, size=size, origin=(0, 0))

  def random_tiling(self, count, level=0, size=(224, 224)):
    dimensions = self.slide.dimensions
    lower_x, lower_y = 0, 0
    upper_x, upper_y = dimensions[0] - size[0], dimensions[1] - size[1]
    for idx in range(count):
      rand_x, rand_y = random.randint(lower_x, upper_x), random.randint(lower_y, upper_y)
      yield self.tile_at((rand_x, rand_y), level=level, size=size, origin=(0, 0))

class SingleSlideData(Dataset):
  def __init__(self, path, size=(224, 224), level=0, transform=lambda x: x):
    self.transform = transform
    self.slide = SlideImage(path)
    self.dims = self.slide.slide.dimensions
    self.dims = (dims[0] - size[0], dims[1] - size[1])
    self.size = size
    self.level = level

  def __len__(self):
    return self.dims[0] * self.dims[1]

  def __getitem__(self, index):
    x_pos = index // self.dims[0]
    y_pos = index % self.dims[0]
    tile = self.slide.tile_at((x_pos, y_pos), level=self.level, size=self.size, origin=(0, 0))
    if self.transform != None:
      tile = self.transform(tile)
    return tile

def MultiSlideData(self, paths, size=(224, 224), level=0, transform=lambda x: x):
  datasets = []
  for path in paths:
    datasets.append(SingleSlideData(path, size=size, level=level, transform=transform))
  return ConcatDataset(datasets)
