import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import openslide
import random
import numpy as np
from read_roi import read_roi_zip
from skimage.draw import polygon
from PIL import Image, ImageSequence

class RoiImage(Dataset):
  def __init__(self, path, size=(226, 226), transform=lambda x: x):
    self.transform = transform
    with Image.open(path + ".tif") as stack:
      frames = []
      for img in ImageSequence.Iterator(stack):
        frame = torch.tensor(np.array(img).astype(float))
        frames.append(frame.unsqueeze(0))
      self.raw_image = torch.cat(frames, dim=0)
    rois = read_roi_zip(path + ".roi.zip")
    self.rois = [
      torch.tensor(zip(*polygon(
        roi[1]["x"], roi[1]["y"],
        shape=(self.raw_image.size(1),
               self.raw_image.size(2))
      )), dtype=torch.long)
      for roi in rois
    ]

  def __getitem__(self, idx):
    pass # TODO
