import torch
import torch.nn
import torch.nn.functional as func
from torch.utils.data import Dataset

import torchsupport.modules.nodegraph as ng
import torchsupport.data.graph as gdata

class QM9(Dataset):
  def __init__(self):
    # self.data = 

