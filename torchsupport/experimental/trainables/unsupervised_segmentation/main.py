import torch
import torch.nn as nn
import torch.nn.functional as func

import parser
import model

opt = parser.parse()

if opt.train:
  # TODO: train
elif opt.eval:
  # TODO: evaluate
