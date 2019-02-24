import io

import torch
import numpy as np
from matplotlib import pyplot as plt

def tensorplot(writer, name, fig, step):
  fig.canvas.draw()
  buf = fig.canvas.tostring_rgb()
  ncols, nrows = fig.canvas.get_width_height()
  shape = (nrows, ncols, 3)
  array = np.fromstring(buf, dtype=np.uint8).reshape(shape)
  tensor = torch.Tensor(array.transpose(2, 0, 1))
  writer.add_image(name, tensor, step)
