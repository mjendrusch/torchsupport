from skimage import io
import torch
import numpy as np
import time

def imread(path, type='float32'):
  """Reads a given image from file, returning a float tensor."""
  reading = True
  while reading:
    try:
      image = io.imread(path)
      reading = False
    except OSError as e:
      if e.errno == 121:
        print("Attempting to recover from Remote IO Error ...")
        time.sleep(10)
      else:
        print("Unexpected OSError. Aborting ...")
        raise e
  image = np.array(image).astype(type)
  image = np.transpose(image,(2,0,1))
  image = torch.from_numpy(image)
  return image