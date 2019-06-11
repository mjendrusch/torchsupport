from skimage import io
import torch
import numpy as np
import time

import os

def imread(path, type='float32'):
  """Reads a given image from file, returning a `Tensor`.
  
  Args:
    path (str): path to an image file.
    type (str): the desired type of the output tensor, defaults to 'float32'.
  """
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

def stackread(path, type='float32'):
  """Reads a given image from file, returning a `Tensor`.
  
  Args:
    path (str): path to an image file.
    type (str): the desired type of the output tensor, defaults to 'float32'.
  """
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
  image = np.transpose(image,(0,1,2))
  image = torch.from_numpy(image)
  return image

def netwrite(network, path):
  """Writes a given neural network to a file.
  
  Args:
    network (nn.Module): the network to be written, needs to inherit from `Module`.
    path (str): path to the file where the network will be written.
  """
  writing = True
  while writing:
    try:
      torch.save(network.state_dict(), path)
      writing = False
    except OSError as e:
      if e.errno == 121:
        print("Attempting to recover from Remote IO Error ...")
        time.sleep(10)
      else:
        print("Unexpected OSError. Aborting ...")
        raise e

def netread(network, path):
  """Tries to read neural network weights from a file.

  Args:
    network (nn.Module): network to be saved.
    path (str): path at which the network will be saved.
  """

  if not os.path.isfile(path):
    print("No checkpoint found. Resuming without checkpoint ...")
    return

  reading = True
  while reading:
    try:
      state_dict = torch.load(path, map_location='cpu')
      network.load_state_dict(state_dict, strict=True)
      reading = False
    except OSError as e:
      if e.errno == 121:
        print("Attempting to recover from Remote IO Error ...")
        time.sleep(10)
      else:
        print("Unexpected OSError. Aborting ...")
        raise e

def to_device(data, device):
  if isinstance(data, torch.Tensor):
    return data.to(device)
  if isinstance(data, (list, tuple)):
    return [
      to_device(point, device)
      for point in data
    ]
  if isinstance(data, dict):
    return {
      key : to_device(data[key], device)
      for key in data
    }
