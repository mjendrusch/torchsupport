import ctypes
from copy import deepcopy
from functools import partial

from skimage import io
import torch
import numpy as np
import time

from torchsupport.data.tensor_provider import TensorProvider
from torchsupport.data.namedtuple import NamedTuple

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

MOVE_REGISTRY = {}
def _move_extension_aux(function, kind=None):
  MOVE_REGISTRY[kind] = function
  return function

def move_extension(kind):
  return partial(_move_extension_aux, kind=kind)

class DeviceMovable():
  def move_to(self, device):
    raise NotImplementedError("Abstract.")

class Detachable():
  def detach(self):
    raise NotImplementedError("Abstract.")

def to_device(data, device):
  if isinstance(data, torch.Tensor):
    return data.to(device)
  if isinstance(data, NamedTuple):
    typ = type(data)
    dict_val = to_device(data.asdict(), device)
    return typ(**dict_val)
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
  if isinstance(data, DeviceMovable):
    return data.move_to(device)
  if type(data) in MOVE_REGISTRY:
    return MOVE_REGISTRY[type(data)](data, device)
  return data

class _MemoDictDetach(dict):
  def get(self, key, default=None):
    result = super().get(key, default)
    if result is default:
      old = ctypes.cast(key, ctypes.py_object).value
      if isinstance(old, (Detachable, torch.Tensor)):
        result = old.detach()
        self[key] = result

    return result

def detach(data):
  memo = _MemoDictDetach()
  return deepcopy(data, memo)

def make_differentiable(data, toggle=True):
  if torch.is_tensor(data) and data.is_floating_point():
    data.requires_grad_(toggle)
  elif isinstance(data, (list, tuple)):
    for item in data:
      make_differentiable(item, toggle=toggle)
  elif isinstance(data, dict):
    for key in data:
      make_differentiable(data[key], toggle=toggle)
  elif isinstance(data, TensorProvider):
    for tensor in data.tensors():
      make_differentiable(tensor, toggle=toggle)
  else:
    pass
