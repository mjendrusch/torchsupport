from skimage import io
import torch

def imread(path, type='float32'):
  """Reads a given image from file, returning a float tensor."""
  image = io.imread(path)
  image = np.array(image).astype(type)
  image = np.transpose(image,(2,0,1))
  image = torch.from_numpy(image)
  return image