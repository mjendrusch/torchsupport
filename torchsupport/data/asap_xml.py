class PolygonAnnotation(object):
  def __init__(self, points):
    self.points = points

  def tile_at(self, position, size=(224, 224), origin=(0.5, 0.5)):
    pass ## TODO

class SplineAnnotation(object):
  def __init__(self, points):
    self.points = points

  def tile_at(self, position, size=(224, 224), origin=(0.5, 0.5)):
    pass ## TODO

class PointSetAnnotation(object):
  def __init__(self, points):
    self.points = points

  def tile_at(self, position, size=(224, 224), origin=(0.5, 0.5)):
    pass ## TODO

class CoordinateAnnotation(object):
  def __init__(self, surface_dict):
    self.surface_dict = surface_dict
    self.n_classes = len(self.surface_dict)

  def tile_at(self, position, size=(224, 224), origin=(0.5, 0.5)):
    to_cat = []
    for key in self.surface_dict:
      label_pixels = self.surface_dict[key].tile_at(position, size, origin)
      to_cat.append(label_pixels.unsqueeze(0))
    return torch.cat(to_cat, dim=0)