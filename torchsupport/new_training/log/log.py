import inspect

from torchsupport.new_training.composable import Run

class log:
  def __init__(self, function):
    self.spec = inspect.getfullargspec(function)
    self.requires = {
      name : name
      for name in self.spec.args[1:]
    }
    self.function = function

  @classmethod
  def at(cls, path=None):
    def decorator(function):
      def log_inner(logger):
        spec = inspect.getfullargspec(function)
        requires = {
          name : name
          for name in spec.args[1:]
        }
        def log_inner_inner(**kwargs):
          function(logger, **kwargs)
        return Run(
          log_inner_inner,
          path=path,
          requires=requires,
          provides=[]
        )
      return log_inner
    return decorator

  def __call__(self, logger):
    def log_inner_inner(**kwargs):
      self.function(logger, **kwargs)
    return Run(
      log_inner_inner,
      path=None,
      requires=self.requires,
      provides=[]
    )
