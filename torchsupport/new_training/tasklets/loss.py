from torchsupport.new_training.composable import nominal, Update

def join(*names, **weighted_names):
  complete_names = {
    name : 1.0
    for name in names
  }
  complete_names.update(weighted_names)

  @nominal(Update)
  def join_helper(loss, context) -> ("loss", "context"):
    result = 0.0
    for key in complete_names:
      if key in context.losses.dict:
        result += complete_names[key] * context.losses.dict[key]
    return result, context

  return join_helper
