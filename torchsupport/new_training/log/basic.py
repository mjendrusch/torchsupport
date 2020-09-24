import torch

from matplotlib import pyplot as plt

from torchsupport.new_training.log.log import log

@log
def log_losses(logger, loss, context):
  step_id = context.parameters.step_id
  print(float(loss))
  logger.add_scalar("total_loss", float(loss), step_id)
  for loss_name in context.losses.dict:
    loss_val = context.losses.dict[loss_name]
    if isinstance(loss_val, (list, tuple)):
      for idx, item in enumerate(loss_val):
        logger.add_scalar(f"{loss_name}_{idx}", float(item), step_id)
    else:
      logger.add_scalar(loss_name, float(loss_val), step_id)

@log.at(path=["context"])
def log_confusion(logger, parameters, predictions, ground_truth):
  step_id = parameters.step_id
  if torch.is_tensor(predictions):
    predictions = [predictions]
  if torch.is_tensor(ground_truth):
    ground_truth = [ground_truth]
  for idx, (predicted, real) in enumerate(zip(predictions, ground_truth)):

    confusion = torch.zeros(predicted.size(1), predicted.size(1))
    for p, r in zip(predicted, real):
      value = p.argmax(dim=0)
      confusion[real, value] += 1

    fig, ax = plt.subplots()
    show = ax.matshow(confusion.numpy())
    plt.colorbar(show)

    logger.add_figure(f"confusion {idx}", fig, step_id)
    plt.close("all")
