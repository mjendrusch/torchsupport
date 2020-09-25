import torch

from torchsupport.new_training.tasklets.tasklet import Tasklet

class Supervised(Tasklet):
  def __init__(self, net, losses):
    super().__init__()
    self.net = net
    self.losses = losses

  def run(self, inputs):
    predictions = self.net(inputs)
    return predictions

  def loss(self, predictions, ground_truth):
    result = 0.0
    if torch.is_tensor(predictions):
      predictions = [predictions]
    if torch.is_tensor(ground_truth):
      ground_truth = [ground_truth]
    subresults = []
    for loss, prediction, target in zip(
        self.losses, predictions, ground_truth
    ):
      value = loss(prediction, target)
      subresults.append(value)
      result += value

    self.store(
      predictions=predictions,
      ground_truth=ground_truth,
      sublosses=subresults,
      total_loss=result
    )

    return result

  def step(self, inputs, ground_truth):
    predictions = self.run(inputs)
    loss = self.loss(predictions, ground_truth)
    return loss
