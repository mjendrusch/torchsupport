import torch
from torchsupport.flex.update.update import Update
from torchsupport.flex.step.step import Step, UpdateStep
from torchsupport.flex.data_distributions.data_distribution import DataDistribution
from torchsupport.flex.context.context import TrainingContext
from torchsupport.data.io import to_device

def canned_supervised(ctx, net, data, losses):
  x, y = data.sample()
  predictions = net(x)
  for idx, (p_val, y_val, loss) in enumerate(zip(
      predictions, y, losses
  )):
    loss_val = loss(p_val, y_val)
    ctx.argmin(**{f"loss {idx}": loss_val})

def SupervisedTraining(net, data, valid_data,
                       losses=None, **kwargs):
  ctx = TrainingContext(kwargs["network_name"], **kwargs)
  net = to_device(net, ctx.device)
  data = DataDistribution(data, batch_size=ctx.batch_size, num_workers=ctx.num_workers)
  valid_data = DataDistribution(valid_data, batch_size=ctx.batch_size, num_workers=ctx.num_workers)
  ctx.checkpoint.add_checkpoint(net=net)
  ctx.loop \
  .add(train=UpdateStep(
    canned_supervised(ctx, net.train(), data, losses),
    Update(net, optimizer=torch.optim.Adam)
  )) \
  .add(valid=Step(
    canned_supervised(ctx, net.eval(), valid_data, losses)
  ))
  return ctx
