from tensorboardX import SummaryWriter
from torchsupport.new_training.context import ctx

from torchsupport.new_training.zoo.base import OneStepTraining
from torchsupport.new_training.loop.loop import TrainingLoop
from torchsupport.new_training.parametric import Parametric, Step
from torchsupport.new_training.data.data import DataDistribution
from torchsupport.new_training.tasklets.supervised import Supervised
from torchsupport.new_training.parameters.parameters import GradientParameters
from torchsupport.new_training.tasklets.loss import join
from torchsupport.new_training.log.basic import log_losses, log_confusion
from torchsupport.new_training.checkpoint import Checkpoint

def SupervisedTrainingFunction(
    net=None,
    losses=None,
    dataset=None,
    logger=SummaryWriter,
    regularisation=None,
    optimiser=None,
    optimiser_kwargs=None,
    regularisation_weight=1.0,
    batch_size=64,
    num_workers=8,
    network_path=".",
    max_steps=int(1e6),
    report_interval=10,
    checkpoint_interval=1000
):
  optimiser_kwargs = optimiser_kwargs or {}
  logger = logger(network_path)
  data = DataDistribution(
    dataset,
    batch_size=batch_size,
    num_workers=num_workers
  )
  supervised = Supervised(net, losses)
  if regularisation is not None:
    supervised = supervised >> regularisation

  return TrainingLoop(
    Step(
      GradientParameters(net, optimiser, **optimiser_kwargs),
      Parametric(
        data,
        supervised.run,
        supervised.loss >> join(
          supervised_loss=1.0,
          regularisation_loss=regularisation_weight
        )
      )
    ),
    log_losses(logger),
    Checkpoint(network_path, net=net),
    max_steps=max_steps,
    report_interval=report_interval,
    checkpoint_interval=checkpoint_interval
  )

class SupervisedTraining(OneStepTraining):
  def __init__(self,
               net=None,
               losses=None,
               dataset=None,
               optimiser=None,
               optimiser_kwargs=None,
               regularisation=None,
               regularisation_weight=1.0,
               **kwargs):
    super().__init__(**kwargs)

    self.net = net
    self.optimiser = optimiser
    self.optimiser_kwargs = optimiser_kwargs
    self.supervised = Supervised(net, losses)
    self.regularisation_weight = regularisation_weight
    if regularisation:
      self.supervised = self.supervised >> regularisation
    self.set_dependents(
      data=DataDistribution(
        dataset,
        batch_size=self.batch_size,
        num_workers=self.num_workers
      ),
      parameters=GradientParameters(
        self.net, self.optimiser, **self.optimiser_kwargs
      ),
      tasklet=self.supervised
    )

  def log(self):
    return log_losses(self.logger)

  def loss(self):
    return self.supervised.loss >> join(
      supervised_loss=1.0,
      regularisation_loss=self.regularisation_weight
    )

  def checkpoint(self):
    return Checkpoint(self.network_path, net=self.net)
