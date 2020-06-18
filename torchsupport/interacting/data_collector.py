import torch
import torch.multiprocessing as mp

def _collector_worker(statistics, buffer, distributor,
                      collector, done, piecewise):
  torch.set_num_threads(1)
  while True:
    if done.value:
      break
    result = collector.sample_trajectory()
    trajectory_statistics = collector.compute_statistics(result)
    trajectory = distributor.commit_trajectory(result)

    if piecewise:
      for item in trajectory:
        buffer.append(item)
    else:
      buffer.append(trajectory)

    statistics.update(trajectory_statistics)

class ExperienceCollector:
  def __init__(self, distributor, collector,
               piecewise=True, n_workers=16):
    self.n_workers = n_workers
    self.piecewise = piecewise
    self.distributor = distributor
    self.collector = collector
    self.done = mp.Value("l", 0)
    self.procs = []

  def start(self, statistics, buffer):
    for idx in range(self.n_workers):
      proc = mp.Process(
        target=_collector_worker,
        args=(statistics, buffer, self.distributor,
              self.collector, self.done, self.piecewise)
      )
      self.procs.append(proc)
      proc.start()

  def schema(self):
    return self.distributor.schema(self.collector.schema())

  def join(self):
    self.done.value = 1
    for proc in self.procs:
      proc.join()
    self.procs = []
