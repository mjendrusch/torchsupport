class AbstractHistory(object):
  def remember(self, observation, action, reward, data):
    raise NotImplementedError("Abstract")

  def sample(self, **kwargs):
    raise NotImplementedError("Abstract")

class AbstractEnvironment(object):
  def observe(self, agent):
    raise NotImplementedError("Abstract")

  def perform(self, choice):
    raise NotImplementedError("Abstract")

  def step(self, agent):
    observation = self.observe(agent)
    action, *data = agent.predict(observation)
    choice = agent.act(action)
    reward = self.perform(choice)
    agent.history.remember(observation, action, reward, data)

  def run(self):
    raise NotImplementedError("Abstract")

class AbstractAgent(object):
  def __init__(self, models,
               device="cpu",
               optimizer=None, optimizer_kwargs=None,
               history=None, history_kwargs=None):
    self.models = list(map(
      lambda x: x.to(device),
      models
    ))

    self.history = history
    self.history_kwargs = history_kwargs

    self.optimizer = optimizer([
      model.parameters() for model in self.models
    ], **optimizer_kwargs)

  def sample(self, **local_kwargs):
    return self.history.sample(
      **self.history_kwargs, **local_kwargs
    )

  def predict(self, state):
    raise NotImplementedError("Abstract")

  def act(self, action_distribution):
    raise NotImplementedError("Abstract")

  def update(self, history):
    raise NotImplementedError("Abstract")

class AbstractReinforcementLearning(object):
  def __init__(self, environment, agents):
    self.env = environment.register(agents)
    self.agents = agents
    self.max_episodes = ...
    self.max_steps = ...
    self.optimizer = ...

  def reward(self, state, action):
    raise NotImplementedError("Abstract")

  def gradient_estimate(self, reward, data):
    raise NotImplementedError("Abstract")

  def train(self):
    for episode in range(self.max_episodes):
      self.optimizer.zero_grad()
      episode_rewards = []
      episode_data = []
      for step_id, (state, action, *data) in enumerate(self.env.run()):
        episode_rewards.append(self.reward(state, action))
        episode_data.append(data)
      gradient_estimates = self.gradient_estimate(
        episode_rewards, episode_data
      )
      gradient_estimates.backward()
    raise NotImplementedError("Abstract")
