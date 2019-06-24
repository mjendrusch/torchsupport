
import random
from itertools import islice
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchsupport.training.training import Training
from torchsupport.reporting.reporting import tensorplot
from torchsupport.data.io import netwrite
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabaz_score

from matplotlib import pyplot as plt

from tensorboardX import SummaryWriter

class ClusteringTraining(Training):
  def __init__(self, net, data,
               clustering=KMeans(3),
               order_less=True,
               loss=nn.CrossEntropyLoss(),
               optimizer=torch.optim.Adam,
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network"):
    super(ClusteringTraining, self).__init__()
    self.net = net.to(device)
    self.clustering = clustering
    self.data = data
    self.train_data = None
    self.loss = loss
    self.max_epochs = max_epochs
    self.batch_size = batch_size
    self.device = device

    self.order_less = order_less

    if not order_less:
      self.classifier = nn.Linear(256, 50)
      self.classifier = self.classifier.to(self.device)
    else:
      self.embedding = nn.Linear(256, 256)
      self.embedding = self.embedding.to(self.device)

    self.optimizer = optimizer(self.net.parameters())

    self.network_name = network_name
    self.writer = SummaryWriter(network_name)

    self.epoch_id = 0
    self.step_id = 0

  def checkpoint(self):
    the_net = self.net
    if isinstance(the_net, torch.nn.DataParallel):
      the_net = the_net.module
    netwrite(
      the_net,
      f"{self.network_name}-encoder-epoch-{self.epoch_id}-step-{self.step_id}.torch"
    )
    self.each_checkpoint()

  def step(self, data, label, centers):
    self.optimizer.zero_grad()
    attention = self.net(data.to(self.device)).squeeze()
    centers = centers.to(self.device).unsqueeze(0)
    if self.order_less:
      center_embedding = self.embedding(centers.squeeze())
      logits = center_embedding.matmul(attention.unsqueeze(2)).squeeze()
    else:
      logits = self.classifier(attention.reshape(attention.size(0), -1))
    label = label.long().to(self.device)
    loss_val = self.loss(logits, label)
    loss_val.backward()
    self.writer.add_scalar("cluster assignment loss", float(loss_val), self.step_id)
    self.optimizer.step()
    self.each_step()

  def embed_all(self):
    self.net.eval()
    with torch.no_grad():
      embedding = []
      batch_loader = DataLoader(
        self.data,
        batch_size=self.batch_size,
        shuffle=False
      )
      for point, *_ in islice(batch_loader, 5000 // self.batch_size):
        latent_point = self.net(point.to(self.device))
        latent_point = latent_point.to("cpu")
        latent_point = latent_point.reshape(latent_point.size(0), -1)
        embedding.append(latent_point)
      embedding = torch.cat(embedding, dim=0)
    self.net.train()
    return embedding

  def cluster(self, embedding):
    print(embedding.size())
    fit = self.clustering.fit(embedding.squeeze())
    labels = list(fit.labels_)
    try:
      cluster_centers = fit.cluster_centers_
    except:
      cluster_centers = [
        embedding[(labels == label).astype(int)].mean(dim=0).squeeze().unsqueeze(0).numpy()
        for label in set(labels)
      ]
      cluster_centers = np.concatenate(cluster_centers, axis=0)
    if len(set(labels)) == 1:
      N = random.randint(2, 10)
      labels = [random.choice(list(range(N))) for label in labels]
      offsets = [
        np.random.randn(*cluster_centers.shape) * 2.0
        for _ in range(N)
      ]
      cluster_centers = np.concatenate([
        cluster_centers + offsets[idx]
        for idx in range(N)
      ], axis=0).squeeze()
    counts = [
      labels.count(label)
      for label in range(len(set(labels)))
    ]
    weights = [1 / counts[label] for label in labels]
    centers = torch.Tensor(cluster_centers)
    return weights, labels, centers

  def _cluster_image(self, labels):
    count = 10
    n_clusters = 50#max(list(set(labels)))
    indices = list(range(len(labels)))
    random.shuffle(indices)
    cluster_done = [False for _ in range(n_clusters)]
    cluster_images = [[] for _ in range(n_clusters)]
    for index in indices:
      label = labels[index]
      if all(cluster_done):
        break
      if len(cluster_images[label]) < count:
        img, *_ = self.data[index]
        img = img - img.min()
        img = img / img.max()
        cluster_images[label].append(img)
      else:
        cluster_done[label] = True
    rows = [
      torch.cat(image_list, dim=2)
      for image_list in cluster_images
      if image_list
    ]
    for idx, row in enumerate(rows):
      self.writer.add_image(f"cluster samples {idx}", row, self.step_id)

  def _cluster_plot(self, embedding, labels):
    silhouette = silhouette_score(embedding.squeeze(), labels)
    chs = calinski_harabaz_score(embedding.squeeze(), labels)
    dbs = davies_bouldin_score(embedding.squeeze(), labels)

    n_labels = len(set(labels))

    self.writer.add_scalar(f"silhouette {n_labels}", silhouette, self.step_id)
    self.writer.add_scalar(f"chs {n_labels}", chs, self.step_id)
    self.writer.add_scalar(f"dbs {n_labels}", dbs, self.step_id)

    indices = list(range(len(labels)))
    random.shuffle(indices)
    samples_to_plot = indices[:1000]
    sample_labels = [labels[idx] for idx in samples_to_plot]
    sample_embedding = embedding[samples_to_plot]
    pca = PCA(2).fit_transform(sample_embedding.squeeze())
    fig, ax = plt.subplots()
    ax.scatter(pca[:, 0], pca[:, 1], c=sample_labels, cmap="tab20")
    self.writer.add_figure(f"clustering {n_labels}", fig, self.step_id)

  def each_cluster(self, embedding, labels):
    self._cluster_image(labels)
    self._cluster_plot(embedding, labels)

  def train(self):
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      embedding = self.embed_all()
      weights, labels, centers = self.cluster(embedding)

      self.each_cluster(embedding, labels)

      self.data.labels = labels
      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=8,
        sampler=WeightedRandomSampler(weights, len(self.data) * 4, replacement=True)
      )
      for data, label in self.train_data:
        self.step(data, label, centers)
        self.step_id += 1
        if self.step_id % 50 == 0:
          self.checkpoint()

    return self.net

class ClusterAETraining(ClusteringTraining):
  def __init__(self, encoder, decoder, data,
               n_clusters=10,
               center_size=1024,
               gamma=0.1,
               alpha=0.1,
               clustering=KMeans(3),
               loss=nn.CrossEntropyLoss(),
               optimizer=torch.optim.Adam,
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network"):
    super(ClusterAETraining, self).__init__(
      encoder, data,
      clustering=clustering,
      loss=loss,
      optimizer=optimizer,
      max_epochs=max_epochs,
      batch_size=batch_size,
      device=device,
      network_name=network_name
    )
    self.decoder = decoder.to(device)

    self.alpha = 0.1
    self.gamma = gamma
    self.centers = torch.rand(
      n_clusters,
      center_size,
      requires_grad=True,
      device=device
    )
    with torch.no_grad():
      self.centers.mul_(
        torch.tensor(2).float().to(device)
      ).add_(
        torch.tensor(-1).float().to(device)
      )

    self.center_optimizer = optimizer(
      [self.centers]
    )

    self.optimizer = optimizer(
      list(self.net.parameters()) +
      list(self.decoder.parameters())
    )

  def hardening_loss(self, predictions):
    diff = predictions.unsqueeze(1) - self.centers.unsqueeze(0)
    assignment = 1 / (1 + ((diff) ** 2).sum(dim=2))
    assignment = assignment / (assignment.sum(dim=1, keepdim=True) + 1e-20)
    hardening = assignment ** 2 / (assignment.sum(dim=0, keepdim=True) + 1e-20)
    hardening = hardening / (hardening.sum(dim=1, keepdim=True) + 1e-20)
    loss = hardening * (torch.log(hardening / (assignment + 1e-20) + 1e-20))
    loss = loss.sum(dim=1).mean(dim=0)

    self.writer.add_scalar("hardening loss", float(loss), self.step_id)

    return 1000 * loss

  def cluster_loss(self, data, alpha=0.0):
    distance = torch.norm(
      data.unsqueeze(1) - self.centers.unsqueeze(0),
      2, dim=2
    )
    gamma = distance * func.softmax(-alpha * (distance - distance.min(dim=1, keepdim=True)[0]), dim=1)
    result = gamma.sum(dim=1).mean()

    self.writer.add_scalar("cluster loss", float(result), self.step_id)

    return result

  def ae_loss(self, predictions, target):
    loss = func.mse_loss(predictions, target)

    self.writer.add_scalar("reconstruction loss", float(loss), self.step_id)

    return loss

  def regularization(self, predictions):
    pred_norm = torch.norm(predictions, 2, dim=1)
    center_norm = torch.norm(self.centers, 2, dim=1)
    pred_norm = ((pred_norm - 1) ** 2).mean()
    center_norm = ((center_norm - 1) ** 2).mean()
    loss = pred_norm + center_norm

    self.writer.add_scalar("regularization loss", float(loss), self.step_id)

    return loss

  def each_cluster(self):
    pass

  def step(self, data):
    self.optimizer.zero_grad()
    data = data.to(self.device)

    features = self.net(data)

    reconstruction = self.decoder(features)

    with torch.no_grad():
      im_vs_rec = torch.cat(
        (
          data[0].cpu(),
          reconstruction[0].cpu()
        ),
        dim=2
      ).numpy()
      im_vs_rec = im_vs_rec - im_vs_rec.min()
      im_vs_rec = im_vs_rec / im_vs_rec.max()
      self.writer.add_image("im vs rec", im_vs_rec, self.step_id)

    loss_val = self.ae_loss(reconstruction, data)
    loss_val += self.gamma * self.cluster_loss(
      features.reshape(features.size(0), -1),
      self.alpha
    )
    loss_val += self.regularization(features.reshape(features.size(0), -1))
    loss_val.backward()

    self.writer.add_scalar("cluster assignment loss", float(loss_val), self.step_id)
    self.optimizer.step()
    self.center_optimizer.step()
    self.each_step()

  def train(self):
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id

      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=8, shuffle=True
      )
      for internal_epoch in range(1):
        for data, *_ in islice(self.train_data, 100):
          self.step(data)
          self.step_id += 1
          if self.step_id % 50 == 0:
            self.checkpoint()

      self.each_cluster()
      self.alpha *= float(np.power(2.0, (-(np.log(epoch_id + 1) ** 2))))

    return self.net

class DEPICTTraining(ClusteringTraining):
  def __init__(self, encoder, decoder, classifier, data,
               clustering=KMeans(3),
               loss=nn.CrossEntropyLoss(),
               optimizer=torch.optim.Adam,
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network"):
    super(DEPICTTraining, self).__init__(
      encoder, data,
      clustering=clustering,
      loss=loss,
      optimizer=optimizer,
      max_epochs=max_epochs,
      batch_size=batch_size,
      device=device,
      network_name=network_name
    )
    self.decoder = decoder.to(device)
    self.classifier = classifier.to(device)

    self.optimizer = optimizer(
      list(self.net.parameters()) +
      list(self.decoder.parameters()) +
      list(self.classifier.parameters())
    )

  def expectation(self):
    self.net.eval()
    with torch.no_grad():
      embedding = []
      batch_loader = DataLoader(
        self.data,
        batch_size=self.batch_size,
        shuffle=False
      )
      for point, *_ in batch_loader:
        features, mean, logvar = self.net(point.to(self.device))
        std = torch.exp(0.5 * logvar)
        sample = torch.randn_like(std).mul(std).add_(mean)
        latent_point = func.adaptive_avg_pool2d(sample, 1)

        latent_point = latent_point
        latent_point = latent_point.reshape(latent_point.size(0), -1)
        embedding.append(latent_point)
      embedding = torch.cat(embedding, dim=0)
      expectation = self.classifier(embedding)
    self.net.train()
    return expectation.to("cpu"), embedding.to("cpu")

  def vae_loss(self, mean, logvar, reconstruction, target, beta=20, c=0.5):
    mse = func.mse_loss(reconstruction, target)
    kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    
    self.writer.add_scalar("mse loss", float(mse), self.step_id)
    self.writer.add_scalar("kld loss", float(kld), self.step_id)
    self.writer.add_scalar("kld-c loss", float(torch.norm(kld - c, 2)), self.step_id)

    return mse + beta * torch.norm(kld - c, 2)

  def depict_loss(self, logits, expected_logits):
    expected_p = func.softmax(expected_logits, dim=1)
    p = func.softmax(logits, dim=1)
    q = (expected_p + 1e-20) / (torch.sqrt(expected_p.sum(dim=0, keepdim=True)) + 1e-20)
    q = (q + 1e-20) / (q.sum(dim=1, keepdim=True) + 1e-20)
    return torch.mean(- q * torch.log(p + 1e-20))

  def step(self, data, expected_logits, centers):
    self.optimizer.zero_grad()
    data = data.to(self.device)

    features, mean, logvar = self.net(data)
    std = torch.exp(0.5 * logvar)
    sample = torch.randn_like(std).mul(std).add_(mean)

    reconstruction = self.decoder(sample)

    with torch.no_grad():
      im_vs_rec = torch.cat(
        (
          data[0].cpu(),
          reconstruction[0].cpu()
        ),
        dim=2
      ).numpy()
      im_vs_rec = im_vs_rec - im_vs_rec.min()
      im_vs_rec = im_vs_rec / im_vs_rec.max()
      self.writer.add_image("im vs rec", im_vs_rec, self.step_id)

    loss_val = torch.tensor(0.0).to(self.device)

    loss_val += self.vae_loss(
      mean, logvar, reconstruction, data,
      beta=20, c=0.5 + (50 - 0.5) * self.step_id * 0.00001
    )

    sample_global = func.adaptive_avg_pool2d(sample, 1)
    logits = self.classifier(sample_global.squeeze())

    depict_loss = self.depict_loss(logits, expected_logits.to(self.device))

    loss_val += depict_loss
    loss_val.backward()

    self.writer.add_scalar("cluster assignment loss", float(loss_val), self.step_id)
    self.optimizer.step()
    self.each_step()

  def train(self):
    expectation, embedding = self.expectation()
    weights, labels, centers = self.cluster(embedding)
    self.data.labels = torch.zeros_like(expectation)
    self.data.labels[expectation.argmax(dim=1)] = 1

    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id

      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=8,
        sampler=WeightedRandomSampler(weights, len(self.data) * 4, replacement=True)
      )
      for data, expected_logits in self.train_data:
        self.step(data, expected_logits, centers)
        self.step_id += 1

      expectation, embedding = self.expectation()
      labels = expectation.argmax(dim=1).to("cpu").squeeze()
      self.each_cluster(
        expectation.to("cpu"),
        labels.numpy()
      )
      self.data.labels = expectation.to("cpu").squeeze()

    return self.net

class HierarchicalClusteringTraining(ClusteringTraining):
  def __init__(self, net, data,
               optimizer=torch.optim.Adam,
               loss=nn.CrossEntropyLoss(),
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               depth=[5, 10, 50]):
    super(HierarchicalClusteringTraining, self).__init__(
      net, data,
      clustering=None,
      loss=loss,
      optimizer=optimizer,
      max_epochs=max_epochs,
      batch_size=batch_size,
      device=device,
      network_name=network_name
    )

    self.depth = depth
    self.clusterings = [
      AgglomerativeClustering(value)
      for value in depth
    ]

    self.cluster_embeddings = nn.ModuleList([
      nn.Linear(128, value).to(self.device)
      for value in depth
    ])

    self.optimizer = optimizer(
      list(net.parameters()) + list(self.cluster_embeddings.parameters())
    )

  def step(self, data, label, centers):
    self.optimizer.zero_grad()
    attention = self.net(data.to(self.device)).squeeze()
    loss_val = torch.tensor(0.0).to(self.device)
    for level, _ in enumerate(self.clusterings):
      level_center = centers[level].to(self.device).unsqueeze(0)
      level_logits = self.cluster_embeddings[level](attention)
      level_label = label[:, level].long().to(self.device)
      level_loss = self.loss(level_logits, level_label)
      loss_val += level_loss / np.log(self.depth[level])
    loss_val.backward()
    self.writer.add_scalar("cluster assignment loss", float(loss_val), self.step_id)
    self.optimizer.step()
    self.each_step()

  def each_cluster(self, embedding, labels):
    self._cluster_image(list(labels[0].squeeze()))
    for level, label in enumerate(labels):
      self._cluster_plot(embedding, list(label.squeeze()))

  def train(self):
    for epoch_id in range(self.max_epochs):
      self.epoch_id = epoch_id
      embedding = self.embed_all()

      label_hierarchy = []
      center_hierarchy = []
      for clustering in self.clusterings:
        self.clustering = clustering
        weights, labels, centers = self.cluster(embedding)
        label_hierarchy.append(np.expand_dims(labels, axis=1))
        center_hierarchy.append(centers)
      self.each_cluster(embedding, label_hierarchy)
      label_hierarchy = np.concatenate(label_hierarchy, axis=1)

      self.data.labels = label_hierarchy
      self.train_data = None
      self.train_data = DataLoader(
        self.data, batch_size=self.batch_size, num_workers=0,
        sampler=WeightedRandomSampler(weights, min(20000, len(self.data)), replacement=True)
      )
      for inner_epoch in range(1):
        for data, label in self.train_data:
          self.step(data, label, center_hierarchy)
          self.step_id += 1
        self.checkpoint()

    return self.net

class VAEClusteringTraining(HierarchicalClusteringTraining):
  def __init__(self, encoder, decoder, data,
               optimizer=torch.optim.Adam,
               loss=nn.CrossEntropyLoss(),
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               depth=[5, 10, 50]):
    super(VAEClusteringTraining, self).__init__(
      encoder, data,
      depth=depth,
      loss=loss,
      optimizer=optimizer,
      max_epochs=max_epochs,
      batch_size=batch_size,
      device=device,
      network_name=network_name
    )

    self.decoder = decoder.to(device)

    self.optimizer = optimizer(
      list(encoder.parameters()) +
      list(decoder.parameters()) +
      list(self.cluster_embeddings.parameters())
    )

  def checkpoint(self):
    the_net = self.net
    if isinstance(the_net, torch.nn.DataParallel):
      the_net = the_net.module
    netwrite(
      the_net,
      f"{self.network_name}-encoder-epoch-{self.epoch_id}-step-{self.step_id}.torch"
    )

    the_net = self.decoder
    if isinstance(the_net, torch.nn.DataParallel):
      the_net = the_net.module
    netwrite(
      the_net,
      f"{self.network_name}-decoder-epoch-{self.epoch_id}-step-{self.step_id}.torch"
    )

    for idx, classifier in enumerate(self.cluster_embeddings):
      the_net = classifier
      if isinstance(the_net, torch.nn.DataParallel):
        the_net = the_net.module
      netwrite(
        the_net,
        f"{self.network_name}-classifier-{idx}-epoch-{self.epoch_id}-step-{self.step_id}.torch"
      )
    self.each_checkpoint()

  def vae_loss(self, mean, logvar, reconstruction, target, beta=20, c=0.5):
    mse = func.mse_loss(reconstruction, target)
    kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return mse + beta * torch.norm(kld - c, 1)

  def step(self, data, label, centers):
    self.optimizer.zero_grad()
    data = data.to(self.device)
    features, mean, logvar = self.net(data)
    std = torch.exp(0.5 * logvar)
    sample = torch.randn_like(std).mul(std).add_(mean)

    reconstruction = self.decoder(sample)

    with torch.no_grad():
      weirdness = (sample[0] + sample[1]).unsqueeze(0) / 2
      self.decoder.eval()
      weird_reconstruction = self.decoder(weirdness)
      self.decoder.train()
      im_vs_rec = torch.cat(
        (
          data[0].cpu(),
          data[1].cpu(),
          reconstruction[0].cpu(),
          reconstruction[1].cpu(),
          weird_reconstruction[0].cpu()
        ),
        dim=2
      ).numpy()
      im_vs_rec = im_vs_rec - im_vs_rec.min()
      im_vs_rec = im_vs_rec / im_vs_rec.max()
      self.writer.add_image("im vs rec", im_vs_rec, self.step_id)

    loss_val = self.vae_loss(
      mean, logvar, reconstruction, data,
      beta=1000, c=self.step_id * 0.00001
    )
    self.writer.add_scalar("reconstruction loss", float(loss_val), self.step_id)

    attention = features.reshape(features.size(0), -1).squeeze()
    for level, _ in enumerate(self.clusterings):
      level_center = centers[level].to(self.device).unsqueeze(0)
      level_logits = self.cluster_embeddings[level](attention)
      level_label = label[:, level].long().to(self.device)
      level_loss = self.loss(level_logits, level_label)
      loss_val += 0.1 * level_loss / np.log(self.depth[level])
    loss_val.backward()
    self.writer.add_scalar("total loss", float(loss_val), self.step_id)
    self.optimizer.step()
    self.each_step()

class RegularizedClusteringTraining(HierarchicalClusteringTraining):
  def __init__(self, encoder, decoder, data,
               optimizer=torch.optim.Adam,
               loss=nn.CrossEntropyLoss(),
               max_epochs=50,
               batch_size=128,
               device="cpu",
               network_name="network",
               depth=[5, 10, 50]):
    super(RegularizedClusteringTraining, self).__init__(
      encoder, data,
      depth=depth,
      loss=loss,
      optimizer=optimizer,
      max_epochs=max_epochs,
      batch_size=batch_size,
      device=device,
      network_name=network_name
    )

    self.decoder = decoder.to(device)

    self.optimizer = optimizer(
      list(encoder.parameters()) +
      list(decoder.parameters()) +
      list(self.cluster_embeddings.parameters())
    )

  def checkpoint(self):
    the_net = self.net
    if isinstance(the_net, torch.nn.DataParallel):
      the_net = the_net.module
    netwrite(
      the_net,
      f"{self.network_name}-encoder-epoch-{self.epoch_id}-step-{self.step_id}.torch"
    )

    the_net = self.decoder
    if isinstance(the_net, torch.nn.DataParallel):
      the_net = the_net.module
    netwrite(
      the_net,
      f"{self.network_name}-decoder-epoch-{self.epoch_id}-step-{self.step_id}.torch"
    )

    for idx, classifier in enumerate(self.cluster_embeddings):
      the_net = classifier
      if isinstance(the_net, torch.nn.DataParallel):
        the_net = the_net.module
      netwrite(
        the_net,
        f"{self.network_name}-classifier-{idx}-epoch-{self.epoch_id}-step-{self.step_id}.torch"
      )
    self.each_checkpoint()

  def ae_loss(self, reconstruction, target):
    mse = func.mse_loss(reconstruction, target)
    return mse

  def step(self, data, label, centers):
    self.optimizer.zero_grad()
    data = data.to(self.device)
    sample = self.net(data)
    sample += 0.05 * torch.randn_like(sample)
    reconstruction = self.decoder(sample)

    with torch.no_grad():
      weirdness = (sample[0] + sample[1]).unsqueeze(0) / 2
      self.decoder.eval()
      weird_reconstruction = self.decoder(weirdness)
      self.decoder.train()
      im_vs_rec = torch.cat(
        (
          data[0].cpu(),
          data[1].cpu(),
          reconstruction[0].cpu(),
          reconstruction[1].cpu(),
          weird_reconstruction[0].cpu()
        ),
        dim=2
      ).numpy()
      im_vs_rec = im_vs_rec - im_vs_rec.min()
      im_vs_rec = im_vs_rec / im_vs_rec.max()
      self.writer.add_image("im vs rec", im_vs_rec, self.step_id)

    loss_val = self.ae_loss(
      reconstruction, data
    )
    self.writer.add_scalar("reconstruction loss", float(loss_val), self.step_id)

    attention = sample.reshape(sample.size(0), -1).squeeze()
    for level, _ in enumerate(self.clusterings):
      level_center = centers[level].to(self.device).unsqueeze(0)
      level_logits = self.cluster_embeddings[level](attention)
      level_label = label[:, level].long().to(self.device)
      level_loss = self.loss(level_logits, level_label)
      loss_val += 0.1 * level_loss / np.log(self.depth[level])
    loss_val.backward()
    self.writer.add_scalar("total loss", float(loss_val), self.step_id)
    self.optimizer.step()
    self.each_step()
