
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchsupport.training.training import Training
from torchsupport.reporting.reporting import tensorplot
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabaz_score

from matplotlib import pyplot as plt

from tensorboardX import SummaryWriter

class ClusteringTraining(Training):
  def __init__(self, net, data,
               clustering=KMeans(3),
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
    self.optimizer = optimizer(self.net.parameters())

    self.network_name = network_name
    self.writer = SummaryWriter(network_name)

    self.epoch_id = 0
    self.step_id = 0

  def step(self, data, label, centers):
    self.optimizer.zero_grad()
    attention = self.net(data.to(self.device)).squeeze()
    centers = centers.to(self.device).unsqueeze(0)
    logits = -abs(centers - attention.unsqueeze(2))
    # logits = centers.matmul(attention.unsqueeze(2))
    label = label.long().unsqueeze(1).to(self.device)
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
      for point, *_ in batch_loader:
        latent_point, *_ = self.net(point.to(self.device))
        latent_point = latent_point.to("cpu")
        latent_point = latent_point.reshape(latent_point.size(0), -1)
        embedding.append(latent_point)
      embedding = torch.cat(embedding, dim=0)
    self.net.train()
    return embedding

  def cluster(self, embedding):
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
      print(cluster_centers.shape)
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
      print(cluster_centers.shape)
    counts = [
      labels.count(label)
      for label in range(len(set(labels)))
    ]
    print(counts)
    weights = [1 / counts[label] for label in labels]
    centers = torch.Tensor(cluster_centers)
    return weights, labels, centers

  def _cluster_image(self, labels):
    count = 5
    n_clusters = len(set(labels))
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
      nn.Linear(256, value).to(self.device)
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
      # level_center = self.cluster_embeddings(level_center)
      # level_logits = level_center.matmul(attention.unsqueeze(2))
      level_logits = self.cluster_embeddings[level](attention)
      level_label = label[:, level].long().to(self.device)
      # print(level_center.size())
      # print(level_logits.size())
      # print(level_label.size())
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
      for data, label in self.train_data:
        self.step(data, label, center_hierarchy)
        self.step_id += 1

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

  def vae_loss(self, mean, logvar, reconstruction, target, beta=20, c=0.5):
    mse = func.mse_loss(reconstruction, target)
    # kld = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return mse# + beta * torch.norm(kld - c, 1)

  def step(self, data, label, centers):
    self.optimizer.zero_grad()
    data = data.to(self.device)
    features, mean, logvar = self.net(data)
    std = torch.exp(0.5 * logvar)
    sample = torch.randn_like(std).mul(std).add_(mean)

    reconstruction = self.decoder(sample)

    with torch.no_grad():
      print(data.size(), reconstruction.size())
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
