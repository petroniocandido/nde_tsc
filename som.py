import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from datetime import date
import copy

import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from nde_tsc.common import checkpoint, checkpoint_all


class SOM(nn.Module):
  def __init__(self, **kwargs):
    super(SOM, self).__init__()
    self.width = kwargs.get('width', 10)
    self.height = kwargs.get('height', 10)
    self.size = self.width * self.height
    self.num_dim = kwargs.get('num_dim', 10)
    self.learning_rate = kwargs.get('learning_rate', 0.3)
    self.neighborhood_radius = kwargs.get('neighborhood', max(self.width, self.height) / 2.0)
    self.num_classes = kwargs.get('num_classes', 10)
    self.weights = torch.rand(self.size, self.num_dim, requires_grad=False)
    self.class_weights = torch.zeros(self.num_classes, requires_grad = False)
    self.probabilities = torch.zeros(self.size, requires_grad=False)
    self.conditional_probabilities = torch.zeros(self.num_classes, self.size, requires_grad=False)
    index = np.array([(j,i) for j in range(self.height) for i in range(self.width)])
    self.index = torch.LongTensor(index)

  def knn(self, x, k=3):
    dist = torch.norm(self.weights - x, dim=1, p=None)
    dist = torch.exp(dist) / torch.sum(torch.exp(dist))
    knn = dist.topk(3, largest=False)
    distances = dist[knn.indices]
    return knn.values, knn.indices, distances

  def forward(self, x, **kwargs):
    k=kwargs.get('k',4)
    mode=kwargs.get('mode', 'marginal')

    _, indexes, distances = self.knn(x, k=k)
    distances = distances / torch.sum(distances)
    if mode == 'marginal':
      prob = torch.stack([self.probabilities[indexes[i]] * distances[i] for i in range(k)])
    elif mode == 'conditional':
      label = kwargs.get('label', 0)
      prob = torch.stack([self.conditional_probabilities[label][indexes[i]] * distances[i] for i in range(k)])
    return torch.sum(prob)

  def activation_map(self, x):
    dist = torch.norm(self.weights - x, dim=1, p=None)
    dist = torch.sum(dist) - (dist / torch.sum(dist))
    return dist.view(self.width, self.height)

  def conditional_probability(self, x, k = 3):
    ret = torch.zeros(self.num_classes)
    for label in range(self.num_classes):
      p = self.forward(x, mode = 'conditional', label = label, k = k)
      ret[label] = self.class_weights[label] * p

    return ret


  def backward(self, x, **kwargs):

    epocas = kwargs.get('epocas', 100)
    it = kwargs.get('it',0)
    mode = kwargs.get('mode','som')

    if mode == 'som':

      x_stacked = torch.stack([x] * self.size).view(self.size, self.num_dim)

      distances = torch.norm(self.weights - x_stacked, dim=1, p=None)

      _, bmu_index = torch.min(distances, 0)

      bmu_index = bmu_index.squeeze()

      adaption_rate = 1.0 - it/epocas

      lr = self.learning_rate * adaption_rate

      neighbourhood = self.neighborhood_radius * adaption_rate

      bmu_distance_squares = torch.norm(self.index.float() - torch.stack([self.index[bmu_index]] * self.size).float(), dim=1, p=None)

      neighbourhood_func = torch.exp(torch.neg(torch.div(bmu_distance_squares, neighbourhood**2)))

      neighbourhood_adaption = lr * neighbourhood_func

      lr_multiplier = torch.stack([torch.stack([neighbourhood_adaption[i]] * self.num_dim) for i in range(self.size)])

      lr_multiplier.view(self.size, self.num_dim)

      delta = torch.mul(lr_multiplier, (x_stacked - self.weights))

      new_weights = torch.add(self.weights, delta)

      self.weights = new_weights

    elif mode == 'distribution':

      k = kwargs.get('k', 10)
      y = kwargs.get('y', 0)

      prob = kwargs.get('prob', 'knn')

      _, idx, dist = self.knn(x, k = k)

      if prob == 'knn':

        for c, id in enumerate(idx):
          self.probabilities[id] += dist[c]/torch.sum(dist)

        for c, id in enumerate(idx):
          self.conditional_probabilities[y][id] += dist[c]/torch.sum(dist)

      elif prob == 'int':

        for c, id in enumerate(idx):
          self.probabilities[id] += 1

        for c, id in enumerate(idx):
          self.conditional_probabilities[y][id] += 1

      self.probabilities /= torch.sum(self.probabilities)
      self.conditional_probabilities[y] /= torch.sum(self.conditional_probabilities[y])

      self.class_weights[y] += 1

      self.class_weights[y] /= torch.sum(self.class_weights)


def plot_probability_map(som):
  plt.imshow(som.probabilities.view(som.width,som.height).detach().numpy())
  plt.colorbar()


def plot_conditional_probability_map(som):
  fig, ax = plt.subplots(1, som.num_classes, figsize=(12,9))
  for i in range(som.num_classes):
    ax[i].imshow(som.conditional_probabilities[i].view(som.width,som.height).detach().numpy())
    ax[i].set_title('Class {}'.format(i))


def plot_activation_map(som, x):
  plt.imshow(som.activation_map(x).detach().numpy())
  plt.colorbar()


def training_loop(embedded_ts, som, checkpoint_file, **kwargs):
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
  epochs = kwargs.get('epochs', 2)
  k = kwargs.get('k', 10)
  prob = kwargs.get('prob','knn')
  epx = torch.tensor(epochs)
  epx.to(DEVICE)
  som.double()
  som.to(DEVICE)
  ix = [i for i in range(embedded_ts.num_instances)]

  print('PHASE I - SOM Weight Learning')
  progress_bar = tqdm(range(epochs))
  for it in progress_bar:
    bar_postfix = 'Weight Norm: {}'.format(torch.norm(som.weights).cpu().item())
    progress_bar.set_postfix_str(bar_postfix)

    np.random.shuffle(ix)

    for i in ix:
      X,y = embedded_ts[i]
      X.to(DEVICE)
      itx = torch.tensor(it)
      itx.to(DEVICE)

      som.backward(X, it = itx, epochs = epx)

  print('PHASE II - Probablity Distribution Learning')
  progress_bar = tqdm(range(len(embedded_ts)))
  for it in progress_bar:
    X,y = embedded_ts[it]
    X.to(DEVICE)
    y.to(DEVICE)
    som.backward(X, y = y, mode = 'distribution', k = k, prob=prob)

  checkpoint(som, checkpoint_file)

