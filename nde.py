import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from datetime import date
import copy

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from nde_tsc.som import SOM

class NDE(nn.Module):
  def __init__(self, dataset, num_dim : int, **kwargs):
    super(NDE, self).__init__()
    self.num_attributes : int = dataset.num_attributes
    self.num_samples : int = dataset.num_samples
    self.out_dim : int = num_dim

    self.encoder_fn = kwargs.get('encoder', None)
    self.encoder = self.encoder_fn(self.num_attributes, self.num_samples, self.out_dim)
    self.encoder_training_loop = kwargs.get('encoder_training_loop', None)
    width = kwargs.get('width', 10)
    height = kwargs.get('height', 10)
    self.som : SOM = SOM(width = width, height = height, num_dim = self.out_dim, 
                         num_classes = len(dataset.labels))
    self.som_training_loop = kwargs.get('som_training_loop', None)

  def train(self):
    train_loss, test_loss = self.encoder_training_loop(self.encoder)
    self.som_training_loop(self.som)
    return train_loss, test_loss

  def forward(self, x, k = 3):
    e = self.encoder(x.view(1, self.num_attributes, self.num_samples))
    p = self.som(torch.flatten(e), k=k)
    return p
  
  def probability(self, x, k = 3):
    return self.forward(x, k=k)
  
  def conditional_probabilities(self, x, k = 3):
    e = self.encoder(x.view(1, self.num_attributes, self.num_samples))
    return self.som.conditional_probability(torch.flatten(e), k=k)
  
  def predict(self, x, k = 3):
    p = self.conditional_probabilities(x, k=k)
    return torch.argmax(p).cpu().numpy()

  def __call__(self, x):
    return self.forward(x)