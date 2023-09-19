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
  def __init__(self, **kwargs):
    super(NDE, self).__init__()
    self.dataset_name : str  = kwargs.get('dataset', '')
    self.num_attributes : int = kwargs.get('num_attributes', 10)
    self.num_samples : int = kwargs.get('num_attributes', 10)
    self.num_attributes : int = kwargs.get('num_dim', 10)

    self.encoder = kwargs.get('encoder', None)
    self.encoder_training_loop = kwargs.get('encoder_training_loop', None)
    #self.classifier : Classifier = 
    self.som : SOM = kwargs.get('som', None)
    self.som_training_loop = kwargs.get('som_training_loop', None)

  def train(self):
    train_loss, test_loss = self.encoder_training_loop(self.encoder)
    self.som_training_loop(self.som)
    return train_loss, test_loss

  def forward(self, x, k = 3):
    e = self.encoder(x.view(1,self.num_attributes, self.num_samples))
    p = self.som(torch.flatten(e), k=k)
    return p
  
  def probability(self, x, k = 3):
    return self.forward(x, k=k)
  
  def conditional_probabilities(self, x, k = 3):
    e = self.encoder(x.view(1,self.num_attributes, self.num_samples))
    return self.som.conditional_probability(torch.flatten(e), k=k)

  def __call__(self, x):
    return self.forward(x)