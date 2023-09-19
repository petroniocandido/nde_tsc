
import numpy as np
import pandas as pd
from datetime import date
import copy

from aeon.datasets import load_classification

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader


class ClassificationTS(Dataset):
  def __init__(self, name, train, **kwargs):
    super().__init__()

    X, y, _ = load_classification(name)

    self.name = name

    self.num_instances, self.num_attributes, self.num_samples = X.shape

    self.train_split = train

    self.labels = np.unique(y)

    self.transform = kwargs.get('transform', None)

    if kwargs.get('string_labels', False):
      classes = { cx : classe for cx,classe in enumerate(self.labels) }
      classes_inv = { classe : cx for cx,classe in classes.items()}
      y = np.array([classes_inv[k] for k in y])
    else:
      y = np.array([int(float(k)) for k in y])

    self.X = torch.from_numpy(X)
    self.X.double()
    self.y = torch.from_numpy(y)
    self.y = self.y.type(torch.LongTensor)  # Targets sempre do tipo Long

    self.labels = torch.unique(self.y, sorted=True)

  def train(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.num_instances = self.train_split
    tmp.X = self.X[0:self.train_split]
    tmp.y = self.y[0:self.train_split]
    return tmp

  def test(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.num_instances = self.num_instances - self.train_split
    tmp.X = self.X[self.train_split:]
    tmp.y = self.y[self.train_split:]
    return tmp

  def __getitem__(self, indice):
    if not self.transform:
      return self.X[indice].double(), self.y[indice].double()
    else:
      return self.transform(self.X[indice]).double(), self.y[indice].double()

  def __len__(self):
    return self.num_instances

  def __iter__(self):
    for ix in range(self.num_instances):
      yield self[ix]


class TripletClassificationTS(ClassificationTS):
  def __init__(self, name, train, **kwargs):
    super(TripletClassificationTS, self).__init__(name, train, **kwargs)

    self.pos_indexes = []
    self.pos_sizes = []

    self.neg_indexes = []
    self.neg_sizes = []

    self._load_indexes()

  def _load_indexes(self):
    self.pos_indexes = [(self.y == k).nonzero().squeeze() for k in self.labels]
    self.pos_sizes = [len(self.pos_indexes[k]) for k in self.labels]

    self.neg_indexes = [(self.y != k).nonzero().squeeze() for k in self.labels]
    self.neg_sizes = [len(self.neg_indexes[k]) for k in self.labels]

  def train(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.num_instances = self.train_split
    tmp.X = self.X[0:self.train_split]
    tmp.y = self.y[0:self.train_split]
    tmp._load_indexes()
    return tmp

  def test(self) -> Dataset:
    tmp = copy.deepcopy(self)
    tmp.num_instances = self.num_instances - self.train_split
    tmp.X = self.X[self.train_split:]
    tmp.y = self.y[self.train_split:]
    tmp._load_indexes()
    return tmp

  def positive_sample(self, index):
    label = self.y[index].item()
    return self.pos_indexes[label][np.random.randint(self.pos_sizes[label])]

  def negative_sample(self, index):
    label = self.y[index].item()
    return self.neg_indexes[label][np.random.randint(self.neg_sizes[label])]

  def __getitem__(self, index):

    if isinstance(index, int):
      positive = self.positive_sample(index)
      negative = self.negative_sample(index)
    else:
      positive = [self.positive_sample(ix) for ix in index]
      negative = [self.negative_sample(ix) for ix in index]

    if not self.transform:
      return self.X[index].double(), self.y[index].double(), \
        self.X[positive].double(), self.y[positive].double(), \
        self.X[negative].double(), self.y[negative].double(),
    else:
      return self.transform(self.X[index]).double(), self.y[index].double(), \
        self.transform(self.X[positive]).double(), self.y[positive].double(), \
        self.transform(self.X[negative]).double(), self.y[negative].double(),


class EmbeddedTS(Dataset):
  def __init__(self, ds_original : ClassificationTS, encoder):
    self.num_instances = len(ds_original)

    self.X = []
    self.y = []

    encoder.double()

    for xs,ys in ds_original:

      self.X.append( encoder.encode(xs.view(1,ds_original.num_attributes,ds_original.num_samples).double()) )
      self.y.append( ys )

    self.X = torch.stack(self.X)
    self.X.double()
    self.y = torch.stack(self.y)
    self.y = self.y.type(torch.LongTensor)  # Targets sempre do tipo Long

  def __getitem__(self, indice):
    return self.X[indice], self.y[indice]

  def __len__(self):
    return self.num_instances

  def __iter__(self):
    for ix in range(self.num_instances):
      yield self[ix]