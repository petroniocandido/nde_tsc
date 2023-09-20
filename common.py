import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt
from datetime import date
import copy

from aeon.datasets import load_classification

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as torch_transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def checkpoint(model, checkpoint_file):
  torch.save(model.state_dict(), checkpoint_file)

def checkpoint_all(model, optimizer, checkpoint_file):
  torch.save({
    'optim': optimizer.state_dict(),
    'model': model.state_dict(),
}, checkpoint_file)

def resume(model, checkpoint_file):
  model.load_state_dict(torch.load(checkpoint_file, map_location=torch.device(DEVICE)))

def resume_all(model, optimizer, checkpoint_file):
  checkpoint = torch.load(checkpoint_file, map_location=torch.device(DEVICE))
  model.load_state_dict(checkpoint['model'])
  optimizer.load_state_dict(checkpoint['optim'])


def classification_metrics(dataloader, model):
  model.cpu()
  model.eval()
  model.double()

  acc = []
  prec = []
  rec = []
  f1 = []
  for X,y in dataloader:
    X = X.cpu()
    prediction = np.array([model.predict(k) for k in X])
    classes = np.array(y.cpu().tolist())

    acc.append(accuracy_score(classes, prediction))
    prec.append(precision_score(classes, prediction, average='macro'))
    rec.append(recall_score(classes, prediction, average='macro'))
    f1.append(f1_score(classes, prediction, average='macro'))

  return pd.DataFrame([[np.mean(acc), np.mean(prec), np.mean(rec), np.mean(f1)]], columns=['Acc','Prec','Rec','F1'])