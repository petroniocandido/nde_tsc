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
from torchvision import transforms

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DISPOSITIVO_EXECUCAO = 'cuda' if torch.cuda.is_available() else 'cpu'
DIRETORIO_PADRAO = "."

def checkpoint(modelo, arquivo):
  torch.save(modelo.state_dict(), DIRETORIO_PADRAO + arquivo)

def checkpoint_all(modelo, otimizador, arquivo):
  torch.save({
    'optim': otimizador.state_dict(),
    'model': modelo.state_dict(),
}, DIRETORIO_PADRAO + arquivo)

def resume(modelo, arquivo):
  modelo.load_state_dict(torch.load(DIRETORIO_PADRAO + arquivo, map_location=torch.device(DISPOSITIVO_EXECUCAO)))

def resume_all(modelo, otimizador, arquivo):
  checkpoint = torch.load(DIRETORIO_PADRAO + arquivo, map_location=torch.device(DISPOSITIVO_EXECUCAO))
  modelo.load_state_dict(checkpoint['model'])
  otimizador.load_state_dict(checkpoint['optim'])


def classification_metrics(dataloader, model):
  model.to(DISPOSITIVO_EXECUCAO)
  model.eval()
  model.double()

  acc = []
  prec = []
  rec = []
  f1 = []
  for X,y in dataloader:
    X = X.to(DISPOSITIVO_EXECUCAO)
    prediction = model.predict(X).cpu().numpy()
    classes = np.array(y.cpu().tolist())

    acc.append(accuracy_score(classes, prediction))
    prec.append(precision_score(classes, prediction, average='macro'))
    rec.append(recall_score(classes, prediction, average='macro'))
    f1.append(f1_score(classes, prediction, average='macro'))

  return pd.DataFrame([[np.mean(acc), np.mean(prec), np.mean(rec), np.mean(f1)]], columns=['Acc','Prec','Rec','F1'])