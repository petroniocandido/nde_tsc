import numpy as np
from datetime import date
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from nde_tsc.common import checkpoint, checkpoint_all, resume
from nde_tsc.data import ClassificationTS, data_augmentation

class AutoEncoderBase(nn.Module):
  def __init__(self, name, num_attributes, num_samples, out_dim, encoder, decoder):
    super(AutoEncoderBase, self).__init__()

    self.name = name

    self.out_dim = out_dim

    self.encoder = encoder(num_attributes, num_samples, out_dim)

    self.decoder = decoder(num_attributes, num_samples, out_dim)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x

  def encode(self, x):
    return self.encoder(x)

  def decode(self, x):
    return self.decoder(x)


class EncoderConvFC1(nn.Module):
  def __init__(self, num_attributes, num_samples, out_dim):
    super().__init__()

    primary = num_samples // 2
    secondary = num_samples // 3

    size = num_attributes * num_samples

    self.bn = nn.LazyBatchNorm1d(momentum=0.5)
    self.pool = nn.AdaptiveMaxPool1d(3)

    self.conv1 = nn.Conv1d(num_attributes, primary, 7, padding=3, padding_mode='reflect')
    self.conv2 = nn.Conv1d(primary, primary, 3, padding=1, padding_mode='reflect')

    self.flat = nn.Flatten(1)

    self.tanh = nn.Tanh()
    self.drop = nn.Dropout1d(.25)

    self.linear1 = nn.LazyLinear(primary)
    self.linear2 = nn.LazyLinear(secondary)
    self.linear3 = nn.LazyLinear(out_dim)

  def forward(self, x):

    x = self.bn(self.conv1(x))
    x = self.bn(self.conv2(x))

    x = self.pool(x)

    x = self.flat(x)

    x = self.tanh(self.drop(self.linear1(x)))
    x = self.tanh(self.drop(self.linear2(x)))
    x = self.linear3(x)

    return x


class EncoderFC1(nn.Module):
  def __init__(self, num_attributes, num_samples, out_dim):
    super().__init__()

    primary = num_samples // 2
    secondary = num_samples // 3

    size = num_attributes * num_samples

    self.flat = nn.Flatten(1)
    self.tanh = nn.Tanh()
    self.drop = nn.Dropout1d(.3)

    self.linear1 = nn.Linear(size, primary)
    self.linear2 = nn.Linear(primary, secondary)
    self.linear3 = nn.Linear(secondary, out_dim)

  def forward(self, x):
    x = self.flat(x)

    x = self.tanh(self.drop(self.linear1(x)))
    x = self.tanh(self.drop(self.linear2(x)))
    x = self.linear3(x)

    return x


class DecoderConvFC1(nn.Module):
  def __init__(self, num_attributes, num_samples, out_dim):
    super().__init__()

    self.num_attributes = num_attributes
    self.num_samples = num_samples
    size = num_attributes * num_samples

    primary = num_samples // 2
    secondary = num_samples // 3

    self.drop = nn.Dropout1d(.25)
    self.tanh = nn.Tanh()
    self.bn = nn.LazyBatchNorm1d(momentum=0.5)

    self.linear1 = nn.LazyLinear(secondary)
    self.linear2 = nn.LazyLinear(primary)
    self.linear3 = nn.LazyLinear(size)

    self.tconv1 = nn.ConvTranspose1d(num_attributes, num_attributes, 3, padding=1)
    self.tconv2 = nn.ConvTranspose1d(num_attributes, num_attributes, 7, padding=3)


  def forward(self, x):

    batch_size, out_dim = x.size()

    x = self.tanh(self.drop(self.linear1(x)))
    x = self.tanh(self.drop(self.linear2(x)))
    x = self.linear3(x)

    x = x.view(batch_size, self.num_attributes, self.num_samples)

    x = self.tconv1(x)
    x = self.tconv2(x)

    return x


class DecoderFC1(nn.Module):
  def __init__(self, num_attributes, num_samples, out_dim):
    super().__init__()

    self.num_attributes = num_attributes
    self.num_samples = num_samples
    size = num_attributes * num_samples

    primary = num_samples // 2
    secondary = num_samples // 3

    self.drop = nn.Dropout1d(.3)
    self.tanh = nn.Tanh()

    self.linear1 = nn.Linear(out_dim, secondary)
    self.linear2 = nn.Linear(secondary, primary)
    self.linear3 = nn.Linear(primary, size)


  def forward(self, x):

    batch_size, out_dim = x.size()

    x = self.tanh(self.drop(self.linear1(x)))
    x = self.tanh(self.drop(self.linear2(x)))
    x = self.linear3(x)

    return x.view(batch_size, self.num_attributes, self.num_samples)


class AutoEncoderConvFC1(AutoEncoderBase):
  def __init__(self, num_attributes, num_samples, out_dim):
    super(AutoEncoderConvFC1, self).__init__('ConvFC1', num_attributes, num_samples, out_dim,
                                      EncoderConvFC1, DecoderConvFC1)
    


def training_loop(train_ldr, test_ldr, model, **kwargs):
  
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

  model.double()

  model.to(DEVICE)

  learning_rate = kwargs.get('learning_rate', 0.0001)
  optimizer = kwargs.get('opt', optim.NAdam(model.parameters(), lr=learning_rate))
  loss = kwargs.get('loss', nn.MSELoss())
  epochs = kwargs.get('epochs', 10)
  
  early_stop = kwargs.get('early_stop', True)

  file_checkpoint = kwargs.get('file_checkpoint', 'model.pt')

  loss_train = [0]
  loss_test = [0]

  if early_stop:
    best_loss = 1000
    patience = kwargs.get('patience',50)
    patience_count = 0

  progress_bar = tqdm(range(epochs))

  for epoch in progress_bar:

    bar_postfix = 'Loss Train: {} Test: {}'.format(round(loss_train[-1], 2), round(loss_test[-1], 2),2)

    progress_bar.set_postfix_str(bar_postfix)

    if not early_stop and epoch % 25 == 0:
      checkpoint_all(model, optimizer, file_checkpoint)


    ##################
    # TRAIN
    ##################

    model.train()              # Habilita o treinamento do modelo

    losses = []
    for X, y in train_ldr:

      X = X.to(DEVICE)

      y_pred = model.forward(X)

      optimizer.zero_grad()

      _loss = loss(y_pred, X)

      _loss.backward()
      optimizer.step()

      # Grava as métricas de avaliação
      losses.append(_loss.cpu().item())

    # Salva o valor médio das métricas de avaliação para o lote
    loss_train.append(np.mean(losses))


    ##################
    # VALIDAÇÃO
    ##################

    model.eval()               # Desabilita a atualização dos parâmetros

    losses = []
    with torch.no_grad():
      _current_loss = 0
      for X, y in test_ldr:
        X = X.to(DEVICE)

        y_pred_val = model.forward(X)

        _loss_val = loss(y_pred_val, X)

        # Grava as métricas de avaliação
        losses.append(_loss_val.cpu().item())

        # Grava o _loss do erly stopping
        _current_loss += losses[-1]

      # Salva o valor médio das métricas de avaliação para o lote
      loss_test.append(np.mean(losses))

      ##################
      # EARLY STOPPING
      ##################

      if early_stop:

        if _current_loss <= best_loss:
          best_loss = _current_loss
          patience_count = 0

          # Checkpoint do melhor modelo
          checkpoint(model, file_checkpoint)

        else:
          patience_count += 1

        if patience_count > patience:
          return loss_train, loss_test

  checkpoint(model, file_checkpoint)

  return loss_train, loss_test

def setup(autoencoder, name, train_split, string_labels=True, batch_size = 80, out_dim = 10):
  ds = ClassificationTS(name, train_split, string_labels=string_labels, transforms=data_augmentation)
  treino_loader = DataLoader(ds.train(), batch_size=batch_size, shuffle=True)
  teste_loader = DataLoader(ds.test(), batch_size=batch_size, shuffle=True)
  ac = autoencoder(ds.num_attributes, ds.num_samples, out_dim)
  return ac, treino_loader, teste_loader, "nde_ac_" + name + "_" + ac.name + "_{}.pt".format(date.today())

def load(autoencoder, name, train_split = 10, string_labels=True, out_dim = 10, arquivo = None, date = date.today()):
  ds = ClassificationTS(name, train_split, string_labels=string_labels, transforms=data_augmentation)
  ac = autoencoder(ds.num_attributes, ds.num_samples, out_dim)
  if arquivo is None:
    arquivo = "nde_ac_" + name + "_" + ac.name + "_{}.pt".format(date)
  resume(ac, arquivo)
  return ac