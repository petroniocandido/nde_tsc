import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from nde_tsc.common import checkpoint, checkpoint_all


class Classifier(nn.Module):
  def __init__(self, in_size, out_size):
    super().__init__()

    self.double()

    self.relu = nn.LeakyReLU(0.1)
    self.drop = nn.Dropout1d(.25)
    self.flat = nn.Flatten()

    k = in_size * out_size

    self.linear1 = nn.Linear(in_size, k)
    self.linear2 = nn.Linear(k, k)
    self.linear3 = nn.Linear(k, out_size)

    self.sm = nn.LogSoftmax(dim=1)

  def forward(self, x):
    x = self.flat(x)

    x = self.drop(self.relu(self.linear1(x)))
    x = self.drop(self.relu(self.linear2(x)))
    x = self.relu(self.linear3(x))
    x = self.sm(x)

    return x

  def predict(self, x):
    x = self.forward(x)
    return torch.argmax(x, dim=1)


def training_loop(train_ldr, test_ldr, model, **kwargs):
  
  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

  model.double()

  model.to(DEVICE)

  learning_rate = kwargs.get('learning_rate', 0.0001)
  optimizer = kwargs.get('opt', optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=0.0005))
  loss = kwargs.get('loss', F.nll_loss)
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
      y = y.to(DEVICE)

      optimizer.zero_grad()

      _loss = loss(y_pred, y)
      _loss.double()
      _loss.backward(retain_graph=True)
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
        y = y.to(DEVICE)

        y_pred_val = model.forward(X)

        _loss_val = loss(y_pred_val, y)
        
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