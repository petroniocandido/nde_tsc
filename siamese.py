import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm
from nde_tsc.common import checkpoint, checkpoint_all
from nde_tsc.autoencoder import EncoderConvFC1


class Siamese(nn.Module):
  def __init__(self, name, num_attributes, num_samples, out_dim, encoder):
    super(Siamese, self).__init__()

    self.name = name

    self.out_dim = out_dim

    self.encoder = encoder(num_attributes, num_samples, out_dim)

  def forward(self, x):
    x = self.encoder(x)
    return x

  def encode(self, x):
    return self.encoder(x)


class SiameseConvFC1(Siamese):
  def __init__(self, num_attributes, num_samples, out_dim):
    super(SiameseConvFC1, self).__init__('SiameseConvFC1', num_attributes, num_samples, out_dim, EncoderConvFC1)


def training_loop(train_ldr, test_ldr, model, **kwargs):

  DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

  model.double()

  model.to(DEVICE)

  learning_rate = kwargs.get('learning_rate', 0.0001)
  optimizer = kwargs.get('opt', optim.NAdam(model.parameters(), lr=learning_rate))
  loss = nn.TripletMarginLoss()
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

    modelo.train()              # Habilita o treinamento do modelo

    losses = []
    for Xa, ya, Xp, yp, Xn, yn in train_ldr:

      Xa = Xa.to(DEVICE)
      Xp = Xp.to(DEVICE)
      Xn = Xn.to(DEVICE)

      a_pred = model.forward(Xa)
      p_pred = model.forward(Xp)
      n_pred = model.forward(Xn)

      optimizer.zero_grad()

      _loss = loss(a_pred, p_pred, n_pred)

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
      for Xat, yat, Xpt, ypt, Xnt, ynt in test_ldr:

        Xat = Xat.to(DEVICE)
        Xpt = Xpt.to(DEVICE)
        Xnt = Xnt.to(DEVICE)

        a_predt = model.forward(Xat)
        p_predt = model.forward(Xpt)
        n_predt = model.forward(Xnt)

        _loss_val = loss(a_predt, p_predt, n_predt)

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
