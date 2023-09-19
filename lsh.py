import torch
from torch import nn

class LSH(nn.Module):
  def __init__(self, num_attributes, num_samples, out_dim):
    super().__init__()

    self.name = 'LSH'

    self.out_dim = out_dim

    self.weights = torch.randn(out_dim, num_samples, num_attributes, requires_grad = False) * .15

    self.weights.double()

  def forward(self, x):
    ret = torch.zeros(self.out_dim)
    ret.double()
    for i in range(self.out_dim):
      x = x.double()
      tmp = torch.matmul(x, self.weights[i].double())
      ret[i] = torch.sum(tmp[0])
    return ret

  def encode(self, x):
    return self.forward(x)
  
def setup(lsh, name, train_split, string_labels=True, batch_size = 80, out_dim = 10):
  ds = ClassificationTS(name, train_split, string_labels=string_labels)
  treino_loader = DataLoader(ds.train(), batch_size=batch_size, shuffle=True)
  teste_loader = DataLoader(ds.test(), batch_size=batch_size, shuffle=True)
  ls = lsh(ds.num_attributes, ds.num_samples, out_dim)
  return ls, treino_loader, teste_loader, "nde_ls_" + name + "_{}.pt".format(date.today())

def load(lsh, name, train_split = 10, string_labels=True, out_dim = 10, arquivo = None, date = date.today()):
  ds = ClassificationTS(name, train_split, string_labels=string_labels)
  ls = lsh(ds.num_attributes, ds.num_samples, out_dim)
  if arquivo is None:
    arquivo = "nde_ls_" + name +  "_{}.pt".format(date)
  resume(ls, arquivo)
  return ls