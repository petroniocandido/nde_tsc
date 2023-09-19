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