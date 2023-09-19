import torch

class Noise(object):
  def __init__(self, type='unif', **kwargs):
    self.type = type
    if self.type == 'unif':
      self.min = kwargs.get('min', 0)
      self.max = kwargs.get('max', 1)
      self.range = self.max - self.min
      print(self.range)
    elif self.type == 'normal':
      self.std = kwargs.get('std', 1)
      self.mean = kwargs.get('mean', 0)

  def __call__(self, tensor):
    if self.type == 'unif':
      return tensor + ((torch.rand(tensor.size()) * self.range) + self.min)
    elif self.type == 'normal':
      return tensor + torch.randn(tensor.size()) * self.std + self.mean

  def __repr__(self):
    if self.type == 'normal':
      return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    elif self.type == 'unif':
      return self.__class__.__name__ + '(min={0}, max={1})'.format(self.min, self.max)


class Linear(object):
  def __init__(self, k):
    self.k = k

  def __call__(self, tensor):
    return tensor * self.k

  def __repr__(self):
    return self.__class__.__name__ + '(k={})'.format(self.k)