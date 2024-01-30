import torch
import torch.nn as nn

# Distillation-style wrapper to learn the teaching data directly.
class Distiller(nn.Module):
  def __init__(self, c, h, w, n_classes, batch_size, inner_lr=.02, inner_momentum=None, conditional_generation=False):
    super(Distiller, self).__init__()
    self.conditional_generation = conditional_generation

    self.x = nn.Parameter(torch.randn((batch_size, c, h, w)), True)
    if not conditional_generation:
      self.y = nn.Parameter(torch.randn((batch_size, n_classes)), True)

    # Inner optimizer parameters
    if inner_lr is not None:
      self.inner_lr = nn.Parameter(torch.tensor(inner_lr), True)
    if inner_momentum is not None:
      self.inner_momentum = nn.Parameter(torch.tensor(inner_momentum), True)

  def forward(self, dummy=None): # dummy is needed for lightning, maybe?
    if self.conditional_generation:
      return self.x
    else:
      return self.x, self.y

class SimpleConvNet(nn.Module):
  def __init__(self, c, h, w, n_classes):
    super(SimpleConvNet, self).__init__()
    self.convs = nn.Sequential(
      nn.Conv2d(c, 16, 3, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 32, 3, padding=1),
      nn.ReLU()
    )
    self.lin = nn.Linear(32*h*w, n_classes)

  def forward(self, x):
    x = self.convs(x)
    return self.lin(x.flatten(1))
