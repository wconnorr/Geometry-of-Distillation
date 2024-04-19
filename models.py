import torch
import torch.nn as nn

# Distiller w/ 3D input (images)
class Distiller3D(nn.Module):
  def __init__(self, c, h, w, n_classes, batch_size, inner_lr=.02, inner_momentum=None, conditional_generation=False):
    super(Distiller3D, self).__init__()
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

# Distiller w/ 1D input (just # features) 
class Distiller1D(nn.Module):
  def __init__(self, batch_size, input_features, num_classes, inner_lr=.02, inner_momentum=None, conditional_generation=False):
    super(Distiller1D, self).__init__()
    self.conditional_generation = conditional_generation

    x = torch.randn((batch_size, input_features))
    self.x = nn.Parameter(x, True)
    if not conditional_generation:
      self.y = nn.Parameter(torch.randn((batch_size, num_classes)), True)

    # Inner optimizer parameters
    if inner_lr is not None:
      self.inner_lr = nn.Parameter(torch.tensor(inner_lr), True)
    if inner_momentum is not None:
      self.inner_momentum = nn.Parameter(torch.tensor(inner_momentum), True)

  def forward(self):
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

class CartpoleActor(nn.Module):
  def __init__(self, state_size=4, action_size=2):
    super(CartpoleActor, self).__init__()

    hidden_size = 64

    # Note: Weight norm does not help Cartpole Distillation!!!
    self.net = nn.Sequential(cartpole_layer_init(nn.Linear(state_size, hidden_size)),
                             nn.Tanh(),
                             cartpole_layer_init(nn.Linear(hidden_size, hidden_size)),
                             nn.Tanh(),
                             cartpole_layer_init(nn.Linear(hidden_size, action_size), std=.01))


  def forward(self, x):
    return self.net(x.view(x.size(0),-1))

# Return a single value for a state, estimating the future discounted reward of following the current policy (it's tied to the PolicyNet it trained with)
class CartpoleCritic(nn.Module):
  def __init__(self, state_size=4):
    super(CartpoleCritic, self).__init__()

    hidden_size = 64

    self.net = nn.Sequential(cartpole_layer_init(nn.Linear(state_size, hidden_size)),
                             nn.Tanh(),
                             cartpole_layer_init(nn.Linear(hidden_size, hidden_size)),
                             nn.Tanh(),
                             cartpole_layer_init(nn.Linear(hidden_size, 1), std=1.))

  def forward(self, x):
    return self.net(x.view(x.size(0),-1))
  
# INITIALIZATION FUNCTIONS #
ROOT_2 = 2**.5

def cartpole_layer_init(layer, std=ROOT_2, bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer

# Dataset that wraps memory for a dataloader
class RLDataset(torch.utils.data.Dataset):
  def __init__(self, rollout):
    super().__init__()
    self.rollout = rollout
    self.width = len(rollout) # Number of distinct value types saved (state, action, etc.)
    rollout_shape = rollout[0].shape
    self.rollout_len, self.num_envs = rollout_shape[0], rollout_shape[1]
    self.full = self.rollout_len * self.num_envs

  def __getitem__(self, index):
    return [self.rollout[i][index//self.num_envs][index%self.num_envs] for i in range(self.width)]

  def __len__(self):
    return self.full