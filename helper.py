"""
Miscellaneous functions
"""

import torch
import torchvision
import numpy as np
from lightning import Fabric

import models
import vector_env

def num_parameters(model):
  """
  Returns number of parameters in the given model
  """
  count = 0
  for param in model.parameters():
    count += np.prod(list(param.shape))
  return count

def load_model(model, filepath, device='cpu'):
  """
  Loads model state dict from .pt file and loads state dict onto model
  returns model (modified in-place)
  """
  sd = torch.load(filepath, map_location=device)
  model.load_state_dict(sd)
  return model

def load_distiller_sd_fabric(filepath):
  """
  Loads distiller model from Lightning Fabric's state file
  """
  fabric = Fabric()
  state = fabric.load(filepath)
  return state['distiller']


def setup_sl(model_sd_file, distiller_sd_file, use_mnist=True, scale=False, device=torch.device('cpu')):
  """
  Loads in models and prepares dataset
  model_sd_file is the location of a saved state dictionary file for a SL model
  distiller_sd_file is the location of a saved state dictionary file for a distillation of the corresponding SL task
  if not use_mnist: CIFAR-10 is loaded instead
  if scale: dataset instances are normalized using a standard normalization technique: make sure the model and distiller were trained w/ scaled dataset!
  """
  c = 1 if use_mnist else 3
  hw = 28 if use_mnist else 32
  classes = 10
  model = models.SimpleConvNet(c, hw, hw, classes).to(device)

  load_model(model, model_sd_file, device)

  if distiller_sd_file is not None:
    distiller_sd = torch.load(distiller_sd_file, map_location=device) if distiller_sd_file[-3:] == '.pt' else load_distiller_sd_fabric(distiller_sd_file) 
    distill_batch_size = distiller_sd['x'].size(0)
    distiller = models.Distiller3D(c, hw, hw, classes, distill_batch_size).to(device)
    distiller.load_state_dict(distiller_sd)
  else:
    distiller = None

  if use_mnist:
    mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=.1307, std=.3081)] )if scale else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=True, transform=mnist_transform, download=True)
  else:
    cifar10_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) if scale else torchvision.transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(r"~/Datasets/CIFAR10", train=True,  transform=cifar10_transform, download=True)
  
  return model, distiller, dataset, (c, hw, hw, classes)

def setup_cartpole(actor_sd_file, critic_sd_file, distiller_sd_file, d_critic_sd_file, num_parallel_envs, device=torch.device('cpu')):
  """
  Sets up models from files and creates vector env for 1D cartpole
  critic_sd_file is the critic corresponding to actor_sd_file: trained w/ cartpole PPO
  d_critic_sd_file is the critic trained alongside distiller_sd_file: trained w/ Meta-PPO
  """
  env = vector_env.make_cartpole_vector_env(num_parallel_envs)
  actor = models.CartpoleActor().to(device)
  actor_sd = torch.load(actor_sd_file, map_location=device)
  actor.load_state_dict(actor_sd)
  critic = models.CartpoleCritic().to(device)
  critic_sd = torch.load(critic_sd_file, map_location=device)
  critic.load_state_dict(critic_sd)
  if distiller_sd_file == '':
    distiller = None
    d_critic = None
  else:
    action_space = env.action_space[0].n
    # Cartpole has a 1D observation space (w/ # vector envs as 1st dim)
    _, state_space = env.observation_space.shape
  
    distiller_sd = torch.load(distiller_sd_file, map_location=device) if distiller_sd_file[-3:] == '.pt' else load_distiller_sd_fabric(distiller_sd_file) 
    distill_batch_size = distiller_sd['x'].size(0)
    distiller = models.Distiller1D(distill_batch_size, state_space, action_space).to(device)
    distiller.load_state_dict(distiller_sd)
    
    d_critic = models.CartpoleCritic().to(device)
    d_critic_sd = torch.load(d_critic_sd_file, map_location=device)
    d_critic.load_state_dict(d_critic_sd)

  return actor, critic, distiller, d_critic, env

def setup_atari(actor_sd_file, critic_sd_file, distiller_sd_file, d_critic_sd_file, env_name, num_parallel_envs, device=torch.device('cpu')):
  """
  Sets up models from files and creates vector env for provided env_name
  For Centipede, we set env_name to 'CentipedeNoFrameskip-v4'
  critic_sd_file is the critic corresponding to actor_sd_file: trained w/ Atari PPO
  d_critic_sd_file is the critic trained alongside distiller_sd_file: trained w/ Meta-PPO
  """
  
  env = vector_env.make_atari_vector_env(num_parallel_envs, env_name)
  action_space = env.action_space[0].n
  # Cartpole has a 1D observation space (w/ # vector envs as 1st dim)
  _, state_c, state_h, state_w = env.observation_space.shape
  
  actor = models.AtariActor(state_c, action_space).to(device)
  actor_sd = torch.load(actor_sd_file, map_location=device)
  actor.load_state_dict(actor_sd)
  critic = models.AtariCritic(state_c).to(device)
  critic_sd = torch.load(critic_sd_file, map_location=device)
  critic.load_state_dict(critic_sd)
  if distiller_sd_file == '':
    distiller = None
    d_critic = None
  else:
  
    distiller_sd = torch.load(distiller_sd_file, map_location=device) if distiller_sd_file[-3:] == '.pt' else load_distiller_sd_fabric(distiller_sd_file) 
    distill_batch_size = distiller_sd['x'].size(0)
    distiller = models.Distiller3D(state_c, state_h, state_w, action_space, distill_batch_size).to(device)
    distiller.load_state_dict(distiller_sd)
    
    d_critic = models.AtariCritic(state_c).to(device)
    d_critic_sd = torch.load(d_critic_sd_file, map_location=device)
    d_critic.load_state_dict(d_critic_sd)

  return actor, critic, distiller, d_critic, env