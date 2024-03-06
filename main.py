import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import copy

import models
import vector_env
from geometry import model_manifold_curvature, select_normalized_direction, normed_visualization, calc_loss, calc_loss_distill, rl_landscape, calc_FIM_distill
from helper import load_model, load_distiller_sd_fabric, num_parameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_sl(model_sd_file, distiller_sd_file, scale_mnist=False):
  model = models.SimpleConvNet(1, 28, 28, 10).to(device)
  print("Small conv network has {} parameters!".format(num_parameters(model)))

  load_model(model, model_sd_file, device)

  distiller_sd = torch.load(distiller_sd_file, map_location=device) if distiller_sd_file[-3:] == '.pt' else load_distiller_sd_fabric(distiller_sd_file) 
  distill_batch_size = distiller_sd['x'].size(0)
  print("Using {} distilled instances".format(distill_batch_size))
  distiller = models.Distiller(1, 28, 28, 10, distill_batch_size).to(device)
  distiller.load_state_dict(distiller_sd)

  mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=.1307, std=.3081)] )if scale_mnist else torchvision.transforms.ToTensor()

  mnist = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=True, transform=mnist_transform, download=True)

  return model, distiller, mnist

def setup_rl():
  raise NotImplementedError()

def graph_rl_landscape(actor, critic, static_dataset, static_critic, num_rl_episodes):

  env = vector_env.make_cartpole_vector_env(10)

  theta_a =  copy.deepcopy(list(actor.parameters()))
  theta_c =  copy.deepcopy(list(critic.parameters()))

  
  delta = select_normalized_direction(theta_a) if static_critic else [select_normalized_direction(theta_a), select_normalized_direction(theta_c)]
  eta   = select_normalized_direction(theta_a) if static_critic else [select_normalized_direction(theta_a), select_normalized_direction(theta_c)]

  model_sd = copy.deepcopy(model.state_dict())

  exp_name = "rl_landscape"
  rl_landscape(actor, theta_a, critic, theta_c, env, delta, eta, 1, 10, exp_name + '_cartpole.png', num_rl_episodes, static_dataset, static_critic)



def graph_landscapes_2x2(model, theta, distiller, mnist):
  
  # # Fix delta and eta for direct comparison
  delta = select_normalized_direction(theta)
  eta   = select_normalized_direction(theta)
  
  model_sd = copy.deepcopy(model.state_dict())

  exp_name = "longlowval"
  
  exp_name_sl = exp_name + '_sltrained'
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss, mnist, exp_name_sl + '_mnist.png')
  # reload model before revisualizing because we change model parameters in visualization
  model.load_state_dict(model_sd)
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss_distill, distiller, exp_name_sl + '_distiller.png')
  
  exp_name_distill = exp_name + '_dltrained'
  
  # DISTILLATION TRAINING
  with torch.no_grad():
    x, y = distiller()
  # randomly initialize model
  model = models.SimpleConvNet(1, 28, 28, 10).to(device)
  inner_optimizer = torch.optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
  inner_loss = F.mse_loss(model(x), y)
  inner_loss.backward()
  inner_optimizer.step()
  inner_optimizer.zero_grad()
  model_sd = copy.deepcopy(model.state_dict())
  theta = copy.deepcopy(list(model.parameters()))
  
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss, mnist, exp_name_distill + '_mnist.png')
  # reload model before revisualizing because we change model parameters in visualization
  model.load_state_dict(model_sd)
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss_distill, distiller, exp_name_distill + '_distiller.png')
    
    
  # do we care about distance of minima in parameter space? Or just the amount it misses by?
  # DISTILLATION TRAINING
  with torch.no_grad():
    x, y = distiller()
  # randomly initialize model
  model_d = models.SimpleConvNet(1, 28, 28, 10).to(device)
  inner_optimizer = torch.optim.SGD(model_d.parameters(), lr=distiller.inner_lr.item())
  inner_loss = F.mse_loss(model_d(x), y)
  inner_loss.backward()
  inner_optimizer.step()
  inner_optimizer.zero_grad()
  
  # Determine how far the parameters are in Theta
  theta = torch.cat([p.flatten() for p in model.parameters()], 0)
  theta_d = torch.cat([p.flatten() for p in model_d.parameters()], 0)
  print("The max distance between any thetas is {}".format(torch.max(torch.abs(theta-theta_d))))
  m = torch.argmax(torch.abs(theta-theta_d))
  print(theta[m])
  print(theta_d[m])
  
def graph_landscapes_combined(model, theta, distiller, mnist):
  
  model_sd = copy.deepcopy(model.state_dict())
  
  exp_name = 'longlowval_both'
  # set delta to line between two minima : DO NOT normalize: we want a=1 to be the line between the two
  model_d = models.SimpleConvNet(1, 28, 28, 10).to(device)
  inner_optimizer = torch.optim.SGD(model_d.parameters(), lr=distiller.inner_lr.item())
  with torch.no_grad():
    x, y = distiller()
  inner_loss = F.mse_loss(model_d(x), y)
  inner_loss.backward()
  inner_optimizer.step()
  inner_optimizer.zero_grad()
  model_sd_d = copy.deepcopy(model_d.state_dict())
  theta_d = copy.deepcopy(list(model_d.parameters()))
  
  # delta=theta_d-theta # theta_d = theta + 1*delta
  # make eta's scale the same as delta
  delta = []
  for l, ld in zip(theta, theta_d):
    delta.append(ld-l)
  eta = select_normalized_direction(theta)
  
  normed_visualization(model, theta, delta, eta, 1, 21, calc_loss, mnist, exp_name + '_mnist.png', two_points=True)
  # reload model before revisualizing because we change model parameters in visualization
  model.load_state_dict(model_sd)
  normed_visualization(model, theta, delta, eta, 1, 11, calc_loss_distill, distiller, exp_name + '_distiller.png', two_points=True)

def calc_fim():
  # FIM: you should be able to calculate this on the supercomputer
  
  distiller_sd = torch.load('./sl_exp/experiments/distill_long_highval/checkpoint_final1/distiller_sd.pt', map_location=device)
  distill_batch_size = distiller_sd['x'].size(0)
  print("Using {} distilled instances".format(distill_batch_size))
  distiller = models.Distiller(1, 28, 28, 10, distill_batch_size).to(device)
  distiller.load_state_dict(distiller_sd)
  with torch.no_grad():
    x, y = distiller()
  # randomly initialize model
  model = models.SimpleConvNet(1, 28, 28, 10).to(device)
  print("Small conv network has {} parameters!".format(num_parameters(model)))
  for param in model.parameters():
    print("\t",param.shape)
  inner_optimizer = torch.optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
  inner_loss = F.mse_loss(model(x), y)
  inner_loss.backward()
  inner_optimizer.step()
  inner_optimizer.zero_grad()
  
  fim = calc_FIM_distill(model, distiller)

def model_manifold(model, distiller, mnist):

  delta = select_normalized_direction(theta)
  model_manifold_curvature(model, distiller, mnist, delta, './curvature_sltrained_')

###
model_sd_file = './sl_exp/experiments/mnist_sl_saveinit/checkpoint_final1/learner_sd.pt'

# distiller_sd_file = './sl_exp/lightning_distill_6gpu/checkpoint6/state.ckpt'
distiller_sd_file = './sl_exp/experiments/distill_long_highval/checkpoint1/distiller_sd.pt'
# distiller_sd_file = './sl_exp/experiments/distill_long_lowval/checkpoint1/state.ckpt'

model, distiller, mnist = setup_sl(model_sd_file, distiller_sd_file)

# "True" parameters (found from direct-task learning)
theta = copy.deepcopy(list(model.parameters()))

graph_landscapes_2x2(model, theta, distiller, mnist)
graph_landscapes_combined(model, theta, distiller, mnist)
###
actor_sd_file = ''
critic_sd_file = ''
distiller_sd_file = ''

actor, critic, distiller = setup_rl(actor_sd_file, critic_sd_file, distiller_sd_file)

rl_landscape()
###

# model_manifold()
# calc_fim()