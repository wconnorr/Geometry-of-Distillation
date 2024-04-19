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

def setup_sl(model_sd_file, distiller_sd_file, use_mnist=True, scale=False):
  c = 1 if use_mnist else 3
  hw = 28 if use_mnist else 32
  classes = 10
  model = models.SimpleConvNet(c, hw, hw, classes).to(device)
  print("Small conv network has {} parameters!".format(num_parameters(model)))

  load_model(model, model_sd_file, device)

  distiller_sd = torch.load(distiller_sd_file, map_location=device) if distiller_sd_file[-3:] == '.pt' else load_distiller_sd_fabric(distiller_sd_file) 
  distill_batch_size = distiller_sd['x'].size(0)
  print("Using {} distilled instances".format(distill_batch_size))
  distiller = models.Distiller3D(c, hw, hw, classes, distill_batch_size).to(device)
  distiller.load_state_dict(distiller_sd)

  if use_mnist:
    mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=.1307, std=.3081)] )if scale else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=True, transform=mnist_transform, download=True)
  else:
    cifar10_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) if scale else torchvision.transforms.ToTensor()

    dataset = torchvision.datasets.CIFAR10(r"~/Datasets/CIFAR10", train=True,  transform=cifar10_transform, download=True )
    pass
  
  return model, distiller, dataset, (c, hw, hw, classes)

def setup_cartpole(actor_sd_file, critic_sd_file, distiller_sd_file, d_critic_sd_file, num_parallel_envs):
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
    print("Using {} distilled instances".format(distill_batch_size))
    distiller = models.Distiller1D(distill_batch_size, state_space, action_space).to(device)
    distiller.load_state_dict(distiller_sd)
    
    d_critic = models.CartpoleCritic().to(device)
    d_critic_sd = torch.load(d_critic_sd_file, map_location=device)
    d_critic.load_state_dict(d_critic_sd)


  return actor, critic, distiller, d_critic, env

def graph_rl_landscape_repeated(actor, critic, env, static_dataset, static_critic, num_rl_episodes, num_envs, batch_size, gamma, gae_lambda):

  theta_a =  copy.deepcopy(list(actor.parameters()))
  theta_c =  copy.deepcopy(list(critic.parameters()))

  actor_sd = copy.deepcopy(actor.state_dict())
  
  delta = select_normalized_direction(theta_a) if static_critic else [select_normalized_direction(theta_a), select_normalized_direction(theta_c)]
  eta   = select_normalized_direction(theta_a) if static_critic else [select_normalized_direction(theta_a), select_normalized_direction(theta_c)]

  exp_name = "rl_landscape_rep"

  for i in range(20):
    rl_landscape(actor, theta_a, critic, theta_c, env, delta, eta, 1, 10, exp_name + '_cartpole{}.png'.format(i), num_rl_episodes, num_envs, batch_size, static_dataset, static_critic, gamma, gae_lambda)
    actor.load_state_dict(actor_sd)

def graph_rl_landscape_2x2(actor, critic, distiller, d_critic, env, static_dataset, static_critic, num_rl_episodes, num_envs, batch_size, gamma, gae_lambda):

  actor_sd = copy.deepcopy(actor.state_dict())
  
  theta_a =  copy.deepcopy(list(actor.parameters()))
  theta_c =  copy.deepcopy(list(critic.parameters()))

  
  delta = select_normalized_direction(theta_a) if static_critic else [select_normalized_direction(theta_a), select_normalized_direction(theta_c)]
  eta   = select_normalized_direction(theta_a) if static_critic else [select_normalized_direction(theta_a), select_normalized_direction(theta_c)]

  # TODO: RL rollout
  # TODO: Distilled rollout

  exp_name = "rl_landscape_2x2_FIXED"
  rl_landscape(actor, theta_a, critic, theta_c, env, delta, eta, 1, 10, exp_name + '_rltrained_cartpole.png', num_rl_episodes, num_envs, batch_size, static_dataset, static_critic, gamma, gae_lambda)
  actor.load_state_dict(actor_sd)
  # Distiller landscape is supervised learning!!!
  normed_visualization(actor, theta_a, delta, eta, 1, 10, calc_loss_distill, distiller, exp_name + '_rltrained_distiller.png')

  # TODO: Train distiller
  with torch.no_grad():
    x, y = distiller()
  # randomly initialize model
  d_actor = models.CartpoleActor().to(device)
  inner_optimizer = torch.optim.SGD(d_actor.parameters(), lr=distiller.inner_lr.item())
  inner_loss = F.mse_loss(d_actor(x), y)
  inner_loss.backward()
  inner_optimizer.step()
  inner_optimizer.zero_grad()
  d_actor_sd = copy.deepcopy(d_actor.state_dict())
  d_theta_a = copy.deepcopy(list(d_actor.parameters()))
  
  # TODO: Distiller-centered datasets
  rl_landscape(d_actor, d_theta_a, d_critic, theta_c, env, delta, eta, 1, 10, exp_name + '_distilltrained_cartpole.png', num_rl_episodes, num_envs, batch_size, static_dataset, static_critic, gamma, gae_lambda)
  d_actor.load_state_dict(d_actor_sd)
  # Distiller landscape is supervised learning!!!
  normed_visualization(d_actor, d_theta_a, delta, eta, 1, 10, calc_loss_distill, distiller, exp_name + '_distilltrained_distiller.png')
  d_actor.load_state_dict(d_actor_sd)
  
  # ZOOMED OUT SYNTHETIC-SYNTHETIC
  normed_visualization(d_actor, d_theta_a, delta, eta, 10, 100, calc_loss_distill, distiller, exp_name + '_distilltrained_distiller_zoomout.png')

def graph_synth_synth_init_space(distiller):
  """
  Goal - model initialization space of model : theta hat = 0, std dev depends on layer (different in each direction), delta to point to trained theta

  - thetahat = 0
  - theta0 = init
  - theta1 = trained theta
  - eta = random
  - delta = theta1, but scaled using per-layer std dev! We don't need to see theta1 (stretched out quite a bit), we just need to see immediately around theta0
  - plot and show that slope moves right for some of the points: others go to other minima: we can even force this by locking seed and picking points that lead to different minima
  - see if we can project theta0 onto the map

    do some that are whole space (thetahat: delta = theta1) and others that are only init to theta1 (delta = theta1 - theta0)
  """

  with torch.no_grad():
    x, y = distiller()
  # randomly initialize model
  d_actor = models.CartpoleActor().to(device)
  theta_0 = copy.deepcopy(list(d_actor.parameters()))
  sd_0 = copy.deepcopy(d_actor.state_dict())
  inner_optimizer = torch.optim.SGD(d_actor.parameters(), lr=distiller.inner_lr.item())
  inner_loss = F.mse_loss(d_actor(x), y)
  inner_loss.backward()
  inner_optimizer.step()
  inner_optimizer.zero_grad()
  d_actor_sd = copy.deepcopy(d_actor.state_dict())
  theta_trained = copy.deepcopy(list(d_actor.parameters()))
  
  theta_hat = [torch.zeros_like(t0).to(device) for t0 in theta_0]

  sd = d_actor.state_dict()
  for layer, (name, _) in zip(theta_hat, d_actor.named_parameters()):
    sd[name] = layer
  d_actor.load_state_dict(sd)

  delta = []
  for layer, (name, _) in zip(theta_trained, d_actor.named_parameters()):
    d_layer = []
    std_dev = (1. if name == 'net.4.weight' else 2**.5) if 'weight' in name else 0
    for filter in layer:
      # Normalize by frobenius norm of theta's corresponding filter
      d_filter = filter * (std_dev / torch.norm(filter,p='fro'))
      d_layer.append(d_filter)
    delta.append(torch.stack(d_layer,0))
    
  eta = select_normalized_direction(theta_trained)

  normed_visualization(d_actor, theta_hat, delta, eta, 2, 100, calc_loss_distill, distiller, "zero_centered_cartpole_dd.png")

  # TODO: zoom closely into initialization to show gradient

  # TODO: show line between init and final result
  # Make delta line between 0 and trained
  delta = [tt - t0 for tt, t0 in zip(theta_trained,theta_0)]
  d_actor.load_state_dict(sd_0)
  normed_visualization(d_actor, theta_0, delta, eta, 1, 100, calc_loss_distill, distiller, "trajectory_cartpole_dd.png", two_points=True, label="Initialization")


def graph_landscapes_2x2(model, theta, distiller, mnist, model_size):
  
  # # Fix delta and eta for direct comparison
  delta = select_normalized_direction(theta)
  eta   = select_normalized_direction(theta)
  
  model_sd = copy.deepcopy(model.state_dict())

  exp_name = "scaled_cifar10"
  
  exp_name_sl = exp_name + '_sltrained'
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss, mnist, exp_name_sl + '_dataset.png')
  # reload model before revisualizing because we change model parameters in visualization
  model.load_state_dict(model_sd)
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss_distill, distiller, exp_name_sl + '_distiller.png')
  
  exp_name_distill = exp_name + '_dltrained'
  
  # DISTILLATION TRAINING
  with torch.no_grad():
    x, y = distiller()
  # randomly initialize model
  model = models.SimpleConvNet(*model_size).to(device)
  inner_optimizer = torch.optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
  inner_loss = F.mse_loss(model(x), y)
  inner_loss.backward()
  inner_optimizer.step()
  inner_optimizer.zero_grad()
  model_sd = copy.deepcopy(model.state_dict())
  theta = copy.deepcopy(list(model.parameters()))
  
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss, mnist, exp_name_distill + '_dataset.png')
  # reload model before revisualizing because we change model parameters in visualization
  model.load_state_dict(model_sd)
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss_distill, distiller, exp_name_distill + '_distiller.png')
    
    
  # do we care about distance of minima in parameter space? Or just the amount it misses by?
  # DISTILLATION TRAINING
  with torch.no_grad():
    x, y = distiller()
  # randomly initialize model
  model_d = models.SimpleConvNet(*model_size).to(device)
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
  
def graph_landscapes_combined(model, theta, distiller, mnist, model_size):
  
  model_sd = copy.deepcopy(model.state_dict())
  
  exp_name = 'scaled_cifar10_both'
  # set delta to line between two minima : DO NOT normalize: we want a=1 to be the line between the two
  model_d = models.SimpleConvNet(*model_size).to(device)
  inner_optimizer = torch.optim.SGD(model_d.parameters(), lr=distiller.inner_lr.item())
  with torch.no_grad():
    x, y = distiller()
  inner_loss = F.mse_loss(model_d(x), y)
  inner_loss.backward()
  inner_optimizer.step()
  inner_optimizer.zero_grad()
  theta_d = copy.deepcopy(list(model_d.parameters()))
  
  # delta=theta_d-theta # theta_d = theta + 1*delta
  # make eta's scale the same as delta
  delta = []
  for l, ld in zip(theta, theta_d):
    delta.append(ld-l)
  eta = select_normalized_direction(theta)
  
  normed_visualization(model, theta, delta, eta, 1, 21, calc_loss, mnist, exp_name + '_dataset.png', two_points=True)
  # reload model before revisualizing because we change model parameters in visualization
  model.load_state_dict(model_sd)
  normed_visualization(model, theta, delta, eta, 1, 11, calc_loss_distill, distiller, exp_name + '_distiller.png', two_points=True)

def calc_fim():
  # FIM: you should be able to calculate this on the supercomputer
  
  distiller_sd = torch.load('./sl_exp/experiments/distill_long_highval/checkpoint_final1/distiller_sd.pt', map_location=device)
  distill_batch_size = distiller_sd['x'].size(0)
  print("Using {} distilled instances".format(distill_batch_size))
  distiller = models.Distiller3D(1, 28, 28, 10, distill_batch_size).to(device)
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
# # model_sd_file = './sl_exp/experiments/mnist_sl_saveinit/checkpoint_final1/learner_sd.pt'
# # model_sd_file = './sl_exp/mnist_normalized_sl_fixed/checkpoint_final1/learner_sd.pt'
# model_sd_file = './sl_exp/cifar_sl_2gpu_fixed/checkpoint_final1/learner_sd.pt'

# # distiller_sd_file = './sl_exp/lightning_distill_6gpu/checkpoint6/state.ckpt'
# # distiller_sd_file = './sl_exp/experiments/distill_long_highval/checkpoint1/distiller_sd.pt'
# # distiller_sd_file = './sl_exp/experiments/distill_long_lowval/checkpoint1/state.ckpt'
# # distiller_sd_file = './sl_exp/mnist_normalized_distill_4gpu_fixed/checkpoint2/state.ckpt'
# distiller_sd_file = './sl_exp/lightning_distill_cifar10_4gpu_fixed/checkpoint2/state.ckpt'

# model, distiller, dataset, model_size = setup_sl(model_sd_file, distiller_sd_file, use_mnist=False, scale=True)

# # "True" parameters (found from direct-task learning)
# theta = copy.deepcopy(list(model.parameters()))

# graph_landscapes_2x2(model, theta, distiller, dataset, model_size)
# graph_landscapes_combined(model, theta, distiller, dataset, model_size)
###
actor_sd_file = '../RL/experiments/1d_cartpole/10000/policy_sd.pt'
critic_sd_file = '../RL/experiments/1d_cartpole/10000/value_sd.pt'
distiller_sd_file = 'rl_models/1d_cp_b2_e5000/distiller_sd.pt'
d_critic_sd_file = 'rl_models/1d_cp_b2_e5000/critic_sd.pt'

static_dataset = True
static_critic = True
if not static_critic:
  raise NotImplementedError("RL experience gathering is set up only for static critic")
num_rl_episodes = 20
num_envs = 10
gamma = .99
gae_lambda = .95
batch_size = 2048

actor, critic, distiller, d_critic, env = setup_cartpole(actor_sd_file, critic_sd_file, distiller_sd_file, d_critic_sd_file, num_envs)

graph_synth_synth_init_space(distiller)

# actor, critic, distiller, d_critic, env = setup_cartpole(actor_sd_file, critic_sd_file, distiller_sd_file, d_critic_sd_file, num_envs)
# graph_rl_landscape_repeated(actor, critic, env, static_dataset, static_critic, num_rl_episodes, num_envs, batch_size, gamma, gae_lambda)
# graph_rl_landscape_2x2(actor, critic, distiller, d_critic, env, static_dataset, static_critic, num_rl_episodes, num_envs, batch_size, gamma, gae_lambda)
###

# model_manifold()
# calc_fim()