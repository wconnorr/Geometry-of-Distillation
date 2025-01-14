import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

import time

import models
from helper import num_parameters, load_model
from rl_helper import perform_in_env, perform_in_env_reward, ppo_actor_loss

NOGRAD_BATCHSIZE = 4096 # higher value than training batch size, lets us parallelize geometric funcs better

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def calc_loss(model, dataset):
  """
  calculates sum cross entropy loss of `model` on provided dataset object
  returns loss
  """
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=NOGRAD_BATCHSIZE, shuffle=False)
  with torch.no_grad():
    return np.sum([F.cross_entropy(model(x.to(device)), y.to(device), reduction='sum').item() for x, y in dataloader]) / len(dataset)

def calc_loss_distill(model, distiller):
  """
  calculates SSE loss of `model` over whole distilled dataset
  returns SSE
  """
  with torch.no_grad():
    x, y = distiller()
    return F.mse_loss(model(x),y).item()

def rl_landscape(actor, theta_a, critic, theta_c, env, delta, eta, width, density, saveto, num_rl_episodes, num_envs, batch_size, gamma, gae_lambda, static_dataset=False, static_critic=False):
  """
  produces a loss landscape on an RL environment using PPO actor loss
  centered on theta and produces a graph with axes delta and eta
  """
  
  begin = time.time()

  alist = np.linspace(-width, width, density)
  blist = np.linspace(-width, width, density)
  amesh, bmesh = np.meshgrid(alist, blist)
  costs = []

  if static_dataset:
    experience_dataloader = perform_in_env(env, actor, critic, num_rl_episodes, device, num_envs, batch_size, gamma, gae_lambda, rollout_len=200)
  
  loop = tqdm(total=density*density, position=0, leave=False)
  for a,b in zip(amesh.flatten(), bmesh.flatten()):
    if static_critic:
      # Use preloaded critic
      set_model_parameters(actor, theta_a, a, delta, b, eta)
    else:
      # Consider actor-critic as a single model: interpolate both in δ, η
      set_actorcritic_parameters(actor, theta_a, critic, theta_c, a, delta, b, eta)
    
    if not static_dataset:
      # Use current theta to create dataset
      experience_dataloader = perform_in_env(env, actor, critic, num_rl_episodes, device, num_envs, batch_size, gamma, gae_lambda, rollout_len=200)
    #else: use dataset calculated at theta
    costs.append(ppo_actor_loss(actor, critic, experience_dataloader, epsilon=.1, use_entropy_loss=False).item())
    loop.update(1)
  loop.close()

  costs = np.array(costs).reshape(amesh.shape)
  fig = plt.figure()
  plt.contourf(amesh, bmesh, np.log(np.abs(costs)))#, locator=locator, levels=levels)
  plt.plot(0, 0, marker='.', linewidth=0, color='black')
  plt.colorbar(label="log cost")
  plt.title("Parameter Space")#: Final Cost={:.3e}".format(actual_cost))
  plt.xlabel("steps toward δ")
  plt.ylabel("steps toward η")
  fig.savefig(saveto, dpi=fig.dpi)
  plt.close('all')

  print("\tLandscape produced in: {}s".format(time.time()-begin))

def rl_reward_landscape(actor, theta_a, critic, theta_c, env, delta, eta, width, density, saveto, num_rl_episodes, num_envs, batch_size, gamma, gae_lambda, static_critic=False):
  """
  produces a reward landscape on an RL environment
  centered on theta and produces a graph with axes delta and eta
  """

  alist = np.linspace(-width, width, density)
  blist = np.linspace(-width, width, density)
  amesh, bmesh = np.meshgrid(alist, blist)
  rewards = []
  
  loop = tqdm(total=density*density, position=0, leave=False)
  for a,b in zip(amesh.flatten(), bmesh.flatten()):
    if static_critic:
      # Use preloaded critic
      set_model_parameters(actor, theta_a, a, delta, b, eta)
    else:
      # Consider actor-critic as a single model: interpolate both in δ, η
      set_actorcritic_parameters(actor, theta_a, critic, theta_c, a, delta, b, eta)
    
    # Use current theta to create dataset
    mean_reward = perform_in_env_reward(env, actor, critic, num_rl_episodes, device, rollout_len=200)
    rewards.append(mean_reward)
    loop.update(1)
  loop.close()

  rewards = np.array(rewards).reshape(amesh.shape)
  fig = plt.figure()
  plt.contourf(amesh, bmesh, rewards)#, locator=locator, levels=levels)
  plt.plot(0, 0, marker='.', linewidth=0, color='black')
  plt.colorbar(label="mean reward")
  plt.title("Parameter Space")#: Final Cost={:.3e}".format(actual_cost))
  plt.xlabel("steps toward δ")
  plt.ylabel("steps toward η")
  fig.savefig(saveto, dpi=fig.dpi)
  plt.close('all')

def select_normalized_direction(theta):
  """
  creates a random vector in parameter space, normalizes it, then scales layerwise to match `theta`'s magnitude
  where theta is model.parameters() of the learner
  """
  d = []
  for layer in theta:
    d_layer = []
    for filter in layer:
      d_filter = torch.randn_like(filter).unsqueeze(0) # create random vector in parameter space
      # Normalize by frobenius norm of theta's corresponding filter
      d_filter = d_filter * (torch.norm(filter,p='fro') / torch.norm(d_filter,p='fro'))
      d_layer.append(d_filter)
    d.append(torch.cat(d_layer,0))
  return d

def get_pca_directions(theta_trajectory, n):
  """
  Returns the n most significant principal component vectors of the trajectory
  Trajectory must contain at least n+1 parameter vectors with n>1

  PCA is useful for mapping a training trajectory, it is not needed to compare minima around a single point
  """
  trajectory_matrix = theta_trajectory[:-1] - theta_trajectory[-1]
  # standardize each vector
  trajectory_matrix = (trajectory_matrix - np.mean(trajectory_matrix, dim=0) / np.std(trajectory_matrix, dim=0))
  cov = ([[np.cov(x, y) for y in trajectory_matrix] for x in trajectory_matrix])

  # compute eigenvalues and eigenvectors
  vals, vecs = np.linalg(eig(cov))

  max_i = []
  for _ in range(n):
    i = np.argmax(vals)
    max_i.append(i)
    vals[i] = -float('inf')
  # return top n eigenvectors (ordered by eigenvalues)
  return vecs[max_i]

def set_model_parameters(model, theta, a, delta, b, eta):
  """
  sets model's parameters to (theta + a*delta + b*eta)
  used for interpolating along delta and eta for the landscapes
  """
  sd = model.state_dict()
  for layer, d, e, (name, _) in zip(theta, delta, eta, model.named_parameters()):
    sd[name] = layer + a*d + b*e
  model.load_state_dict(sd)

def set_actorcritic_parameters(actor, theta_a, critic, theta_c, a, delta, b, eta):
  """
  performs set_model_parameters() but for an actor and critic that form a single landscape
  only used when the critic's parameters are represented o
  """
  delta_a, delta_c = delta
  eta_a,   eta_c   = eta

  sd_a = actor.state_dict()
  for layer, d, e, (name, _) in zip(theta_a, delta_a, eta_a, actor.named_parameters()):
    sd_a[name] = layer + a*d + b*e
  actor.load_state_dict(sd_a)

  sd_c = critic.state_dict()
  for layer, d, e, (name, _) in zip(theta_c, delta_c, eta_c, critic.named_parameters()):
    sd_c[name] = layer + a*d + b*e
  critic.load_state_dict(sd_c)


# Produces a visualization centered on model.parameters() that extends in 2 dimension by width, with density points
def normed_visualization(model, theta, delta, eta, width, density, calc_loss_func, data, saveto, two_points=False, label="MNIST-trained"):
  """
  produces a loss landscape on a supervised dataset w/ the provided loss function
  centered on theta and produces a graph with axes delta and eta
  """
  actual_cost = calc_loss_func(model, data)
  print(actual_cost)
  print("Producing visualization...")
  # f(a,b) = L(θ+aδ+bη)

  begin = time.time()
  
  # alist = np.linspace(-width, width, density)
  alist = np.linspace(-.5, 1.5, density) if two_points else np.linspace(-width, width, density)
  blist = np.linspace(-width, width, density)
  amesh, bmesh = np.meshgrid(alist, blist)
  costs = []
  
  loop = tqdm(total=density*density, position=0, leave=False)
  for a,b in zip(amesh.flatten(), bmesh.flatten()):
    set_model_parameters(model, theta, a, delta, b, eta)
    costs.append(calc_loss_func(model, data))
    loop.update(1)
  costs = np.array(costs).reshape(amesh.shape)
  fig = plt.figure()
  plt.contourf(amesh, bmesh, np.log(costs))#, locator=locator, levels=levels)
  plt.plot(0, 0, marker='.', linewidth=0, label=(label if two_points else 'trained parameters'), color='black')
  if two_points:
    plt.plot(1, 0, marker='.', linewidth=0, label='Distill-trained')
    plt.legend()
  plt.colorbar(label="log cost")
  plt.title("Parameter Space")#: Final Cost={:.3e}".format(actual_cost))
  plt.xlabel("steps toward δ")
  plt.ylabel("steps toward η")
  fig.savefig(saveto, dpi=fig.dpi)

  plt.close('all')
  
  print("\tLandscape produced in: {}s".format(time.time()-begin))

