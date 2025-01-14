import os
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from geometry import select_normalized_direction, normed_visualization, calc_loss_distill, rl_landscape, rl_reward_landscape
from helper import load_model, num_parameters, setup_cartpole, setup_atari

def graph_rl_landscape_2x2(actor, critic, distiller, d_critic, env, static_dataset, static_critic, num_rl_episodes, num_envs, batch_size, gamma, gae_lambda, is_cartpole, result_dir, graph_reward_landscapes=False, graph_distilled_zoomout=False, generalization_test=False, device=torch.device('cpu')):

  actor_sd = copy.deepcopy(actor.state_dict())
  
  theta_a =  copy.deepcopy(list(actor.parameters()))
  theta_c =  copy.deepcopy(list(critic.parameters()))

  
  delta = select_normalized_direction(theta_a) if static_critic else [select_normalized_direction(theta_a), select_normalized_direction(theta_c)]
  eta   = select_normalized_direction(theta_a) if static_critic else [select_normalized_direction(theta_a), select_normalized_direction(theta_c)]

  print("rl-centered on rl scape:")
  rl_landscape(actor, theta_a, critic, theta_c, env, delta, eta, 1, 10, os.path.join(result_dir, 'rltrained_environment.png'), num_rl_episodes, num_envs, batch_size, static_dataset, static_critic, gamma, gae_lambda)
  actor.load_state_dict(actor_sd)
  # Distiller landscape is supervised learning!!!
  print("rl-centered on dl scape:")
  normed_visualization(actor, theta_a, delta, eta, 1, 10, calc_loss_distill, distiller, os.path.join(result_dir, 'rltrained_distiller.png'))

  with torch.no_grad():
    x, y = distiller()
  # randomly initialize model
  if is_cartpole:
    d_actor = models.CartpoleActor().to(device)
  else:
    action_space = env.action_space[0].n
    _, state_c, state_h, state_w = env.observation_space.shape
    d_actor = models.AtariActor(state_c, action_space).to(device)
  inner_optimizer = torch.optim.SGD(d_actor.parameters(), lr=distiller.inner_lr.item())
  inner_loss = F.mse_loss(d_actor(x), y)
  inner_loss.backward()
  inner_optimizer.step()
  inner_optimizer.zero_grad()
  d_actor_sd = copy.deepcopy(d_actor.state_dict())
  d_theta_a = copy.deepcopy(list(d_actor.parameters()))

  
  delta = select_normalized_direction(d_theta_a) 
  eta   = select_normalized_direction(d_theta_a) 
  
  # TODO: Distiller-centered datasets
  print("rl-centered on rl scape:")
  rl_landscape(d_actor, d_theta_a, critic, theta_c, env, delta, eta, 1, 10, os.path.join(result_dir, 'distilltrained_environment.png'), num_rl_episodes, num_envs, batch_size, static_dataset, static_critic, gamma, gae_lambda)
  d_actor.load_state_dict(d_actor_sd)
  # Distiller landscape is supervised learning!!!
  print("dl-centered on dl scape:")
  normed_visualization(d_actor, d_theta_a, delta, eta, 1, 10, calc_loss_distill, distiller, os.path.join(result_dir, 'distilltrained_distiller.png'))
  d_actor.load_state_dict(d_actor_sd)


  # Larger network
  if generalization_test:
    with torch.no_grad():
      x, y = distiller()
    # randomly initialize model
    if is_cartpole:
      d_actor = models.CartpoleActor().to(device)
    else:
      action_space = env.action_space[0].n
      _, state_c, state_h, state_w = env.observation_space.shape
      d_actor = models.AtariActor(state_c, action_space, hidden_convs=5).to(device)
    inner_optimizer = torch.optim.SGD(d_actor.parameters(), lr=distiller.inner_lr.item())
    inner_loss = F.mse_loss(d_actor(x), y)
    inner_loss.backward()
    inner_optimizer.step()
    inner_optimizer.zero_grad()
    d_actor_sd = copy.deepcopy(d_actor.state_dict())
    d_theta_a = copy.deepcopy(list(d_actor.parameters()))

    # We can't use the same delta and eta as before, since the space is bigger
    delta2 = select_normalized_direction(d_theta_a) if static_critic else [select_normalized_direction(d_theta_a), select_normalized_direction(d_theta_c)]
    eta2   = select_normalized_direction(d_theta_a) if static_critic else [select_normalized_direction(d_theta_a), select_normalized_direction(d_theta_c)]
    
    rl_landscape(d_actor, d_theta_a, critic, theta_c, env, delta2, eta2, 1, 10, os.path.join(result_dir, 'bigdistilltrained_environment.png'), num_rl_episodes, num_envs, batch_size, static_dataset, static_critic, gamma, gae_lambda)
    d_actor.load_state_dict(d_actor_sd)
    normed_visualization(d_actor, d_theta_a, delta2, eta2, 1, 10, calc_loss_distill, distiller, os.path.join(result_dir, 'bigdistilltrained_distiller.png'))
    d_actor.load_state_dict(d_actor_sd)
  
  # ZOOMED OUT SYNTHETIC-SYNTHETIC
  if graph_distilled_zoomout:
    normed_visualization(d_actor, d_theta_a, delta, eta, 10, 100, calc_loss_distill, distiller, os.path.join(result_dir, 'distilltrained_distiller_zoomout.png'))
    d_actor.load_state_dict(d_actor_sd)

  # Reward landscapes
  if graph_reward_landscapes:
    rl_reward_landscape(actor, theta_a, critic, theta_c, env, delta, eta, 1, 10, os.path.join(result_dir, 'rltrained_environment_REWARD.png'), num_rl_episodes, num_envs, batch_size, static_critic, gamma, gae_lambda)
    d_actor.load_state_dict(d_actor_sd)
    rl_reward_landscape(d_actor, d_theta_a, critic, theta_c, env, delta, eta, 1, 10, os.path.join(result_dir, 'distilltrained_environment_REWARD.png'), num_rl_episodes, num_envs, batch_size, static_critic, gamma, gae_lambda)

def graph_distill_init_space(distiller, result_dir, device=torch.device('cpu')):
  """
  Produces two landscapes: 
    1. centered on the model initialization space mean (0) & showing 2 std devs out, w/ +delta pointing towards a minimum
    2. showing the whole one-step trajectory from initialization space (delta=0, eta=0) to final point (delta=1, eta=0)

  - thetahat = 0
  - theta0 = init
  - theta1 = trained theta
  - eta = random
  - delta = theta1, but scaled using per-layer std dev! We don't need to see theta1 (stretched out quite a bit), we just need to see immediately around theta0

    whole space (thetahat: delta = theta1) and others that are only init to theta1 (delta = theta1 - theta0)
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

  normed_visualization(d_actor, theta_hat, delta, eta, 2, 100, calc_loss_distill, distiller, os.path.join(result_dir, "zero_centered_cartpole_dd.png"))

  # TODO: zoom closely into initialization to show gradient

  # TODO: show line between init and final result
  # Make delta line between 0 and trained
  delta = [tt - t0 for tt, t0 in zip(theta_trained,theta_0)]
  d_actor.load_state_dict(sd_0)
  normed_visualization(d_actor, theta_0, delta, eta, 1, 100, calc_loss_distill, distiller, os.path.join(result_dir, "trajectory_cartpole_dd.png"), two_points=True, label="Initialization")

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'], default='cpu')
  parser.add_argument("--distiller", help="path to distiller state dict file: either Torch .pt or Lightning .ckpt")
  parser.add_argument("--d_critic", help="path to meta-RL-trained critic model state dict file: Torch .pt file")
  parser.add_argument("--actor", help="path to RL-trained actor model state dict file: Torch .pt file")
  parser.add_argument("--critic", help="path to RL-trained critic model state dict file: Torch .pt file")
  parser.add_argument("-e", "--environment", help="determines which RL environment to use: supports 'cartpole' or OpenAI Gym Atari environments", default='cartpole')
  parser.add_argument("--distill_zoomout", help="graphs 10x zoomout of distill-trained model on synthetic landscape", action="store_true")
  parser.add_argument("--reward_landscape", help="graphs additional RL landscapes w/ reward instead of loss.", action="store_true")
  parser.add_argument("--initspace_landscape", help="graphs landscapes around initialization space.", action="store_true")
  parser.add_argument("--generalization_test", help="trains OOD network on distiller and plots landscapes.", action="store_true")
  parser.add_argument("result_dir", help="folder to save images")

  # RL landscape hyperparameters: we used static datasets and static critics in all experiments
  static_dataset = True
  static_critic = True
  if not static_critic:
    raise NotImplementedError("RL experience gathering is set up only for static critic")
  # RL experience-gathering and loss hyperparameters
  num_rl_episodes = 20
  num_envs = 10
  gamma = .99
  gae_lambda = .95
  batch_size = 2048
    
  
  args = parser.parse_args()
  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
  
  device = torch.device(args.device)

  use_cartpole = args.environment == 'cartpole'

  if use_cartpole:
    actor, critic, distiller, d_critic, env = setup_cartpole(args.actor, args.critic, args.distiller, args.d_critic, num_envs, device=device)
  else:
    actor, critic, distiller, d_critic, env = setup_atari(args.actor, args.critic, args.distiller, args.d_critic, args.environment, num_envs, device=device)
    
  graph_rl_landscape_2x2(actor, critic, distiller, d_critic, env, static_dataset, static_critic, num_rl_episodes, num_envs, batch_size, gamma, gae_lambda, use_cartpole, args.result_dir, graph_reward_landscapes=args.reward_landscape, graph_distilled_zoomout=args.distill_zoomout, generalization_test=args.generalization_test, device=device)

  if args.initspace_landscape:
    graph_distill_init_space(distiller, args.result_dir, device=device)
    
if __name__ == '__main__':
  main()