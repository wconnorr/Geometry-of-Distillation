import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical

from models import RLDataset, CartpoleActor

def perform_in_env(env, actor, critic, num_episodes, device, num_envs, batch_size, gamma, gae_lambda, rollout_len=200):
  state_space = env.observation_space.shape[1:]
  states = torch.zeros((rollout_len, num_envs, *state_space)).to(device)
  actions = torch.zeros((rollout_len, num_envs)).to(device)
  prior_policy = torch.zeros((rollout_len, num_envs)).to(device) # stored for chosen action only!
  rewards = torch.zeros((rollout_len, num_envs)).to(device)
  dones = torch.zeros((rollout_len, num_envs)).to(device)
  values = torch.zeros((rollout_len, num_envs)).to(device)
  returns = None # this will be overwritten with a new tensor, rather than being filled piecewise
  advantages = torch.zeros((rollout_len, num_envs)).to(device)
  entropies = torch.zeros((rollout_len, num_envs)).to(device)

  rollout = [states, actions, prior_policy, rewards, dones, values, returns, advantages, entropies]

  # Begin by resetting env
  state = torch.from_numpy(env.reset()[0]).to(device)
  
  for ep in range(num_episodes):
    final_rewards, state, done = perform_rollout(actor, critic, env, rollout, rollout_len, state, device)
    # print("ep_rewards:", final_rewards)
  if critic is not None:
    general_advantage_estimation(critic, rollout, state, done, gamma, gae_lambda, device)
  memory_dataloader = DataLoader(RLDataset(rollout), batch_size=batch_size, shuffle=True)
  return memory_dataloader

def perform_in_env_state_action(env, actor, num_episodes, device, num_envs, batch_size, rollout_len=200):
  state_space = env.observation_space.shape[1:]
  states = torch.zeros((rollout_len, num_envs, *state_space)).to(device)
  actions = torch.zeros((rollout_len, num_envs)).to(device)

  rollout = [states, actions]

  # Begin by resetting env
  state = torch.from_numpy(env.reset()[0]).to(device)
  
  for ep in range(num_episodes):
    final_rewards, state, done = perform_rollout_state_action(actor, env, rollout, rollout_len, state, device)
  return rollout

def perform_in_env_reward(env, actor, critic, num_episodes, device, rollout_len=200):
  all_final_rewards = []
  
  # Begin by resetting env
  state = torch.from_numpy(env.reset()[0]).to(device)
  
  for ep in range(num_episodes):
    final_rewards, _, _ = perform_rollout_reward(actor, critic, env, rollout_len, state, device)
    all_final_rewards.extend(final_rewards)
  return np.mean(all_final_rewards)

def ppo_actor_loss(actor, critic, experience_dataloader, epsilon=.1, use_entropy_loss=False):
  # sum of losses on experience dataset
  actor_loss = 0
  entropy_loss = 0
  for transition in experience_dataloader:
    actor_loss_i, entropy_loss_i = calculate_actor_loss(actor, transition, epsilon)
    actor_loss += actor_loss_i
    entropy_loss += entropy_loss_i
  if use_entropy_loss:
    actor_loss += entropy_loss
  return actor_loss

def calculate_actor_loss(actor, transition, epsilon):
  states, actions, prior_policy, _, _, _, _, advantages, entropies = transition
  
  current_policy = actor(states)
  current_policy = current_policy[F.one_hot(actions.long(), current_policy.size(-1)).bool()]

  # calculate ratio quicker this way, rather than softmaxing them both
  ratio = (current_policy - prior_policy).exp()

  # normalize advantages
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  policy_loss = -torch.min(advantages*ratio, advantages*torch.clamp(ratio, 1-epsilon, 1+epsilon)).mean()
  return policy_loss, entropies.mean()

def act(actor, critic, state, encoder=None):
  with torch.no_grad():
    value = critic(state) if critic is not None else None
    if encoder is None:
      policy = actor(state)
    else:
      policy = actor(encoder(state))
    probs = Categorical(logits=policy)
    action = probs.sample()
  return action, probs.log_prob(action), probs.entropy(), value

# Performs 1 rollout of fixed length of the agent acting in the environment.
def perform_rollout(agent, critic, vec_env, rollout, rollout_len, state, device):
  with torch.no_grad():
    final_rewards = []
    states, actions, prior_policy, rewards, dones, values, _, _, entropies = rollout

    # Episode loop
    for i in range(rollout_len):
      # Agent chooses action
      action, action_distribution, entropy, value = act(agent, critic, state.to(device))

      # Env takes step based on action
      next_state, reward, done, _, info = vec_env.step(action.cpu().numpy())

      # Store step for learning
      states[i] = state
      actions[i] = action
      prior_policy[i] = action_distribution
      rewards[i] = torch.from_numpy(reward)
      dones[i] = torch.from_numpy(done)
      if critic is not None:
        values[i] = value.squeeze(1)
      entropies[i] = entropy

      state = torch.from_numpy(next_state)

      if isinstance(info, dict) and 'final_info' in info.keys():
        epis = [a for a in info['final_info'] if a is not None]
        for item in epis:
          final_rewards.append(int(item['episode']['r'])) # REWARDS TURNED TO INTEGERS FOR CHEAPER CHECKPOINTING

  return final_rewards, state, done # no need to return rollout, its updated in-place
  
# Performs 1 rollout of fixed length of the agent acting in the environment. Returns only state and action
def perform_rollout_state_action(agent, vec_env, rollout, rollout_len, state, device):
  with torch.no_grad():
    states, actions = rollout

    # Episode loop
    for i in range(rollout_len):
      # Agent chooses action
      action, _, _, _= act(agent, critic, state.to(device))

      # Env takes step based on action
      next_state, _, _, _, info = vec_env.step(action.cpu().numpy())

      # Store step for learning
      states[i] = state
      actions[i] = action

      state = torch.from_numpy(next_state)

  return state # no need to return rollout, its updated in-place
  
# Performs 1 rollout of fixed length of the agent acting in the environment.
def perform_rollout_reward(agent, critic, vec_env, rollout_len, state, device):
  with torch.no_grad():
    final_rewards = []

    # Episode loop
    for i in range(rollout_len):
      # Agent chooses action
      action, action_distribution, entropy, value = act(agent, critic, state.to(device))

      # Env takes step based on action
      next_state, reward, done, _, info = vec_env.step(action.cpu().numpy())

      state = torch.from_numpy(next_state)

      if isinstance(info, dict) and 'final_info' in info.keys():
        epis = [a for a in info['final_info'] if a is not None]
        for item in epis:
          final_rewards.append(int(item['episode']['r'])) # REWARDS TURNED TO INTEGERS FOR CHEAPER CHECKPOINTING

  return final_rewards, state, done # no need to return rollout, its updated in-place
  
# Calculates advantage and return, bootstrapping using value when environment has not terminated
# Modifies them in-place in the rollout
def general_advantage_estimation(critic, rollout, next_state, next_done, gamma, gae_lambda, device):
  _, _, _, rewards, dones, values, returns, advantages, _ = rollout
  rollout_len = rewards.size(0)

  with torch.no_grad():
    next_value = critic(next_state.to(device)).squeeze()
    last_lambda = 0

    nextnonterminal = 1. - torch.from_numpy(next_done).float().to(device)
    nextvalues = next_value
    delta = rewards[rollout_len-1] + gamma * nextvalues*nextnonterminal - values[rollout_len-1]
    advantages[rollout_len-1] = last_lambda = delta # + (gamma * gae_lambda * nextnonterminal * last_lambda = 0 at iteration 0), so we can leave this part out
    for t in reversed(range(rollout_len-1)):
      nextnonterminal = 1.0 - dones[t+1]
      nextvalues = values[t+1]
      delta = rewards[t] + gamma * nextvalues*nextnonterminal - values[t]
      advantages[t] = last_lambda = delta + gamma * gae_lambda * nextnonterminal * last_lambda
    returns = advantages + values
    rollout[6] = returns