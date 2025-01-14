# Test divergence metrics between two minima

import torch
import argparse
import torchvision

import models

import vector_env
import rl_helper

# Euclidean distance
# KL divergence distance of policies (KL(A||B) + KL(B||A))/2
# KL divergence of reward distributions???

def main():  
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'])
  parser.add_argument("model1_dir", help="path to model state dictionary")
  parser.add_argument("model2_dir", help="path to model state dictionary")
  parser.add_argument("dataset", help="dataset name", choices=['MNIST', 'CIFAR10', 'CARTPOLE', 'CENTIPEDE'])
  
  args = parser.parse_args()
  print("EXPERIMENT: ", args)

  # Load in both models
  # Load in dataset OR create env and sample
  dataset = None
  if args.dataset == 'MNIST':
    c = 1
    hw = 28
    classes = 10
    m1 = models.SimpleConvNet(c, hw, hw, classes).to(device)
    m1.load_state_dict(torch.load(args.model1_dir, map_location=device))
    m2 = models.SimpleConvNet(c, hw, hw, classes).to(device)
    m2.load_state_dict(torch.load(args.model2_dir, map_location=device))
    mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=.1307, std=.3081)] )if scale else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=True, transform=mnist_transform, download=True)
    
  elif args.dataset == 'CIFAR10':
    c = 3
    hw = 32
    classes = 10
    m1 = models.SimpleConvNet(c, hw, hw, classes).to(device)
    m1.load_state_dict(torch.load(args.model1_dir, map_location=device))
    m2 = models.SimpleConvNet(c, hw, hw, classes).to(device)
    m2.load_state_dict(torch.load(args.model2_dir, map_location=device))
    cifar10_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) if scale else torchvision.transforms.ToTensor()
    dataset = torchvision.datasets.CIFAR10(r"~/Datasets/CIFAR10", train=True,  transform=cifar10_transform, download=True)

  # TODO: CREATE MODELS, ENV, AND SAMPLE
  elif args.dataset == 'CARTPOLE':
    env = vector_env.make_cartpole_vector_env(10)
    n_actions = env.action_space[0].n
    _, c, h, w = env.observation_space.shape
    
    m1 = models.CartpoleActor().to(device)
    m1.load_state_dict(torch.load(args.model1_dir, map_location=device))
    m2 = models.CartpoleActor().to(device)
    m2.load_state_dict(torch.load(args.model2_dir, map_location=device))
    
  elif args.dataset == 'CENTIPEDE':
    env = vector_env.make_atari_vector_env(10, 'CentipedeNoFrameskip-v4')
    n_actions = env.action_space[0].n
    _, c, h, w = env.observation_space.shape
    
    m1 = models.AtariActor(c, n_actions).to(device)
    m1.load_state_dict(torch.load(args.model1_dir, map_location=device))
    m2 = models.AtariActor(c, n_actions).to(device)
    m2.load_state_dict(torch.load(args.model2_dir, map_location=device))
  
  else:
    print("DATASET {} NOT RECOGNIZED".format(args.dataset))
    quit()

  if dataset is None:
    # Do RL - get 25k from each
    states1, actions1 = perform_in_env_state_action(env, m1, 1, device, 10, batch_size, rollout_len=2500)
    states2, actions2 = perform_in_env_state_action(env, m2, 1, device, 10, batch_size, rollout_len=2500)
    dataset = models.RLDataset((np.concatenate((states1, states2), 0), np.concatenate((actions1, actions2), 0)))
    
  dataloader = DataLoader(dataset, shuffle=False, batch_size=512)
  # Print metrics
  print(parameter_euclidean_distance(m1, m2))
  print(kl_distance(m1, m2, dataloader))
  rl_helper.perform_in_env()
  
# Returns distance in parameter space
def parameter_euclidean_distance(m1, m2):
  dist = 0.
  for p1, p2 in zip(m1.parameters, m2.parameters):
    dist += ((p1-p2)**2).sum()
  # TODO: Scale by num dims? or just leave as-is?
  return dist**.5

# Returns mean KL distance (2-way KL divergence)
def kl_distance(m1, m2, dataloader):
  return (kl_divergence(m1, m2, dataloader) - kl_divergence(m2, m1, dataloader))/2

# Returns mean KL divergence over the given dataset
def kl_divergence(m1, m2, dataloader):
  sum_kl = 0
  for state, _ in dataloader:
    state = state.to(device)
    p1 = m1(state)
    p2 = m2(state)
    # assuming the outputs are logits of the policy (which is how we treat them in training)
    # sum over the classes/actions (here we sum over the batch as well, but we only divide by batch size at end)
    sum_kl += (torch.exp(p1)*(p1 - p2)).sum()
  return sum_kl / len(dataset)

if __name__ == '__main__':
  main()