import os
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


import models
from geometry import normed_visualization, calc_loss, calc_loss_distill, select_normalized_direction
from helper import load_model, load_distiller_sd_fabric, num_parameters, setup_sl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SubsetDataset(torch.utils.data.Dataset):
  def __init__(self, original_dataset):
    super().__init__()
    self.original_dataset = original_dataset
    self.size = 0
    self.indices = torch.randperm(len(original_dataset))

  def set_size(self, size):
    self.size = size
  
  def __getitem__(self, i):
    return self.original_dataset[self.indices[i]]
  
  def __len__(self):
    return self.size

def graph_landscapes_2x2(model, theta, distiller, mnist, model_size, result_dir):
  
  # # Fix delta and eta for direct comparison
  delta = select_normalized_direction(theta)
  eta   = select_normalized_direction(theta)
  
  model_sd = copy.deepcopy(model.state_dict())
  
  print("slcentered-datasetscape:")
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss, mnist, os.path.join(result_dir, 'sltrained_dataset.png'))
  # reload model before revisualizing because we change model parameters in visualization
  model.load_state_dict(model_sd)
  print("slcentered-distillscape:")
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss_distill, distiller, os.path.join(result_dir, 'sltrained_distiller.png'))
  
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
  
  print("dlcentered-datasetscape:")
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss, mnist, os.path.join(result_dir, 'dltrained_dataset.png'))
  # reload model before revisualizing because we change model parameters in visualization
  model.load_state_dict(model_sd)
  print("dlcentered-distillscape:")
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss_distill, distiller, os.path.join(result_dir, 'dltrained_distiller.png'))
  
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

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'], default='cpu')
  parser.add_argument("--model", help="path to SL-trained model state dict file: Torch .pt file")
  parser.add_argument("-n", "--normalize", help="normalize dataset values", action='store_true')
  parser.add_argument("--dataset", help="determines which dataset to use: 'mnist' or 'cifar10'", choices=['mnist', 'cifar10'], default='mnist')
  parser.add_argument("result_dir", help="folder to save images")
  
  
  args = parser.parse_args()
  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
  
  device = torch.device(args.device)

  use_mnist = args.dataset == 'mnist'

  # PLAN : take random subsets of the dataset and see how many are required to put the minimum where the main model converges to
  
  model, _, mnist, model_size = setup_sl(args.model, None, use_mnist=use_mnist, scale=args.normalize, device=device)
  
  # "True" parameters (found from direct-task learning)
  theta = copy.deepcopy(list(model.parameters()))

  delta = select_normalized_direction(theta)
  eta   = select_normalized_direction(theta)
  
  model_sd = copy.deepcopy(model.state_dict())
  
  print("slcentered-datasetscape (goal):")
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss, mnist, os.path.join(args.result_dir, 'MNIST-full.png'))

  sub_mnist = SubsetDataset(mnist)
  
  # TODO: MORE SIZES!!!
  for size in [6, 10, 60, 100, 600, 100, 6000, 10000, len(mnist)-1]:
    model.load_state_dict(model_sd)
    sub_mnist.set_size(size)
    normed_visualization(model, theta, delta, eta, 1, 10, calc_loss, sub_mnist, os.path.join(args.result_dir, 'MNIST-{}.png'.format(size)))
    
if __name__ == '__main__':
  main()