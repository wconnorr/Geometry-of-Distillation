# Demonstrate the evolution of the learned distillation

# TODO: Figure out why it seems to work from epoch 0...

import os
import sys
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


import models
from geometry import normed_visualization, calc_loss, calc_loss_distill, select_normalized_direction
from helper import load_model, load_distiller_sd_fabric, num_parameters, setup_sl

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def graph_landscapes_evolution(model, theta, distiller, distiller_trajectory, distiller_epochs, mnist, model_size, result_dir):
  # Start w/ standard 2x2
  
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
  model_d_sd = copy.deepcopy(model.state_dict())
  theta_d = copy.deepcopy(list(model.parameters()))
  
  print("dlcentered-datasetscape:")
  normed_visualization(model, theta_d, delta, eta, 1, 10, calc_loss, mnist, os.path.join(result_dir, 'dltrained_dataset.png'))
  # reload model before revisualizing because we change model parameters in visualization
  model.load_state_dict(model_d_sd)
  print("dlcentered-distillscape:")
  normed_visualization(model, theta_d, delta, eta, 1, 10, calc_loss_distill, distiller, os.path.join(result_dir, 'dltrained_distiller.png'))

  for distiller_sd, epoch in zip(distiller_trajectory, distiller_epochs):
    # normed_visualization of inner-trained model on distill_evo
    distiller.load_state_dict(distiller_sd)
    # model.load_state_dict(model_sd)
    model.load_state_dict(model_d_sd)
    normed_visualization(model, theta_d, delta, eta, 1, 10, calc_loss_distill, distiller, os.path.join(result_dir, "dldl-e{}.png".format(epoch)))
    # normed_visualization(model, theta, delta, eta, 1, 10, calc_loss_distill, distiller, os.path.join(result_dir, "sldl-e{}.png".format(epoch)))

  # SANITY CHECK - should NOT create minimum
  distiller = models.Distiller3D(*model_size, 6).to(device)
  model.load_state_dict(model_sd)
  normed_visualization(model, theta, delta, eta, 1, 10, calc_loss_distill, distiller, os.path.join(result_dir, "SANITY-randomdldl.png"))
    
def main():
  print(sys.argv)
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'], default='cpu')
  parser.add_argument("--finaldistiller", help="path to distiller state dict file: either Torch .pt or Lightning .ckpt")
  parser.add_argument("--distillers", help="path to directory containing distillers")
  parser.add_argument("--model", help="path to SL-trained model state dict file: Torch .pt file")
  parser.add_argument("-n", "--normalize", help="normalize dataset values", action='store_true')
  parser.add_argument("--dataset", help="determines which dataset to use: 'mnist' or 'cifar10'", choices=['mnist', 'cifar10'], default='mnist')
  parser.add_argument("result_dir", help="folder to save images")
  
  args = parser.parse_args()
  if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
  
  device = torch.device(args.device)

  use_mnist = args.dataset == 'mnist'

  model, distiller, dataset, model_size = setup_sl(args.model, args.finaldistiller, use_mnist=use_mnist, scale=args.normalize, device=device)
  
  # "True" parameters (found from direct-task learning)
  theta = copy.deepcopy(list(model.parameters()))
  distiller_sds, epochs = crawl_distiller_files(args.distillers)
  graph_landscapes_evolution(model, theta, distiller, distiller_sds, epochs, dataset, model_size, args.result_dir)

def crawl_distiller_files(head_dir):
  from lightning.fabric import Fabric
  fabric = Fabric(accelerator='cpu')
  distillers = []
  epochs = []
  for dir in os.listdir(head_dir):
    path = os.path.join(head_dir, dir)
    if os.path.isdir(path) and dir[0] != '.' and dir.isnumeric():
      try:
        file = os.path.join(path, 'distiller_state.ckpt')
        distiller = fabric.load(file)['distiller']
        distillers.append(distiller)
        epochs.append(dir)
      except:
        pass
  assert len(distillers) == len(epochs)
  return distillers, epochs

if __name__ == '__main__':
  main()