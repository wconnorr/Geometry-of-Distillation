"""
Visualizes MNIST distilled instances
"""

import numpy as np
import torch
import cv2
import argparse
import os

import models

from helper import load_distiller_sd_fabric

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'], default='cpu')
parser.add_argument("--distiller", help="path to distiller state dict file: either Torch .pt or Lightning .ckpt")
parser.add_argument("-n", "--normalize", help="normalize MNIST values", action='store_true')
parser.add_argument("result_dir", help="folder to save images")


args = parser.parse_args()
if not os.path.exists(args.result_dir):
  os.makedirs(args.result_dir)

device = torch.device(args.device)

# Load distiller sd
if args.distiller[-3:] == '.pt':
  distiller_sd = torch.load(args.distiller, map_location=device)
else: # assume its a Lightning Fabric state file
  distiller_sd = load_distiller_sd_fabric(args.distiller)

distill_batch_size = distiller_sd['x'].size(0)
distiller = models.Distiller3D(1, 28, 28, 10, distill_batch_size).to(device)
distiller.load_state_dict(distiller_sd)

# Produce instances and create images
x, y = distiller()
y = torch.softmax(y, 1).detach()
if args.normalize:
  # if the distiller was trained on normalized MNIST, we need to reverse the normalization for these instances
  mnist_mean = .1307
  mnist_std=.3081
  x = mnist_std * x + mnist_mean
for i in range(x.size(0)):
  xi = x[i].squeeze(0).detach().numpy()*256
  xi_large = np.zeros((28*8, 28*8))
  for r in range(28):
    for c in range(28):
      xi_large[r*8:(r+1)*8,c*8:(c+1)*8] = xi[r,c]
  yi = y[i].detach().numpy()*256 # start with white base to separate classes
  y_im = np.ones((220, 28))*256
  for c in range(10):
    y_im[c*22:(c+1)*22-1,:] = yi[c]
  cv2.imwrite(os.path.join(args.result_dir, 'distillx{}.png'.format(i)), xi_large)
  cv2.imwrite(os.path.join(args.result_dir, 'distilly{}.png'.format(i)), y_im)
  print("Label : [" + ','.join(['{:.2f}'.format(yi.item()) for yi in y[i]]) + "]")