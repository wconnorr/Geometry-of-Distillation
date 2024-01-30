import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import cv2

import models

device = torch.device('cpu')

# mnist = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=True, transform=torchvision.transforms.ToTensor(), download=True)
# cv2.imwrite('./mnist0.png', mnist[0][0].squeeze(0).numpy()*256)

distiller_sd = torch.load('./sl_exp/experiments/distill_long_highval/checkpoint_final1/distiller_sd.pt', map_location=device)
distill_batch_size = distiller_sd['x'].size(0)
print("Using {} distilled instances".format(distill_batch_size))
distiller = models.Distiller(1, 28, 28, 10, distill_batch_size).to(device)
distiller.load_state_dict(distiller_sd)

x, y = distiller()
y = torch.softmax(y, 1).detach()
for i in range(x.size(0)):
  xi = x[i].squeeze(0).detach().numpy()*256
  xi_large = np.zeros((28*8, 28*8))
  for r in range(28):
    for c in range(28):
      xi_large[r*8:(r+1)*8,c*8:(c+1)*8] = xi[r,c]
  yi = y[i].detach().numpy()*256
  y_im = np.zeros((28, 220))
  for c in range(10):
    y_im[:, c*22:(c+1)*22] = yi[c]
  cv2.imwrite('./ims/distillx{}.png'.format(i), xi_large)
  cv2.imwrite('./ims/distilly{}.png'.format(i), y_im)
  # print("[" + ','.join(['{:.2f}'.format(yi.item()) for yi in y[0]]) + "]")