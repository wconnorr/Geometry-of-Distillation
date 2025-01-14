import os
import copy
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassConfusionMatrix
import torchvision

import models
from helper import load_model, load_distiller_sd_fabric

def setup_sl(model_sd_file, distiller_sd_file, scale=False):
  c = 1
  hw = 28
  classes = 10
  model = models.SimpleConvNet(c, hw, hw, classes).to(device)
  
  load_model(model, model_sd_file, device)

  distiller_sd = torch.load(distiller_sd_file, map_location=device) if distiller_sd_file[-3:] == '.pt' else load_distiller_sd_fabric(distiller_sd_file) 
  distill_batch_size = distiller_sd['x'].size(0)
  print("Using {} distilled instances".format(distill_batch_size))
  distiller = models.Distiller3D(c, hw, hw, classes, distill_batch_size).to(device)
  distiller.load_state_dict(distiller_sd)

  mnist_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=.1307, std=.3081)] )if scale else torchvision.transforms.ToTensor()
  dataset = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=False, transform=mnist_transform, download=True)

  
  return model, distiller, dataset, (c, hw, hw, classes)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'], default='cpu')
parser.add_argument("-n", "--normalize", help="normalize MNIST values", action='store_true')
parser.add_argument("--distiller", help="path to distiller state dict file: either Torch .pt or Lightning .ckpt")
parser.add_argument("--model", help="path to MNIST-trained model state dict file: Torch .pt file")
parser.add_argument("result_dir", help="folder to save images")

args = parser.parse_args()
if not os.path.exists(args.result_dir):
  os.makedirs(args.result_dir)

device = torch.device(args.device)

# Load MNIST data
model_sd_file = args.model
distiller_sd_file = args.distiller

model, distiller, val_set, model_size = setup_sl(model_sd_file, distiller_sd_file, scale=args.normalize)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=512, shuffle=False)

confusion_func = MulticlassConfusionMatrix(num_classes=10)#, normalize='true')

y_truths = torch.cat([y for _,y in val_loader], dim=0)

# True model
with torch.no_grad():
  y_hats = torch.cat([model(x.to(device)).argmax(1).cpu() for x,y in val_loader], dim=0)

sl_conf = confusion_func(y_hats, y_truths)
print(sl_conf)
fig, _ = confusion_func.plot(sl_conf, add_text=False)
fig.savefig(os.path.join(args.result_dir, 'sl_confusion.png'), dpi=fig.dpi)

print("SL-trained acc:",(y_hats == y_truths).float().mean().item())

# Distillation-trained model
with torch.no_grad():
  x, y = distiller()
accs = []
# randomly initialize model
model_d = models.SimpleConvNet(*model_size).to(device)
inner_optimizer = torch.optim.SGD(model_d.parameters(), lr=distiller.inner_lr.item())
inner_loss = F.mse_loss(model_d(x), y)
inner_loss.backward()
inner_optimizer.step()
inner_optimizer.zero_grad()

with torch.no_grad():
  y_hats = torch.cat([model_d(x.to(device)).argmax(1).cpu() for x,y in val_loader], dim=0)

sl_conf = confusion_func(y_hats, y_truths)
print(sl_conf)
fig, _ = confusion_func.plot(sl_conf, add_text=False)
fig.savefig(os.path.join(args.result_dir, 'distill_confusion.png'), dpi=fig.dpi)
accs.append((y_hats == y_truths).float().mean().item())
print("distill-trained acc:",np.mean(accs))