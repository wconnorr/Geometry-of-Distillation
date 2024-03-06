import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# model = models.SimpleConvNet(1, 28, 28, 10).to(device)
# print("Small conv network has {} parameters!".format(num_parameters(model)))

# model_sd_file = './sl_exp/experiments/mnist_sl_saveinit/checkpoint_final1/learner_sd.pt'
# load_model(model, model_sd_file, device)

# print(model)

# distiller_sd = torch.load('./sl_exp/experiments/distill_long_highval/checkpoint_final1/distiller_sd.pt', map_location=device)
# distill_batch_size = distiller_sd['x'].size(0)
# print("Using {} distilled instances".format(distill_batch_size))
# distiller = models.Distiller(1, 28, 28, 10, distill_batch_size).to(device)
# distiller.load_state_dict(distiller_sd)
# x, y = distiller()

# mnist = torchvision.datasets.MNIST(r"~/Datasets/MNIST", train=True, transform=torchvision.transforms.ToTensor(), download=True)

# # For comparisons, we'll scale by SL-trained model parameters!
# theta = copy.deepcopy(list(model.parameters()))
# # Fix delta and eta for direct comparison
# delta = select_normalized_direction(theta)
# eta   = select_normalized_direction(theta)

# model_sd = copy.deepcopy(model.state_dict())

# exp_name = 'TEST_close_contour_sltrained'
# normed_visualization(model, theta, delta, eta, 1, 10, calc_loss, exp_name + '_mnist.png')
# # reload model before revisualizing because we change model parameters in visualization
# model.load_state_dict(model_sd)
# normed_visualization(model, theta, delta, eta, 1, 10, calc_loss_distill, exp_name + '_distiller.png')


# for i in range(1):
#   exp_name = 'close_contour_dltrained_' + str(i)
  
#   # DISTILLATION TRAINING
#   with torch.no_grad():
#     x, y = distiller()
#   # randomly initialize model
#   model = models.SimpleConvNet(1, 28, 28, 10).to(device)
#   inner_optimizer = torch.optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
#   inner_loss = F.mse_loss(model(x), y)
#   inner_loss.backward()
#   inner_optimizer.step()
#   inner_optimizer.zero_grad()
#   model_sd = copy.deepcopy(model.state_dict())
#   theta = copy.deepcopy(list(model.parameters()))
  
#   normed_visualization(model, theta, delta, eta, 1, 10, calc_loss, exp_name + '_mnist.png')
#   # reload model before revisualizing because we change model parameters in visualization
#   model.load_state_dict(model_sd)
#   normed_visualization(model, theta, delta, eta, 1, 10, calc_loss_distill, exp_name + '_distiller.png')
  
  
# # do we care about distance of minima in parameter space? Or just the amount it misses by?
# # DISTILLATION TRAINING
# with torch.no_grad():
#   x, y = distiller()
# # randomly initialize model
# model_d = models.SimpleConvNet(1, 28, 28, 10).to(device)
# inner_optimizer = torch.optim.SGD(model_d.parameters(), lr=distiller.inner_lr.item())
# inner_loss = F.mse_loss(model_d(x), y)
# inner_loss.backward()
# inner_optimizer.step()
# inner_optimizer.zero_grad()

# # Determine how far the parameters are in Theta
# distance = 0
# theta = torch.cat([p.flatten() for p in model.parameters()], 0)
# theta_d = torch.cat([p.flatten() for p in model_d.parameters()], 0)
# print("The max distance between any thetas is {}".format(torch.max(torch.abs(theta-theta_d))))
# m = torch.argmax(torch.abs(theta-theta_d))
# print(theta[m])
# print(theta_d[m])

# exp_name = 'TEST3_both_contour'
# # set delta to line between two minima : DO NOT normalize: we want a=1 to be the line between the two
# model_d = models.SimpleConvNet(1, 28, 28, 10).to(device)
# inner_optimizer = torch.optim.SGD(model_d.parameters(), lr=distiller.inner_lr.item())
# inner_loss = F.mse_loss(model_d(x), y)
# inner_loss.backward()
# inner_optimizer.step()
# inner_optimizer.zero_grad()
# model_sd_d = copy.deepcopy(model_d.state_dict())
# theta_d = copy.deepcopy(list(model_d.parameters()))

# # delta=theta_d-theta # theta_d = theta + 1*delta
# # make eta's scale the same as delta
# delta = []
# for l, ld in zip(theta, theta_d):
#   delta.append(ld-l)
# eta = select_normalized_direction(theta)

# normed_visualization(model, theta, delta, eta, 1, 21, calc_loss, exp_name + '_mnist.png')
# # reload model before revisualizing because we change model parameters in visualization
# model.load_state_dict(model_sd)
# normed_visualization(model, theta, delta, eta, 1, 11, calc_loss_distill, exp_name + '_distiller.png')



# Model manifold curvature
# model_manifold_curvature(model, distiller, delta, './curvature_sltrained_')






######################################
# FIM

# distiller_sd = torch.load('./sl_exp/experiments/distill_long_highval/checkpoint_final1/distiller_sd.pt', map_location=device)
# distill_batch_size = distiller_sd['x'].size(0)
# print("Using {} distilled instances".format(distill_batch_size))
# distiller = models.Distiller(1, 28, 28, 10, distill_batch_size).to(device)
# distiller.load_state_dict(distiller_sd)
# with torch.no_grad():
#   x, y = distiller()
# # randomly initialize model
# model = models.SimpleConvNet(1, 28, 28, 10).to(device)
# print("Small conv network has {} parameters!".format(num_parameters(model)))
# for param in model.parameters():
#   print("\t",param.shape)
# inner_optimizer = torch.optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
# inner_loss = F.mse_loss(model(x), y)
# inner_loss.backward()
# inner_optimizer.step()
# inner_optimizer.zero_grad()


# fim = calc_FIM_distill(model, distiller)
# print(fim.shape)