import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

import models

NOGRAD_BATCHSIZE = 4096 # higher value than training batch size, lets us parallelize geometric funcs better

device = 'cuda'#'cuda' if torch.cuda.is_available() else 'cpu'

def calc_FIM(model, dataset):
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=NOGRAD_BATCHSIZE, shuffle=False)
  with torch.no_grad():
    M = len(dataset)
    N = num_parameters(model)
    sigma2 = np.linalg.norm(torch.cat([(y - model(x)).view(x.size(0),-1) for x,y in dataloader], dim=0).numpy()) / (M-N)
  J = torch.cat([torch.autograd.functional.jacobian(model, x).view(x.size(0),-1) for x,_ in dataloader], dim=0).numpy()
  return J.T@J / sigma2

def loss_on_distiller(theta):
  model = models.SimpleConvNet(1, 28, 28, 10).to(device)
  set_model_parameters(model, theta)
  return F.mse_loss(model(x), y, reduction="None")

def calc_FIM_distill(model, distiller):
  with torch.no_grad():
    x, y = distiller()
    print("x size: {}".format(x.shape))
    M = len(x)
    N = num_parameters(model)
    sigma2 = np.linalg.norm((y - model(x)).view(x.size(0),-1).cpu().numpy()) / (M-N)
    print("σ^2 = {}".format(sigma2))
  # J = torch.autograd.functional.jacobian(model, x)
  theta = model.parameters()
  J = torch.autograd.functional.jacobian(loss_on_distiller, theta)
  print("Jacobian size: {}".format(J.shape))
  # NOTE: Jacobian is combined (output_size, input_size): [(6, 10), (6,1,28,28)]
  # TODO: SHOULD BE (# datapoints x # params), so FIM is (# params x # params)!!!!
  #       THAT IS TO SAY, WE WANT jacobian(loss_on_xy, θ), where loss is not reduced
  J = J.view(-1, x.flatten().size(0)).cpu().numpy()
  print("Resized Jacobian: {}".format(J.shape))
  return J.T@J / sigma2

def num_parameters(model):
  count = 0
  for param in model.parameters():
    count += np.prod(list(param.shape))
  return count

def calc_loss(model):
  dataloader = torch.utils.data.DataLoader(mnist, batch_size=NOGRAD_BATCHSIZE, shuffle=False)
  with torch.no_grad():
    return np.sum([F.cross_entropy(model(x.to(device)), y.to(device), reduction='sum').item() for x, y in dataloader]) / len(mnist)
    
def calc_loss_distill(model):
  with torch.no_grad():
    x, y = distiller()
    return F.mse_loss(model(x),y).item()

def select_normalized_direction(theta):
  # Where theta is model.parameters() of the learner
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

def set_model_parameters(model, theta, a, delta, b, eta):
  # theta + a*delta + b*eta)
  sd = model.state_dict()
  for layer, d, e, (name, _) in zip(theta, delta, eta, model.named_parameters()):
    sd[name] = layer + a*d + b*e
  model.load_state_dict(sd)

# Produces a visualization centered on model.parameters() that extends in 2 dimension by width, with density points
def normed_visualization(model, theta, delta, eta, width, density, calc_loss_func, saveto):
  actual_cost = calc_loss_func(model)
  print(actual_cost)
  print("Producing visualization...")
  # f(a,b) = L(θ+aδ+bη)
  
  # alist = np.linspace(-width, width, density)
  alist = np.linspace(-.5, 1.5, density)
  blist = np.linspace(-width, width, density)
  amesh, bmesh = np.meshgrid(alist, blist)
  costs = []
  
  loop = tqdm(total=density*density, position=0, leave=False)
  for a,b in zip(amesh.flatten(), bmesh.flatten()):
    set_model_parameters(model, theta, a, delta, b, eta)
    costs.append(calc_loss_func(model))
    loop.update(1)
  costs = np.array(costs).reshape(amesh.shape)
  fig = plt.figure()
  # locator = matplotlib.ticker.LogLocator()
  # levels = np.logspace(-4.5, 8.5, 53)
  # plt.contourf(theta1_mesh, theta2_mesh, costs, locator=locator, levels=levels)
  plt.contourf(amesh, bmesh, np.log(costs))#, locator=locator, levels=levels)
  plt.plot(0, 0, marker='.', linewidth=0, label='MNIST-trained')
  plt.plot(1, 0, marker='.', linewidth=0, label='Distill-trained')
  plt.colorbar(label="log cost")
  plt.legend()
  plt.title("Parameter Space")#: Final Cost={:.3e}".format(actual_cost))
  plt.xlabel("steps toward δ")
  plt.ylabel("steps toward η")
  fig.savefig(saveto, dpi=fig.dpi)

  plt.close('all')

def model_manifold_curvature(model, distiller, delta, saveto):
  theta = copy.deepcopy(list(model.parameters()))
  blist = np.logspace(-1, 1, 20)
  loop = tqdm(total=len(blist), position=0, leave=False)
  curvature = []
  x, _ = distiller()
  for b in blist:
    # Set model parameters but without eta
    sd = model.state_dict()
    for filter, d, (name, _) in zip(theta, delta, model.named_parameters()):
      sd[name] = filter + b*d
    model.load_state_dict(sd)
    v = torch.cat([layer.flatten() for layer in torch.autograd.grad(model(x).mean(), model.parameters())], 0).numpy()
    a = torch.cat([layer.flatten() for layer in torch.autograd.functional.hessian(lambda x: model(x).mean(), x)], 0).numpy()
    # vel = 1st derivative of the model wrt theta
    # acc = 2nd derivative wrt theta
    
    a_par = np.dot(a,v) * v/(np.linalg.norm(v)**2)
    a_perp = a - a_par
    curvature.append(np.linalg.norm(a_perp) / np.linalg.norm(v)**2)
    loop.update(1)
  fig = plt.figure()
  plt.title("Curvature of Model Manifold: Distilled")
  plt.plot(blist, curvature)
  plt.xlabel("steps from θ to δ")
  plt.xscale('log')
  plt.ylabel("κ")
  fig.savefig(saveto + 'dl.png', dpi=fig.dpi)
  plt.close('all')

  loop = tqdm(total=len(blist), position=0, leave=False)
  dataloader = torch.utils.data.DataLoader(mnist, batch_size=NOGRAD_BATCHSIZE, shuffle=False)
  curvature = []
  for b in blist:
    # Set model parameters but without eta
    sd = model.state_dict()
    for filter, d, (name, _) in zip(theta, delta, model.named_parameters()):
      sd[name] = filter + b*d
    model.load_state_dict(sd)
    v = []
    for x,_ in dataloader:
      v.extend([layer.flatten() for layer in torch.autograd.grad(model(x.to(device)).mean(), model.parameters())])
    v = torch.cat(v, 0).flatten().numpy()
    a = []
    for x,_ in dataloader:
      a.extend([layer.flatten() for layer in torch.autograd.functional.hessian(lambda x: model(x).mean(), x.to(device))])  
    a = torch.cat(a, 0).flatten().numpy()
    # vel = 1st derivative of the model wrt theta
    # acc = 2nd derivative wrt theta
    
    a_par = np.dot(a,v) * v/(np.linalg.norm(v)**2)
    a_perp = a - a_par
    curvature.append(np.linalg.norm(a_perp) / np.linalg.norm(v)**2)
    loop.update(1)
  fig = plt.figure()
  plt.title("Curvature of Model Manifold: MNIST")
  plt.plot(blist, curvature)
  plt.xlabel("steps from θ to δ")
  plt.xscale('log')
  plt.ylabel("κ")
  fig.savefig(saveto + 'sl.png', dpi=fig.dpi)
  plt.close('all')

def load_model(model, filepath):
  sd = torch.load(filepath, map_location=device)
  model.load_state_dict(sd)
  return model

# model = models.SimpleConvNet(1, 28, 28, 10).to(device)
# print("Small conv network has {} parameters!".format(num_parameters(model)))

# model_sd_file = './sl_exp/experiments/mnist_sl_saveinit/checkpoint_final1/learner_sd.pt'
# load_model(model, model_sd_file)

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

distiller_sd = torch.load('./sl_exp/experiments/distill_long_highval/checkpoint_final1/distiller_sd.pt', map_location=device)
distill_batch_size = distiller_sd['x'].size(0)
print("Using {} distilled instances".format(distill_batch_size))
distiller = models.Distiller(1, 28, 28, 10, distill_batch_size).to(device)
distiller.load_state_dict(distiller_sd)
with torch.no_grad():
  x, y = distiller()
# randomly initialize model
model = models.SimpleConvNet(1, 28, 28, 10).to(device)
print("Small conv network has {} parameters!".format(num_parameters(model)))
for param in model.parameters():
  print("\t",param.shape)
inner_optimizer = torch.optim.SGD(model.parameters(), lr=distiller.inner_lr.item())
inner_loss = F.mse_loss(model(x), y)
inner_loss.backward()
inner_optimizer.step()
inner_optimizer.zero_grad()


fim = calc_FIM_distill(model, distiller)
print(fim.shape)