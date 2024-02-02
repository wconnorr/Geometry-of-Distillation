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
from helper import num_parameters, load_model

NOGRAD_BATCHSIZE = 4096 # higher value than training batch size, lets us parallelize geometric funcs better

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loss_on_distiller(theta, distiller):
  x, y = distiller()
  model = models.SimpleConvNet(1, 28, 28, 10).to(device)
  set_model_parameters(model, 0, 0, 0, 0)
  return F.mse_loss(model(x), y, reduction="None")

def calc_FIM_distill(model, distiller):
  with torch.no_grad():
    x, y = distiller()
    print("x size: {}".format(x.shape))
    M = len(x)
    N = num_parameters(model)
    sigma2 = np.linalg.norm((y - model(x)).view(x.size(0),-1).cpu().numpy()) / (M-N)
    print("σ^2 = {}".format(sigma2))
  # If we want to use .backward(), we can get each row of the jacobian individually
  J = []
  model.zero_grad()
  for xi, yi in zip(x, y):
    lossi = F.mse_loss(model(xi.unsqueeze(0)), yi.unsqueeze(0))
    # NOTE: may not be loss? Maybe treat each output as its own datapoint (not sure how that makes sense tho)
    lossi.backward()
    # Flatten param.grad for all params: this is one row of J
    grads = [param.grad.flatten().cpu() for param in model.parameters()]
    J.append(torch.cat(grads, dim=0))
    model.zero_grad()
  with torch.no_grad():
    J = torch.stack(J).cpu().numpy()
    print(J.shape)
    # TODO: Need a smaller network to have reasonable J
    # POTENTIAL FIX: get one layer's J anc calculate prediction uncertainty from that.
    return J.T@J / sigma2
    # Now we can get parameter uncertainty: diag(FIM^-1); and prediction/output uncertainty: diag(J FIM^-1 J`)

def calc_loss(model, dataset):
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=NOGRAD_BATCHSIZE, shuffle=False)
  with torch.no_grad():
    return np.sum([F.cross_entropy(model(x.to(device)), y.to(device), reduction='sum').item() for x, y in dataloader]) / len(dataset)
    
def calc_loss_distill(model, distiller):
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
def normed_visualization(model, theta, delta, eta, width, density, calc_loss_func, data, saveto):
  actual_cost = calc_loss_func(model, data)
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
    costs.append(calc_loss_func(model, data))
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
