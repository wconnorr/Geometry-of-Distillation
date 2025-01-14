# Contains functions that could be useful in the future but remain unused

def model_manifold_curvature(model, distiller, mnist, delta, saveto):
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
    v = torch.cat([layer.flatten() for layer in torch.autograd.grad(model(x).mean(1), model.parameters())], 0).cpu().numpy()
    print("v", v.shape)
    a = torch.cat([layer.flatten() for layer in torch.autograd.functional.hessian(lambda x: model(x).mean(), x)], 0).cpu().numpy()
    print("a", a.shape)
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

  # loop = tqdm(total=len(blist), position=0, leave=False)
  # dataloader = torch.utils.data.DataLoader(mnist, batch_size=NOGRAD_BATCHSIZE, shuffle=False)
  # curvature = []
  # for b in blist:
  #   # Set model parameters but without eta
  #   sd = model.state_dict()
  #   for filter, d, (name, _) in zip(theta, delta, model.named_parameters()):
  #     sd[name] = filter + b*d
  #   model.load_state_dict(sd)
  #   v = []
  #   for x,_ in dataloader:
  #     v.extend([layer.flatten() for layer in torch.autograd.grad(model(x.to(device)).mean(), model.parameters())])
  #   v = torch.cat(v, 0).flatten().numpy()
  #   a = []
  #   for x,_ in dataloader:
  #     a.extend([layer.flatten() for layer in torch.autograd.functional.hessian(lambda x: model(x).mean(), x.to(device))])  
  #   a = torch.cat(a, 0).flatten().numpy()
  #   # vel = 1st derivative of the model wrt theta
  #   # acc = 2nd derivative wrt theta
    
  #   a_par = np.dot(a,v) * v/(np.linalg.norm(v)**2)
  #   a_perp = a - a_par
  #   curvature.append(np.linalg.norm(a_perp) / np.linalg.norm(v)**2)
  #   loop.update(1)
  # fig = plt.figure()
  # plt.title("Curvature of Model Manifold: MNIST")
  # plt.plot(blist, curvature)
  # plt.xlabel("steps from θ to δ")
  # plt.xscale('log')
  # plt.ylabel("κ")
  # fig.savefig(saveto + 'sl.png', dpi=fig.dpi)
  # plt.close('all')




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




def calc_loss_stochastic_sl(model, dataset):
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=NOGRAD_BATCHSIZE, shuffle=True)
  x, y = next(dataloader)
  with torch.no_grad():
    return F.cross_entropy(model(x.to(device)), y.to(device), reduction='sum').item()




def loss_on_distiller(theta, distiller):
  """
  uses SimpleConvNet model at parameters `theta` to determine SSE loss over whole distilled dataset
  returns SSE
  """
  x, y = distiller()
  model = models.SimpleConvNet(1, 28, 28, 10).to(device)
  set_model_parameters(model, 0, 0, 0, 0)
  return F.mse_loss(model(x), y, reduction="None")



def calc_fim():
  # FIM: you should be able to calculate this on the supercomputer
  
  distiller_sd = torch.load('./sl_exp/experiments/distill_long_highval/checkpoint_final1/distiller_sd.pt', map_location=device)
  distill_batch_size = distiller_sd['x'].size(0)
  print("Using {} distilled instances".format(distill_batch_size))
  distiller = models.Distiller3D(1, 28, 28, 10, distill_batch_size).to(device)
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




def graph_rl_landscape_repeated(actor, critic, env, static_dataset, static_critic, num_rl_episodes, num_envs, batch_size, gamma, gae_lambda):

  theta_a =  copy.deepcopy(list(actor.parameters()))
  theta_c =  copy.deepcopy(list(critic.parameters()))

  actor_sd = copy.deepcopy(actor.state_dict())
  
  delta = select_normalized_direction(theta_a) if static_critic else [select_normalized_direction(theta_a), select_normalized_direction(theta_c)]
  eta   = select_normalized_direction(theta_a) if static_critic else [select_normalized_direction(theta_a), select_normalized_direction(theta_c)]

  exp_name = "rl_landscape_rep"

  for i in range(20):
    rl_landscape(actor, theta_a, critic, theta_c, env, delta, eta, 1, 10, exp_name + '_cartpole{}.png'.format(i), num_rl_episodes, num_envs, batch_size, static_dataset, static_critic, gamma, gae_lambda)
    actor.load_state_dict(actor_sd)