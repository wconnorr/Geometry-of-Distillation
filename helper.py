import torch
import numpy as np

def num_parameters(model):
  count = 0
  for param in model.parameters():
    count += np.prod(list(param.shape))
  return count

def load_model(model, filepath, device='cpu'):
  sd = torch.load(filepath, map_location=device)
  model.load_state_dict(sd)
  return model