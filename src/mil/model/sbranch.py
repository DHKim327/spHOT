import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class StudentBranch(nn.Module):
  def __init__(self, input_dims, latent_dims, 
               num_classes=2, 
               activation_function=nn.Tanh):
    super().__init__()
    self.input_dims = input_dims
    self.L = latent_dims
    self.K = 1
    self.D = latent_dims
    self.num_classes = num_classes 
    
    self.instanceNN = nn.Sequential(
        nn.Linear(self.input_dims, self.L),
        activation_function(),
        nn.Linear(self.L, self.L),
        activation_function(),
        nn.Linear(self.L, self.num_classes )
      )
    self.initialize_weights()
  def forward(self, input):  
    NN_out = input
    output = self.instanceNN(NN_out)
    
    return output 
  
  def initialize_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)