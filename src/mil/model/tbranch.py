import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class TeacherBranch(nn.Module):
  def __init__(self, input_dims, latent_dims, attention_module, 
               num_classes=2, 
               activation_function=nn.Tanh):
    super().__init__()
    self.input_dims = input_dims
    self.L = latent_dims
    self.K = 1
    self.D = latent_dims
    self.attention_module = attention_module
    self.num_classes = num_classes
    
    self.bagNN = nn.Sequential(
        nn.Linear(self.input_dims, self.L),
        activation_function(),
        nn.Linear(self.L, self.L),
        activation_function(),
        nn.Linear(self.L, self.num_classes ),
    )
    self.initialize_weights()
      
  def forward(self, input, replaceAS=None):  
    if replaceAS is not None:
      attention_weights = F.softmax(replaceAS,dim=1)
    else:
      attention_weights = self.attention_module(input)
      attention_weights = F.softmax(attention_weights,dim=1)
    
    aggregated_instance = torch.mm(attention_weights, input)
    output = aggregated_instance.squeeze()
    output = self.bagNN(output)
    return output
  
  def initialize_weights(self):
      for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)