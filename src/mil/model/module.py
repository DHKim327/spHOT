import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class AttentionModule(nn.Module):
    def __init__(self, L, D, K):
        super(AttentionModule, self).__init__()
        self.L = L
        self.D = D
        self.K = K

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )


    def forward(self, H):
        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        return A