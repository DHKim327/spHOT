import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
# from https://github.com/kahnchana/opl/blob/master/loss.py
class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5, device='cuda'):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma
        self.device = device

    def forward(self, features, labels=None):
        device = self.device

        #  features are normalized
        # print(features.shape)
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs ### ? 

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean
        return loss