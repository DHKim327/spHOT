import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    1-layer 인코더: LayerNorm -> Linear(in_dim -> latent_dim) -> GELU -> Dropout
    """
    def __init__(self, input_dim, latent_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) 