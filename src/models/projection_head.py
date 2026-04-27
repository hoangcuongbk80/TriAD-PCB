from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 512, out_dim: int = 512, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, patch_feats: torch.Tensor, global_feat: torch.Tensor):
        b, m, c = patch_feats.shape
        z_patch = self.net(patch_feats.reshape(-1, c)).reshape(b, m, -1)
        z_global = self.net(global_feat)
        z_patch = F.normalize(z_patch, dim=-1)
        z_global = F.normalize(z_global, dim=-1)
        return z_patch, z_global
