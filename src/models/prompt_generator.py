from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class PromptConditionedGenerator(nn.Module):
    """Generates feature-space variants conditioned on a prompt embedding."""
    def __init__(self, dim: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, feats: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        if feats.dim() == 2:
            x = torch.cat([feats, prompt], dim=-1)
            return F.normalize(feats + 0.1 * self.fuse(x), dim=-1)
        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(1).expand(-1, feats.size(1), -1)
        x = torch.cat([feats, prompt], dim=-1)
        return F.normalize(feats + 0.1 * self.fuse(x), dim=-1)
