from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from ..memory.redundancy_sampling import redundancy_aware_sampling

class ReferenceBranch(nn.Module):
    def __init__(self, sample_ratio: float = 0.5, min_rep: int = 32):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.min_rep = min_rep
        self.reference_bank: torch.Tensor | None = None

    def build_bank(self, ref_patch_feats: list[torch.Tensor]) -> None:
        sampled = []
        for feats in ref_patch_feats:
            sel = redundancy_aware_sampling(feats.detach(), self.sample_ratio)
            if sel.size(0) < self.min_rep:
                k = min(feats.size(0), max(self.min_rep, int(round(feats.size(0) * 0.75))))
                sel = feats[:k].detach()
            sampled.append(sel)
        self.reference_bank = torch.cat(sampled, dim=0) if sampled else None

    def infer(self, patch_feats: torch.Tensor) -> torch.Tensor:
        if self.reference_bank is None or self.reference_bank.numel() == 0:
            return torch.zeros(patch_feats.size(0), patch_feats.size(1), device=patch_feats.device, dtype=patch_feats.dtype)
        ref = F.normalize(self.reference_bank.to(patch_feats.device), dim=-1)
        z = F.normalize(patch_feats, dim=-1)
        sim = torch.einsum("bmd,nd->bmn", z, ref).max(dim=-1).values
        minv = sim.min(dim=1, keepdim=True).values
        maxv = sim.max(dim=1, keepdim=True).values
        norm = (sim - minv) / (maxv - minv + 1e-6)
        return 1.0 - norm

    def confidence(self, nominal_per_image: int = 128) -> float:
        if self.reference_bank is None:
            return 0.0
        return float(min(1.0, self.reference_bank.size(0) / max(nominal_per_image, 1)))
