from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from ..memory.historical_bank import HistoricalBank
from ..memory.redundancy_sampling import redundancy_aware_sampling

class OnlineBranch(nn.Module):
    def __init__(
        self,
        capacity: int = 50,
        sample_ratio: float = 0.5,
        top_fraction: float = 0.3,
        gate_threshold: float = 0.5,
    ):
        super().__init__()
        self.capacity = capacity
        self.sample_ratio = sample_ratio
        self.top_fraction = top_fraction
        self.gate_threshold = gate_threshold
        self.bank = HistoricalBank(capacity=capacity)

    def reset(self) -> None:
        self.bank.clear()

    def _match_patch_to_histories(self, patch: torch.Tensor) -> torch.Tensor:
        sims = []
        patch = F.normalize(patch, dim=-1)
        for hist in self.bank.entries():
            hist = F.normalize(hist.to(patch.device), dim=-1)
            sims.append(torch.matmul(hist, patch).max())
        return torch.stack(sims, dim=0)

    def infer(self, patch_feats: torch.Tensor) -> torch.Tensor:
        b, m, d = patch_feats.shape
        if len(self.bank) == 0:
            return torch.zeros(b, m, device=patch_feats.device, dtype=patch_feats.dtype)
        out = []
        for bi in range(b):
            amap = []
            for pi in range(m):
                sims = self._match_patch_to_histories(patch_feats[bi, pi])
                if sims.numel() == 0:
                    amap.append(torch.tensor(0.0, device=patch_feats.device))
                    continue
                k = max(1, int(round(self.top_fraction * sims.numel())))
                topk = torch.topk(sims, k=k, largest=True).values
                score = 0.5 * (1.0 - topk.mean())
                amap.append(score.clamp(0, 1))
            out.append(torch.stack(amap, dim=0))
        return torch.stack(out, dim=0)

    @torch.no_grad()
    def update(self, patch_feats: torch.Tensor, image_score: float | torch.Tensor) -> None:
        if isinstance(image_score, torch.Tensor):
            image_score = float(image_score.detach().cpu().item())
        gate = 1 if image_score < self.gate_threshold else 0
        if gate == 0:
            return
        if patch_feats.dim() == 3:
            patch_feats = patch_feats[0]
        sampled = redundancy_aware_sampling(patch_feats.detach(), self.sample_ratio)
        self.bank.append(sampled)

    def confidence(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return min(1.0, len(self.bank) / self.capacity)
