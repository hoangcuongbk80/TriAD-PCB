from __future__ import annotations
import torch
import torch.nn.functional as F

def redundancy_aware_sampling(z: torch.Tensor, sample_ratio: float) -> torch.Tensor:
    """One-pass redundancy-aware sampling.
    z: [M,D]
    returns: [K,D]
    """
    if z.numel() == 0:
        return z
    if z.dim() != 2:
        raise ValueError("Expected [M,D] feature tensor.")
    m = z.size(0)
    k = max(1, int(round(m * sample_ratio)))
    z_norm = F.normalize(z, dim=-1)
    redundancy = torch.zeros(m, device=z.device)
    for i in range(m):
        if i == 0:
            redundancy[i] = 0.0
        else:
            redundancy[i] = torch.matmul(z_norm[i], z_norm[:i].T).max()
    order = torch.argsort(redundancy, descending=False)
    return z[order[:k]]
