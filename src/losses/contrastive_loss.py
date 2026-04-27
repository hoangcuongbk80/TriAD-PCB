from __future__ import annotations
import torch
import torch.nn.functional as F

def image_conditioned_contrastive_loss(
    patches: torch.Tensor,
    positive_prompt: torch.Tensor,
    negative_bank: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """patches: [B,M,D], positive_prompt: [B,D], negative_bank: [N,D] or [B,N,D]"""
    if negative_bank.dim() == 2:
        negative_bank = negative_bank.unsqueeze(0).expand(patches.size(0), -1, -1)
    patches = F.normalize(patches, dim=-1)
    positive_prompt = F.normalize(positive_prompt, dim=-1)
    negative_bank = F.normalize(negative_bank, dim=-1)

    pos = torch.einsum("bmd,bd->bm", patches, positive_prompt) / temperature
    neg = torch.einsum("bmd,bnd->bmn", patches, negative_bank) / temperature
    logits = torch.cat([pos.unsqueeze(-1), neg], dim=-1)
    labels = torch.zeros(patches.size(0), patches.size(1), dtype=torch.long, device=patches.device)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    return loss
