from __future__ import annotations
import torch
import torch.nn.functional as F

def abnormality_loss(
    pseudo_normal: torch.Tensor,
    normal_prompt: torch.Tensor,
    pseudo_abnormal: torch.Tensor,
    abnormal_prompts: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """pseudo_normal: [B,K,D], pseudo_abnormal: [B,K,D]
    normal_prompt: [K,D], abnormal_prompts: [K,J,D]
    """
    pn = F.cosine_similarity(pseudo_normal, normal_prompt.unsqueeze(0), dim=-1)
    pa = F.cosine_similarity(
        pseudo_abnormal.unsqueeze(2),
        abnormal_prompts.unsqueeze(0),
        dim=-1
    ).max(dim=-1).values
    na = F.cosine_similarity(pseudo_abnormal, normal_prompt.unsqueeze(0), dim=-1)
    loss = (1 - pn) + (1 - pa) + torch.relu(margin - pa + na)
    return loss.mean()
