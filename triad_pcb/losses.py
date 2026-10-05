from __future__ import annotations

import torch
import torch.nn.functional as F

from .utils import l2_normalize


def image_conditioned_nce(
    patches: torch.Tensor,
    anchor_weights: torch.Tensor,
    normal_prompts: torch.Tensor,
    abnormal_prompts: torch.Tensor,
    pseudo_abnormal: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    positive = l2_normalize(anchor_weights @ normal_prompts)
    fixed = abnormal_prompts.flatten(0, 1)[None].expand(patches.shape[0], -1, -1)
    negatives = torch.cat([fixed, pseudo_abnormal], dim=1)
    positives = torch.einsum("bmd,bd->bm", l2_normalize(patches), positive).unsqueeze(-1)
    negative_logits = torch.einsum("bmd,bnd->bmn", l2_normalize(patches), l2_normalize(negatives))
    logits = torch.cat([positives, negative_logits], dim=-1) / temperature
    targets = torch.zeros(logits.shape[:2], device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten())


def abnormality_loss(
    pseudo_normal: torch.Tensor,
    pseudo_abnormal: torch.Tensor,
    normal_prompts: torch.Tensor,
    abnormal_prompts: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    pn = l2_normalize(pseudo_normal)
    pa = l2_normalize(pseudo_abnormal)
    normal = l2_normalize(normal_prompts)
    abnormal = l2_normalize(abnormal_prompts)
    normal_alignment = 1.0 - torch.einsum("bkd,kd->bk", pn, normal)
    abnormal_similarity = torch.einsum("bkd,kjd->bkj", pa, abnormal).amax(dim=-1)
    abnormal_alignment = 1.0 - abnormal_similarity
    normal_similarity = torch.einsum("bkd,kd->bk", pa, normal)
    separation = F.relu(margin - abnormal_similarity + normal_similarity)
    return (normal_alignment + abnormal_alignment + separation).mean()
