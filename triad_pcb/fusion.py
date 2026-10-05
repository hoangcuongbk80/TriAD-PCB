from __future__ import annotations

import math

import torch


def confidence_weights(
    history_confidence: float,
    reference_confidence: float,
    temperature: float,
    *,
    device,
    active: tuple[bool, bool, bool] = (True, True, True),
):
    if not any(active):
        raise ValueError("At least one branch must be active")
    reliability = torch.tensor([1.0, history_confidence, reference_confidence], device=device)
    mask = torch.tensor(active, device=device, dtype=torch.bool)
    reliability = reliability.masked_fill(~mask, -torch.inf)
    return torch.softmax(reliability / temperature, dim=0)


def fuse_maps(
    text_map: torch.Tensor,
    history_map: torch.Tensor,
    reference_map: torch.Tensor,
    weights: torch.Tensor,
    residual_weight: float,
) -> torch.Tensor:
    stacked = torch.stack([text_map, history_map, reference_map], dim=0)
    convex = torch.einsum("b,bm->m", weights, stacked)
    residual = stacked.amax(dim=0)
    return ((1.0 - residual_weight) * convex + residual_weight * residual).clamp(0, 1)


def top_q_pool(anomaly_map: torch.Tensor, q: float) -> torch.Tensor:
    k = max(1, math.ceil(float(q) * anomaly_map.numel()))
    return anomaly_map.flatten().topk(k).values.mean()
