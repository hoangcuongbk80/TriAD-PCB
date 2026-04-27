from __future__ import annotations
import torch

def focal_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.clamp(eps, 1 - eps)
    ce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    p_t = target * pred + (1 - target) * (1 - pred)
    loss = alpha * (1 - p_t).pow(gamma) * ce
    return loss.mean()

def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.clamp(0, 1)
    num = 2 * (pred * target).sum(dim=(-2, -1)) + eps
    den = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1)) + eps
    return (1 - num / den).mean()

def segmentation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return focal_loss(pred, target) + dice_loss(pred, target)
