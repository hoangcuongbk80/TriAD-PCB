from __future__ import annotations
import torch
import torch.nn.functional as F

def resize_image(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x.squeeze(0)

def resize_mask(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=size, mode="nearest")
    return x.squeeze(0)
