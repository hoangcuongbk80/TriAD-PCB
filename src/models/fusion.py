from __future__ import annotations
import torch

def fuse_anomaly_maps(
    a_text: torch.Tensor,
    a_hist: torch.Tensor,
    a_ref: torch.Tensor,
    hist_conf: float,
    ref_conf: float,
    text_conf: float = 1.0,
    fusion_temperature: float = 0.5,
    residual_weight: float = 0.5,
    top_q: float = 0.1,
) -> dict:
    device = a_text.device
    u = torch.tensor([text_conf, float(hist_conf), float(ref_conf)], device=device, dtype=a_text.dtype)
    weights = torch.softmax(u / fusion_temperature, dim=0)
    fused = weights[0] * a_text + weights[1] * a_hist + weights[2] * a_ref
    amax = torch.stack([a_text, a_hist, a_ref], dim=0).max(dim=0).values
    refined = (1 - residual_weight) * fused + residual_weight * amax
    k = max(1, int(round(top_q * refined.size(1))))
    topq = torch.topk(refined, k=k, dim=1).values.mean(dim=1)
    return {
        "fused_map": fused,
        "refined_map": refined,
        "image_score": topq,
        "weights": weights,
    }
