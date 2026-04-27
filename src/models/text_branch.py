from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from ..prompts.anchors import PCB_ANCHORS, NORMAL_TEMPLATES, ABNORMAL_TEMPLATES
from .prompt_generator import PromptConditionedGenerator

class TextBranch(nn.Module):
    def __init__(
        self,
        text_encoder: nn.Module,
        dim: int = 512,
        num_anchors: int = 8,
        num_abnormal_prompts: int = 4,
        temperature_anchor: float = 0.07,
        sharpness: float = 5.0,
    ):
        super().__init__()
        self.text_encoder = text_encoder
        self.dim = dim
        self.num_anchors = num_anchors
        self.num_abnormal_prompts = num_abnormal_prompts
        self.temperature_anchor = temperature_anchor
        self.sharpness = sharpness

        anchors = PCB_ANCHORS[:num_anchors]
        if len(anchors) < num_anchors:
            anchors = anchors + [f"pcb part {i}" for i in range(num_anchors - len(anchors))]
        self.anchor_names = anchors

        anchor_texts = [f"a photo of a {a}" for a in anchors]
        anchor_init = text_encoder.encode_text(anchor_texts)
        self.anchor_vectors = nn.Parameter(anchor_init.clone())

        normal_init = []
        abnormal_init = []
        for a in anchors:
            ntexts = [t.format(anchor=a) for t in NORMAL_TEMPLATES]
            atexts = [t.format(anchor=a) for t in ABNORMAL_TEMPLATES[:num_abnormal_prompts]]
            normal_init.append(text_encoder.encode_text(ntexts).mean(dim=0))
            abnormal_init.append(text_encoder.encode_text(atexts))
        self.normal_prompts = nn.Parameter(torch.stack(normal_init, dim=0))
        self.abnormal_prompts = nn.Parameter(torch.stack(abnormal_init, dim=0))

        self.generator = PromptConditionedGenerator(dim=dim)

    def anchor_weights(self, global_feat: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(global_feat, self.anchor_vectors.t()) / self.temperature_anchor
        return F.softmax(logits, dim=-1)

    def forward(self, patch_feats: torch.Tensor, global_feat: torch.Tensor) -> dict:
        w = self.anchor_weights(global_feat)
        n = torch.einsum("bmd,kd->bmk", patch_feats, self.normal_prompts)
        a = torch.einsum("bmd,kjd->bmkj", patch_feats, self.abnormal_prompts).max(dim=-1).values
        sn = torch.einsum("bk,bmk->bm", w, n)
        sa = torch.einsum("bk,bmk->bm", w, a)
        atext = torch.sigmoid(self.sharpness * (sa - sn))

        pn = self.generator(patch_feats, torch.einsum("bk,kd->bd", w, self.normal_prompts))
        strongest_abn = self.abnormal_prompts.mean(dim=1)
        pa = self.generator(patch_feats, torch.einsum("bk,kd->bd", w, strongest_abn))
        return {
            "anomaly_map": atext,
            "anchor_weights": w,
            "normal_score": sn,
            "abnormal_score": sa,
            "pseudo_normal": pn,
            "pseudo_abnormal": pa,
        }

    def regularization_loss(self, anchor_seed: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.anchor_vectors, anchor_seed)

    def seed_anchor_embeddings(self) -> torch.Tensor:
        return self.anchor_vectors.detach().clone()
