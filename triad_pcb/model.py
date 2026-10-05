from __future__ import annotations

import torch
from torch import nn

from .backbone import FrozenCLIPEncoder
from .losses import abnormality_loss, image_conditioned_nce
from .prompts import SemanticPromptBank, initialize_prompt_tensors
from .utils import l2_normalize


class PromptConditionedGenerator(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * dimension, 2 * dimension),
            nn.GELU(),
            nn.LayerNorm(2 * dimension),
            nn.Linear(2 * dimension, dimension),
        )

    def forward(self, features: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.network(torch.cat([features, conditions], dim=-1)))


class TriADPCB(nn.Module):
    def __init__(self, config: dict, anchors: list[str], templates: dict[str, list[str]]):
        super().__init__()
        model_cfg = config["model"]
        self.config = config
        self.encoder = FrozenCLIPEncoder(model_cfg["model_name"], freeze=model_cfg.get("freeze_clip", True))
        dimension = self.encoder.output_dim
        expected = int(model_cfg["embedding_dim"])
        if dimension != expected:
            raise ValueError(f"Backbone projection dimension {dimension} does not match configured {expected}")
        self.projection = nn.Linear(dimension, dimension, bias=False)
        nn.init.eye_(self.projection.weight)
        seed, normal, abnormal = initialize_prompt_tensors(self.encoder, anchors, templates)
        count = int(model_cfg["abnormal_prompts_per_anchor"])
        if abnormal.shape[1] < count:
            raise ValueError("The prompt file provides fewer abnormal descriptions than requested")
        self.prompt_bank = SemanticPromptBank(
            seed,
            normal,
            abnormal[:, :count],
            context_length=int(model_cfg["context_length"]),
        )
        self.generator = PromptConditionedGenerator(dimension)

    def encode_image(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches, global_features = self.encoder.encode_image(images)
        return l2_normalize(self.projection(patches)), l2_normalize(self.projection(global_features))

    def text_map(self, patches: torch.Tensor, global_features: torch.Tensor):
        cfg = self.config["model"]
        return self.prompt_bank.text_anomaly_map(
            patches,
            global_features,
            anchor_temperature=float(cfg["anchor_temperature"]),
            sharpness=float(cfg["anomaly_sharpness"]),
        )

    def training_loss(self, images: torch.Tensor, patches_per_image: int | None = None):
        patches, global_features = self.encode_image(images)
        if patches_per_image and patches.shape[1] > patches_per_image:
            indices = torch.randperm(patches.shape[1], device=patches.device)[:patches_per_image]
            patches = patches[:, indices]
        cfg = self.config["model"]
        weights = self.prompt_bank.anchor_weights(global_features, float(cfg["anchor_temperature"]))
        _, normal, abnormal = self.prompt_bank.normalized()
        batch, _, dimension = patches.shape
        # Use actual normal patch features, rather than an image-pooled vector, as
        # the generator input described by G([z^{b,m}; e]) in the manuscript.
        patch_index = torch.randint(
            patches.shape[1], (batch, normal.shape[0]), device=patches.device
        )
        base = patches.gather(
            1, patch_index[..., None].expand(-1, -1, dimension)
        )
        normal_condition = normal[None].expand(batch, -1, -1)
        pseudo_normal = self.generator(base, normal_condition)
        abnormal_index = torch.randint(abnormal.shape[1], (batch, abnormal.shape[0]), device=patches.device)
        abnormal_condition = abnormal[None].expand(batch, -1, -1, -1).gather(
            2, abnormal_index[..., None, None].expand(-1, -1, 1, dimension)
        ).squeeze(2)
        pseudo_abnormal = self.generator(base, abnormal_condition)
        nce = image_conditioned_nce(
            patches,
            weights,
            normal,
            abnormal,
            pseudo_abnormal,
            float(cfg["contrastive_temperature"]),
        )
        abnormal_loss = abnormality_loss(
            pseudo_normal, pseudo_abnormal, normal, abnormal, float(cfg["margin"])
        )
        anchor_loss = self.prompt_bank.anchor_regularization()
        training_cfg = self.config["training"]
        total = (
            float(training_cfg["beta_nce"]) * nce
            + float(training_cfg["gamma_abnormality"]) * abnormal_loss
            + float(training_cfg["mu_anchor"]) * anchor_loss
        )
        return total, {
            "loss": total.detach(),
            "nce": nce.detach(),
            "abnormality": abnormal_loss.detach(),
            "anchor": anchor_loss.detach(),
        }
