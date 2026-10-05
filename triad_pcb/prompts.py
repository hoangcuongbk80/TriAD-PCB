from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn

from .utils import l2_normalize


def load_anchors(path: str | Path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    anchors = data["anchors"] if isinstance(data, dict) else data
    if not anchors:
        raise ValueError("The anchor list is empty")
    return [str(value) for value in anchors]


def load_prompt_templates(path: str | Path) -> dict[str, list[str]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    key_map = {
        "anchor_seed": "normal_seed_templates",
        "normal": "normal_prompt_templates",
        "abnormal": "abnormal_prompt_templates",
    }
    required = set(key_map.values())
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Prompt specification is missing: {sorted(missing)}")
    return {short: [str(item) for item in data[source]] for short, source in key_map.items()}


@torch.no_grad()
def initialize_prompt_tensors(encoder, anchors: list[str], templates: dict[str, list[str]]):
    device = next(encoder.parameters()).device

    def mean_embedding(patterns: list[str], anchor: str) -> torch.Tensor:
        texts = [pattern.format(anchor=anchor) for pattern in patterns]
        return l2_normalize(encoder.encode_text(texts, device=device).mean(dim=0))

    anchor_seed = torch.stack([mean_embedding(templates["anchor_seed"], anchor) for anchor in anchors])
    normal = torch.stack([mean_embedding(templates["normal"], anchor) for anchor in anchors])
    abnormal = torch.stack(
        [
            torch.stack(
                [encoder.encode_text([pattern.format(anchor=anchor)], device=device)[0]
                 for pattern in templates["abnormal"]]
            )
            for anchor in anchors
        ]
    )
    return l2_normalize(anchor_seed), l2_normalize(normal), l2_normalize(abnormal)


class SemanticPromptBank(nn.Module):
    """Learnable anchors and embedding-space prompt context tokens.

    The natural-language prompt embeddings remain fixed initialization targets. A
    configurable set of context tokens is optimized for each normal and abnormal
    prompt, and its mean residual is added to the corresponding initialization.
    This keeps the initialization reproducible while exposing the context length
    specified in the manuscript as an actual trainable model parameter.
    """

    def __init__(
        self,
        anchor_seed: torch.Tensor,
        normal: torch.Tensor,
        abnormal: torch.Tensor,
        context_length: int,
    ):
        super().__init__()
        if context_length < 1:
            raise ValueError("context_length must be positive")
        self.register_buffer("anchor_seed", anchor_seed.detach().clone())
        self.register_buffer("normal_seed", normal.detach().clone())
        self.register_buffer("abnormal_seed", abnormal.detach().clone())
        self.anchors = nn.Parameter(anchor_seed.detach().clone())
        self.normal_context = nn.Parameter(
            torch.empty(normal.shape[0], context_length, normal.shape[-1])
        )
        self.abnormal_context = nn.Parameter(
            torch.empty(*abnormal.shape[:2], context_length, abnormal.shape[-1])
        )
        nn.init.normal_(self.normal_context, std=0.002)
        nn.init.normal_(self.abnormal_context, std=0.002)

    @property
    def num_anchors(self) -> int:
        return int(self.anchors.shape[0])

    def normalized(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normal = self.normal_seed + self.normal_context.mean(dim=1)
        abnormal = self.abnormal_seed + self.abnormal_context.mean(dim=2)
        return l2_normalize(self.anchors), l2_normalize(normal), l2_normalize(abnormal)

    def anchor_weights(self, global_features: torch.Tensor, temperature: float) -> torch.Tensor:
        anchors, _, _ = self.normalized()
        logits = l2_normalize(global_features) @ anchors.T
        return torch.softmax(logits / temperature, dim=-1)

    def text_anomaly_map(
        self,
        patch_features: torch.Tensor,
        global_features: torch.Tensor,
        anchor_temperature: float,
        sharpness: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        anchors, normal, abnormal = self.normalized()
        del anchors
        patches = l2_normalize(patch_features)
        weights = self.anchor_weights(global_features, anchor_temperature)
        normal_response = torch.einsum("bmd,kd->bmk", patches, normal)
        abnormal_response = torch.einsum("bmd,kjd->bmkj", patches, abnormal).amax(dim=-1)
        normal_score = torch.einsum("bmk,bk->bm", normal_response, weights)
        abnormal_score = torch.einsum("bmk,bk->bm", abnormal_response, weights)
        anomaly = torch.sigmoid(sharpness * (abnormal_score - normal_score))
        return anomaly, weights

    def anchor_regularization(self) -> torch.Tensor:
        return (self.anchors - self.anchor_seed).square().sum()
