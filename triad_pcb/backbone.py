from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from transformers import AutoTokenizer, CLIPModel

from .utils import l2_normalize


class FrozenCLIPEncoder(nn.Module):
    """CLIP ViT encoder exposing projected patch, global, and text embeddings."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch16", freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.clip = CLIPModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if freeze:
            self.clip.requires_grad_(False)
            self.clip.eval()

    @property
    def output_dim(self) -> int:
        return int(self.clip.config.projection_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip.eval()
        return self

    def encode_image(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.clip.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=True,
            return_dict=True,
        )
        patch_hidden = outputs.last_hidden_state[:, 1:, :]
        global_hidden = outputs.pooler_output
        patch_features = self.clip.visual_projection(patch_hidden)
        global_features = self.clip.visual_projection(global_hidden)
        return l2_normalize(patch_features), l2_normalize(global_features)

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str], device: torch.device | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        tokens = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        tokens = {key: value.to(device) for key, value in tokens.items()}
        features = self.clip.get_text_features(**tokens)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        return l2_normalize(features)

