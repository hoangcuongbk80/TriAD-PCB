from __future__ import annotations
import hashlib
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

def _tokenize(text: str, max_tokens: int = 16) -> list[str]:
    tokens = text.lower().replace("-", " ").replace(",", " ").replace(".", " ").split()
    return tokens[:max_tokens] if tokens else ["<empty>"]

class HashTextEncoder(nn.Module):
    """Simple text encoder based on hashed token embeddings."""
    def __init__(self, embed_dim: int = 512, vocab_size: int = 8192):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _token_id(self, token: str) -> int:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        return int(h, 16) % self.vocab_size

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        device = self.token_embed.weight.device
        ids = []
        for t in texts:
            toks = _tokenize(t)
            ids.append([self._token_id(tok) for tok in toks])
        max_len = max(len(x) for x in ids)
        token_ids = torch.zeros(len(ids), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(ids), max_len, dtype=torch.float32, device=device)
        for i, seq in enumerate(ids):
            token_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, :len(seq)] = 1.0
        emb = self.token_embed(token_ids)
        emb = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        emb = self.proj(emb)
        return F.normalize(emb, dim=-1)

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.encode_text(texts)

class ConvPatchEncoder(nn.Module):
    """Backbone producing patch-grid features and a global descriptor."""
    def __init__(self, in_channels: int = 3, embed_dim: int = 512, patch_size: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        hidden = max(embed_dim // 2, 64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        feat = self.stem(x)
        b, c, h, w = feat.shape
        patch_feats = feat.flatten(2).transpose(1, 2).contiguous()
        patch_feats = self.out_norm(patch_feats)
        global_feat = patch_feats.mean(dim=1)
        global_feat = F.normalize(global_feat, dim=-1)
        patch_feats = F.normalize(patch_feats, dim=-1)
        return patch_feats, global_feat, (h, w)

class CLIPLikeBackbone(nn.Module):
    def __init__(self, embed_dim: int = 512, patch_size: int = 16):
        super().__init__()
        self.image_encoder = ConvPatchEncoder(embed_dim=embed_dim, patch_size=patch_size)
        self.text_encoder = HashTextEncoder(embed_dim=embed_dim)

    def encode_image(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        return self.image_encoder(x)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        return self.text_encoder(texts)
