from __future__ import annotations

import math
from collections import deque

import torch

from .utils import l2_normalize


def redundancy_aware_sample(features: torch.Tensor, sample_count: int) -> torch.Tensor:
    """Single-pass RAS with stable tie-breaking by encoder order."""
    if features.ndim != 2:
        raise ValueError("RAS expects a [patch, channel] tensor")
    count = min(max(int(sample_count), 1), features.shape[0])
    z = l2_normalize(features)
    redundancy = torch.zeros((z.shape[0],), device=z.device, dtype=z.dtype)
    if z.shape[0] > 1:
        similarities = z @ z.T
        lower = torch.tril(similarities, diagonal=-1)
        valid = torch.tril(torch.ones_like(similarities, dtype=torch.bool), diagonal=-1)
        redundancy[1:] = lower[1:].masked_fill(~valid[1:], -torch.inf).amax(dim=1)
    # The tiny monotonic offset gives deterministic encoder-order tie-breaking.
    offset = torch.arange(z.shape[0], device=z.device, dtype=z.dtype) * torch.finfo(z.dtype).eps
    indices = torch.argsort(redundancy + offset, stable=True)[:count]
    return features[indices]


def _max_cosine(query: torch.Tensor, bank: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    query = l2_normalize(query)
    bank = l2_normalize(bank)
    maxima = []
    for start in range(0, bank.shape[0], chunk_size):
        maxima.append(query @ bank[start : start + chunk_size].T)
    return torch.cat(maxima, dim=1).amax(dim=1)


class ReferenceBank:
    def __init__(self, sampling_ratio: float, minimum_patches: int = 32):
        self.sampling_ratio = float(sampling_ratio)
        self.minimum_patches = int(minimum_patches)
        self.features: torch.Tensor | None = None
        self.support_size = 0
        self.nominal_per_image = 0

    def build(self, support_features: list[torch.Tensor]) -> None:
        if not support_features:
            raise ValueError("At least one support image is required")
        retained = []
        for patches in support_features:
            nominal = math.ceil(self.sampling_ratio * patches.shape[0])
            count = min(patches.shape[0], max(nominal, self.minimum_patches))
            retained.append(redundancy_aware_sample(patches, count).detach())
            self.nominal_per_image = nominal
        self.features = torch.cat(retained, dim=0)
        self.support_size = len(retained)

    def score(self, query: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        if self.features is None:
            return torch.zeros(query.shape[0], device=query.device, dtype=query.dtype)
        similarity = _max_cosine(query, self.features.to(query.device))
        normalized = (similarity - similarity.min()) / (similarity.max() - similarity.min() + epsilon)
        return (1.0 - normalized).clamp(0, 1)

    @property
    def confidence(self) -> float:
        if self.features is None or self.support_size == 0 or self.nominal_per_image == 0:
            return 0.0
        target = self.support_size * self.nominal_per_image
        return min(1.0, self.features.shape[0] / target)


class HistoricalBank:
    def __init__(self, capacity: int, sampling_ratio: float, top_ratio: float):
        self.capacity = int(capacity)
        self.sampling_ratio = float(sampling_ratio)
        self.top_ratio = float(top_ratio)
        self._items: deque[torch.Tensor] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def confidence(self) -> float:
        return min(1.0, len(self) / self.capacity) if self.capacity else 0.0

    def score(self, query: torch.Tensor) -> torch.Tensor:
        if not self._items:
            return torch.zeros(query.shape[0], device=query.device, dtype=query.dtype)
        per_image = torch.stack([_max_cosine(query, item.to(query.device)) for item in self._items], dim=1)
        k = max(1, math.ceil(self.top_ratio * len(self._items)))
        agreement = per_image.topk(k, dim=1).values.mean(dim=1)
        return (0.5 * (1.0 - agreement)).clamp(0, 1)

    def update(self, features: torch.Tensor) -> None:
        count = math.ceil(self.sampling_ratio * features.shape[0])
        self._items.append(redundancy_aware_sample(features, count).detach())

    def clear(self) -> None:
        self._items.clear()
