from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import torch

@dataclass
class HistoricalEntry:
    features: torch.Tensor  # [M,D]

class HistoricalBank:
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.queue: deque[HistoricalEntry] = deque()

    def __len__(self) -> int:
        return len(self.queue)

    def clear(self) -> None:
        self.queue.clear()

    def append(self, features: torch.Tensor) -> None:
        self.queue.append(HistoricalEntry(features.detach().cpu()))
        while len(self.queue) > self.capacity:
            self.queue.popleft()

    def entries(self) -> list[torch.Tensor]:
        return [e.features for e in self.queue]
