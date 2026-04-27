from __future__ import annotations
from pathlib import Path
import torch

def save_checkpoint(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str | Path, map_location: str | None = "cpu") -> dict:
    return torch.load(path, map_location=map_location)
