from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def mask_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ]
    )


class ManifestDataset(Dataset):
    required_columns = {"image_path", "mask_path", "label", "split", "category", "source_id"}

    def __init__(
        self,
        manifest: str | Path,
        image_size: int,
        split: str,
        normal_only: bool = False,
        categories: list[str] | None = None,
    ):
        self.manifest_path = Path(manifest).resolve()
        frame = pd.read_csv(self.manifest_path).fillna("")
        missing = self.required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        frame = frame[frame["split"] == split]
        if normal_only:
            frame = frame[frame["label"].astype(int) == 0]
        if categories:
            frame = frame[frame["category"].isin(categories)]
        self.frame = frame.reset_index(drop=True)
        self.root = self.manifest_path.parent
        self.image_tf = image_transform(image_size)
        self.mask_tf = mask_transform(image_size)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        image_path = self._resolve(str(row.image_path))
        image = Image.open(image_path).convert("RGB")
        tensor = self.image_tf(image)
        label = int(row.label)
        mask_value = str(row.mask_path)
        if label and mask_value:
            mask = Image.open(self._resolve(mask_value)).convert("L")
            mask_tensor = (self.mask_tf(mask).float() / 255.0).clamp(0, 1).squeeze(0)
        else:
            mask_tensor = torch.zeros((self.image_size, self.image_size), dtype=torch.float32)
        return {
            "image": tensor,
            "mask": mask_tensor,
            "label": label,
            "image_path": str(image_path),
            "category": str(row.category),
            "source_id": str(row.source_id),
        }


def collate_samples(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "image_path": [item["image_path"] for item in batch],
        "category": [item["category"] for item in batch],
        "source_id": [item["source_id"] for item in batch],
    }

