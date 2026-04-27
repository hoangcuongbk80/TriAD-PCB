from __future__ import annotations
from pathlib import Path
from typing import Optional, Callable
import csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def _default_image_loader(path: str | Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = torch.from_numpy(arr).permute(2, 0, 1)
    return arr

class FolderAnomalyDataset(Dataset):
    """Generic dataset:
    root/
      images/*.png
      masks/*.png  (optional)
      labels.csv   (optional: filename,label)
    """
    def __init__(
        self,
        root: str | Path,
        image_dir: str = "images",
        mask_dir: str = "masks",
        label_csv: str = "labels.csv",
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = sorted([p for p in self.image_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])

        self.labels = {}
        csv_path = self.root / label_csv
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.labels[row["filename"]] = int(row["label"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = _default_image_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)

        mask_path = self.mask_dir / img_path.name
        mask = None
        if mask_path.exists():
            m = Image.open(mask_path).convert("L")
            m = np.asarray(m).astype(np.float32) / 255.0
            mask = torch.from_numpy(m).unsqueeze(0)
            if self.mask_transform is not None:
                mask = self.mask_transform(mask)

        label = self.labels.get(img_path.name, int(mask is not None) if mask is not None else -1)
        return {
            "image": image,
            "mask": mask,
            "label": label,
            "path": str(img_path),
        }

class ReferenceDataset(FolderAnomalyDataset):
    pass

class StreamDataset(FolderAnomalyDataset):
    pass
