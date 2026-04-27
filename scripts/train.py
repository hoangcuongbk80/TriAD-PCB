from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from src.models.triad_pcb import TriADPCB
from src.data.datasets import FolderAnomalyDataset
from src.utils.seed import set_seed
from src.utils.logger import get_logger
from src.utils.checkpoint import save_checkpoint

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = [b["mask"] for b in batch]
    if any(m is None for m in masks):
        masks_t = None
        masks_available = False
    else:
        masks_t = torch.stack(masks, dim=0)
        masks_available = True
    return images, masks_t, masks_available

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-dir", default="checkpoints")
    args = p.parse_args()

    set_seed(42)
    logger = get_logger()
    ds = FolderAnomalyDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    model = TriADPCB()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    model.train()
    for epoch in range(args.epochs):
        last_losses = None
        for images, masks, masks_available in dl:
            images = images.to(device)
            if masks is not None:
                masks = masks.to(device)
            opt.zero_grad()
            losses = model.training_losses(images, masks=masks, masks_available=masks_available)
            losses["total"].backward()
            opt.step()
            last_losses = losses
        if last_losses is not None:
            logger.info(f"Epoch {epoch+1}/{args.epochs}: total={last_losses['total'].item():.4f}")
        save_checkpoint(Path(args.save_dir) / f"epoch_{epoch+1}.pt", {"model": model.state_dict()})

if __name__ == "__main__":
    main()
