from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import load_yaml, resolve_path, save_yaml
from .data import ManifestDataset, collate_samples
from .model import TriADPCB
from .prompts import load_anchors, load_prompt_templates
from .utils import device_from_arg, seed_everything


def train(config_path: str, seed: int, device_arg: str | None = None, run_name: str | None = None) -> Path:
    config = load_yaml(config_path)
    seed_everything(seed)
    device = device_from_arg(device_arg)
    dataset_cfg = config["dataset"]
    dataset = ManifestDataset(
        resolve_path(config, dataset_cfg["manifest"]),
        int(dataset_cfg["image_size"]),
        "optimization",
        normal_only=True,
        categories=dataset_cfg.get("categories"),
    )
    if len(dataset) == 0:
        raise ValueError("The optimization split contains no normal images")
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["workers"]),
        collate_fn=collate_samples,
        pin_memory=device.type == "cuda",
    )
    anchors = load_anchors(resolve_path(config, config["materials"]["anchors"]))
    prompts = load_prompt_templates(resolve_path(config, config["materials"]["prompts"]))
    model = TriADPCB(config, anchors, prompts).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    warmup = int(config["training"]["warmup_epochs"])

    def learning_rate(epoch: int):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    name = run_name or f"default_seed{seed}"
    output = resolve_path(config, config["output"]["root"]) / name
    output.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output / "config_resolved.yaml")
    rows = []
    for epoch in range(epochs):
        model.train()
        totals: dict[str, float] = {}
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, terms = model.training_loss(
                batch["image"].to(device), int(config["training"]["patches_per_image"])
            )
            loss.backward()
            optimizer.step()
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.item())
        scheduler.step()
        row = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({key: value / max(len(loader), 1) for key, value in totals.items()})
        rows.append(row)
    if not rows:
        raise ValueError("training.epochs must be at least one")
    torch.save({"model": model.state_dict(), "seed": seed, "anchors": anchors}, output / "model.pt")
    with (output / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return output / "model.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    args = parser.parse_args()
    print(train(args.config, args.seed, args.device, args.run_name))


if __name__ == "__main__":
    main()
