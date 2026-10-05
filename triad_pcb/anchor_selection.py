from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .backbone import FrozenCLIPEncoder
from .config import load_yaml, resolve_path
from .data import ManifestDataset, collate_samples
from .prompts import load_prompt_templates
from .utils import device_from_arg, l2_normalize


@torch.no_grad()
def rank_anchors(config_path: str, candidates_path: str | None, output_path: str, device_arg: str | None):
    config = load_yaml(config_path)
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
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_samples, num_workers=0)
    encoder = FrozenCLIPEncoder(config["model"]["model_name"], freeze=True).to(device)
    globals_ = []
    for batch in loader:
        _, global_features = encoder.encode_image(batch["image"].to(device))
        globals_.append(global_features)
    image_features = l2_normalize(torch.cat(globals_))
    candidate_file = resolve_path(config, candidates_path or "materials/candidate_anchors_16.txt")
    candidates = [line.strip() for line in candidate_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not candidates:
        raise ValueError("The candidate anchor list is empty")
    templates = load_prompt_templates(resolve_path(config, config["materials"]["prompts"]))["anchor_seed"]
    rows = []
    for candidate in candidates:
        texts = [template.format(anchor=candidate) for template in templates]
        embedding = l2_normalize(encoder.encode_text(texts, device=device).mean(dim=0))
        score = (image_features @ embedding).mean()
        rows.append({"anchor": candidate, "score": float(score.item())})
    rows.sort(key=lambda item: item["score"], reverse=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "anchor", "score"])
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow({"rank": rank, **row})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates")
    parser.add_argument("--output", default="anchor_scores.csv")
    parser.add_argument("--device")
    args = parser.parse_args()
    rank_anchors(args.config, args.candidates, args.output, args.device)


if __name__ == "__main__":
    main()
