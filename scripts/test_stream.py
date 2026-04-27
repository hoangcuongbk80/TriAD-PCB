from __future__ import annotations
import argparse
from pathlib import Path
import json

import torch

from src.models.triad_pcb import TriADPCB
from src.data.datasets import ReferenceDataset, StreamDataset

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stream-root", required=True)
    p.add_argument("--reference-root", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out", default="outputs/predictions.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriADPCB().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state["model"], strict=False)

    ref_ds = ReferenceDataset(args.reference_root)
    stream_ds = StreamDataset(args.stream_root)

    ref_images = torch.stack([ref_ds[i]["image"] for i in range(len(ref_ds))], dim=0).to(device)
    model.build_reference_bank(ref_images)
    model.reset_stream_state()

    results = []
    for i in range(len(stream_ds)):
        sample = stream_ds[i]
        img = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img)
        results.append({
            "path": sample["path"],
            "score": float(out["image_score"][0].detach().cpu().item()),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
