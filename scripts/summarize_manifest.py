from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    frame = pd.read_csv(manifest).fillna("")
    required = {"image_path", "mask_path", "label", "split", "category", "source_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    resolutions = {}
    for value in frame["image_path"].drop_duplicates():
        path = Path(value)
        path = path if path.is_absolute() else (manifest.parent / path).resolve()
        with Image.open(path) as image:
            key = f"{image.width}x{image.height}"
            resolutions[key] = resolutions.get(key, 0) + 1
    grouped = (
        frame.groupby(["category", "split", "label"], dropna=False)
        .agg(images=("image_path", "count"), sources=("source_id", "nunique"))
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "manifest": str(manifest),
        "total_images": int(len(frame)),
        "normal_images": int((frame["label"].astype(int) == 0).sum()),
        "anomalous_images": int((frame["label"].astype(int) == 1).sum()),
        "anomalous_images_with_masks": int(
            ((frame["label"].astype(int) == 1) & frame["mask_path"].astype(bool)).sum()
        ),
        "categories": sorted(frame["category"].astype(str).unique().tolist()),
        "resolutions": resolutions,
        "partitions": grouped,
    }
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
