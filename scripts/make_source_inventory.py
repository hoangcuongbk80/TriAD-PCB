from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a source-level table before crop extraction")
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--category", default="pcb")
    args = parser.parse_args()
    root = Path(args.images).resolve()
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    rows = [
        {"source_id": path.stem, "category": args.category, "image_path": str(path)}
        for path in sorted(root.rglob("*"))
        if path.suffix.lower() in suffixes
    ]
    frame = pd.DataFrame(rows)
    if frame["source_id"].duplicated().any():
        duplicates = frame.loc[frame["source_id"].duplicated(), "source_id"].tolist()
        raise ValueError(f"Source stems must be unique: {duplicates[:10]}")
    frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
