from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--checksums")
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    frame = pd.read_csv(manifest).fillna("")
    required = {"image_path", "mask_path", "label", "split", "category", "source_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if not set(frame["label"].astype(int)).issubset({0, 1}):
        raise ValueError("Labels must be 0 or 1")
    allowed_splits = {"optimization", "validation", "test"}
    if not set(frame["split"]).issubset(allowed_splits):
        raise ValueError(f"Splits must be in {sorted(allowed_splits)}")
    if frame["image_path"].astype(str).duplicated().any():
        raise ValueError("Each image path must occur exactly once")
    if (frame["source_id"].astype(str).str.len() == 0).any():
        raise ValueError("Every row must define a source_id")
    labels = frame["label"].astype(int)
    if ((frame["split"] != "test") & (labels == 1)).any():
        raise ValueError("Optimization and validation splits must contain only normal images")
    if ((labels == 1) & (frame["mask_path"].astype(str).str.len() == 0)).any():
        raise ValueError("Every anomalous image must have a segmentation mask")
    leakage = frame.groupby("source_id")["split"].nunique()
    leakage = leakage[leakage > 1]
    if len(leakage):
        raise ValueError(f"Source-level split leakage for {len(leakage)} source IDs")
    checksum_rows = []
    for row in frame.itertuples():
        image = Path(row.image_path)
        image = image if image.is_absolute() else (manifest.parent / image).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        checksum_rows.append({"path": str(image), "sha256": sha256(image)})
        if int(row.label):
            mask = Path(row.mask_path)
            mask = mask if mask.is_absolute() else (manifest.parent / mask).resolve()
            if not mask.is_file():
                raise FileNotFoundError(mask)
    print(frame.groupby(["split", "category", "label"]).size().to_string())
    if args.checksums:
        pd.DataFrame(checksum_rows).to_csv(args.checksums, index=False)


if __name__ == "__main__":
    main()
