from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and package released evaluation masks")
    parser.add_argument("--mask-root", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--license", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-dspcbsd", type=int, default=450)
    parser.add_argument("--expected-tdd", type=int, default=1006)
    args = parser.parse_args()
    root = Path(args.mask_root).resolve()
    metadata = Path(args.metadata).resolve()
    license_path = Path(args.license).resolve()
    rows = list(csv.DictReader(metadata.open(encoding="utf-8")))
    required = {"dataset", "image_id", "mask_path", "annotator_count", "review_status"}
    if not rows or required.difference(rows[0]):
        raise ValueError(f"Metadata must contain columns: {sorted(required)}")
    expected = {"DsPCBSD+": args.expected_dspcbsd, "TDD-Net PCB": args.expected_tdd}
    counts = {dataset: 0 for dataset in expected}
    files = []
    for row in rows:
        if row["dataset"] not in expected:
            raise ValueError(f"Unexpected dataset: {row['dataset']}")
        mask = (root / row["mask_path"]).resolve()
        if root not in mask.parents:
            raise ValueError(f"Mask path escapes the release root: {mask}")
        if not mask.is_file():
            raise FileNotFoundError(mask)
        with Image.open(mask) as image:
            if image.convert("L").getextrema()[1] == 0:
                raise ValueError(f"An anomalous-image mask is empty: {mask}")
        counts[row["dataset"]] += 1
        files.append((mask, row["mask_path"], sha256(mask)))
    if counts != expected:
        raise ValueError(f"Mask counts {counts} do not match expected counts {expected}")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checksum_text = "\n".join(f"{digest}  {relative}" for _, relative, digest in files) + "\n"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(metadata, "metadata.csv")
        archive.write(license_path, "LICENSE.txt")
        archive.writestr("SHA256SUMS", checksum_text)
        archive.writestr("release_summary.json", json.dumps({"counts": counts}, indent=2))
        for mask, relative, _ in files:
            archive.write(mask, f"masks/{relative}")
    print(json.dumps({"archive": str(output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
