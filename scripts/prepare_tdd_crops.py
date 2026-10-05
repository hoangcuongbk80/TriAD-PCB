from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def boxes_from_voc(xml_path: Path) -> list[tuple[int, int, int, int]]:
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is not None:
            boxes.append(
                tuple(int(float(box.find(name).text)) for name in ["xmin", "ymin", "xmax", "ymax"])
            )
    return boxes


def expanded_box_mask(
    size: tuple[int, int], boxes: list[tuple[int, int, int, int]], buffer_pixels: int
) -> Image.Image:
    mask = Image.new("L", size)
    draw = ImageDraw.Draw(mask)
    width, height = size
    for xmin, ymin, xmax, ymax in boxes:
        draw.rectangle(
            (
                max(0, xmin - buffer_pixels),
                max(0, ymin - buffer_pixels),
                min(width - 1, xmax + buffer_pixels),
                min(height - 1, ymax + buffer_pixels),
            ),
            fill=255,
        )
    return mask


def find_mask(mask_root: Path, stem: str) -> Path | None:
    matches = [path for path in mask_root.rglob(f"{stem}.*") if path.suffix.lower() in IMAGE_SUFFIXES]
    if len(matches) > 1:
        raise ValueError(f"Multiple masks found for source {stem}: {matches}")
    return matches[0] if matches else None


def load_source_splits(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    required = {"source_id", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Source split file is missing columns: {sorted(missing)}")
    if frame["source_id"].duplicated().any():
        raise ValueError("Each source_id must occur exactly once in the source split file")
    allowed = {"optimization", "validation", "test"}
    if not set(frame["split"]).issubset(allowed):
        raise ValueError(f"Source splits must be in {sorted(allowed)}")
    return dict(zip(frame["source_id"].astype(str), frame["split"].astype(str)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create TDD-Net PCB crops after assigning source-level partitions"
    )
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", required=True, help="Pascal VOC bounding boxes")
    parser.add_argument("--masks", required=True, help="Pixel masks used for anomalous evaluation crops")
    parser.add_argument("--source-splits", required=True, help="CSV with source_id,split")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--exclusion-buffer", type=int, default=16)
    args = parser.parse_args()

    image_root = Path(args.images).resolve()
    annotation_root = Path(args.annotations).resolve()
    mask_root = Path(args.masks).resolve()
    source_splits = load_source_splits(Path(args.source_splits).resolve())
    output = Path(args.output_root).resolve()
    image_output, mask_output = output / "images", output / "masks"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)
    rows, excluded_near_defect = [], 0

    image_paths = sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in image_paths:
        source_id = image_path.stem
        if source_id not in source_splits:
            raise ValueError(f"No source-level split is defined for {source_id}")
        image = Image.open(image_path).convert("RGB")
        xml_path = annotation_root / f"{source_id}.xml"
        boxes = boxes_from_voc(xml_path) if xml_path.exists() else []
        expanded = expanded_box_mask(image.size, boxes, args.exclusion_buffer)
        mask_path = find_mask(mask_root, source_id)
        if boxes and mask_path is None:
            raise FileNotFoundError(f"Annotated source {source_id} has no pixel mask")
        full_mask = Image.open(mask_path).convert("L") if mask_path else Image.new("L", image.size)
        if full_mask.size != image.size:
            raise ValueError(f"Image and mask sizes differ for source {source_id}")

        width, height = image.size
        if width < args.crop_size or height < args.crop_size:
            continue
        # These limits deliberately discard incomplete right and bottom boundary crops.
        for top in range(0, height - args.crop_size + 1, args.stride):
            for left in range(0, width - args.crop_size + 1, args.stride):
                rectangle = (left, top, left + args.crop_size, top + args.crop_size)
                crop = image.crop(rectangle)
                mask = full_mask.crop(rectangle)
                label = int(np.asarray(mask).max() > 0)
                if not label and np.asarray(expanded.crop(rectangle)).max() > 0:
                    excluded_near_defect += 1
                    continue
                stem = f"{source_id}_y{top}_x{left}"
                crop_path = image_output / f"{stem}.png"
                crop_mask_path = mask_output / f"{stem}.png"
                crop.save(crop_path)
                if label:
                    mask.save(crop_mask_path)
                rows.append(
                    {
                        "image_path": str(crop_path),
                        "mask_path": str(crop_mask_path) if label else "",
                        "label": label,
                        "split": source_splits[source_id],
                        "category": "pcb",
                        "source_id": source_id,
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(args.manifest, index=False)
    print(f"wrote={len(frame)} excluded_near_defect={excluded_near_defect}")


if __name__ == "__main__":
    main()
