from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def first_present(frame: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"None of the expected columns are present: {names}")


def parse_validation_counts(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Validation counts must use category=count syntax")
        category, count = value.split("=", maxsplit=1)
        result[category] = int(count)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an official VisA split CSV to TriAD-PCB format")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--categories", nargs="+", default=["pcb1", "pcb2", "pcb3", "pcb4"])
    parser.add_argument(
        "--validation-counts",
        nargs="+",
        default=["pcb1=89", "pcb2=89", "pcb3=86", "pcb4=86"],
        help="Held-out normal images per category, using category=count entries",
    )
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    validation_counts = parse_validation_counts(args.validation_counts)
    missing_counts = set(args.categories).difference(validation_counts)
    if missing_counts:
        raise ValueError(f"Validation counts are missing for: {sorted(missing_counts)}")
    root = Path(args.dataset_root).resolve()
    source = pd.read_csv(args.split_csv).fillna("")
    category_col = first_present(source, ["object", "category", "class"])
    image_col = first_present(source, ["image", "image_path", "img_path"])
    mask_col = next((name for name in ["mask", "mask_path"] if name in source.columns), None)
    label_col = first_present(source, ["label", "anomaly", "is_anomaly"])
    split_col = first_present(source, ["split", "set"])
    source = source[source[category_col].isin(args.categories)]
    rows = []
    for item in source.itertuples(index=False):
        values = item._asdict()
        raw_label = str(values[label_col]).lower()
        label = int(raw_label not in {"0", "normal", "good", "false"})
        source_split = str(values[split_col]).lower()
        split = "test" if source_split == "test" else "training_pool"
        if split != "test" and label:
            raise ValueError("The VisA training pool is expected to contain only normal images")
        image = (root / str(values[image_col])).resolve()
        mask_value = str(values[mask_col]) if mask_col else ""
        mask = str((root / mask_value).resolve()) if mask_value else ""
        rows.append(
            {
                "image_path": str(image),
                "mask_path": mask,
                "label": label,
                "split": split,
                "category": str(values[category_col]),
                "source_id": f"{values[category_col]}/{image.stem}",
            }
        )
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(args.seed)
    for category in args.categories:
        candidates = frame.index[
            (frame["category"] == category) & (frame["split"] == "training_pool")
        ].to_numpy()
        count = validation_counts[category]
        if len(candidates) <= count:
            raise ValueError(
                f"Category {category} has {len(candidates)} training images, "
                f"which is insufficient for {count} validation images"
            )
        selected = rng.choice(candidates, size=count, replace=False)
        frame.loc[candidates, "split"] = "optimization"
        frame.loc[selected, "split"] = "validation"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


if __name__ == "__main__":
    main()
