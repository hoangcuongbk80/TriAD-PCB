from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expected", default="materials/dataset_inventory_expected.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest).fillna("")
    expected_all = json.loads(Path(args.expected).read_text(encoding="utf-8"))
    if args.dataset not in expected_all:
        raise ValueError(f"No expected inventory for {args.dataset}")
    expected = expected_all[args.dataset]
    label = frame["label"].astype(int)
    split = frame["split"].astype(str)
    actual = {
        "total_images": int(len(frame)),
        "normal_images": int((label == 0).sum()),
        "anomalous_images": int((label == 1).sum()),
        "optimization_normal": int(((split == "optimization") & (label == 0)).sum()),
        "validation_normal": int(((split == "validation") & (label == 0)).sum()),
        "test_normal": int(((split == "test") & (label == 0)).sum()),
        "test_anomalous": int(((split == "test") & (label == 1)).sum()),
        "anomalous_test_masks": int(
            ((split == "test") & (label == 1) & frame["mask_path"].astype(bool)).sum()
        ),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": value}
        for key, value in actual.items()
        if value != expected[key]
    }
    print(json.dumps({"dataset": args.dataset, "actual": actual, "mismatches": mismatches}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
