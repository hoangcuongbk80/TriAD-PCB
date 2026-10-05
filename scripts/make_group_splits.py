from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--optimization", type=float, default=0.7)
    parser.add_argument("--validation", type=float, default=0.1)
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest).fillna("")
    rng = np.random.default_rng(args.seed)
    assignments = {}
    for category, rows in frame.groupby("category"):
        source_labels = rows.groupby("source_id")["label"].max().astype(int)
        anomalous_groups = source_labels.index[source_labels == 1].to_numpy()
        normal_groups = source_labels.index[source_labels == 0].to_numpy().copy()
        rng.shuffle(normal_groups)
        n_opt = round(args.optimization * len(normal_groups))
        n_val = round(args.validation * len(normal_groups))
        for group in anomalous_groups:
            assignments[(category, group)] = "test"
        for group in normal_groups[:n_opt]:
            assignments[(category, group)] = "optimization"
        for group in normal_groups[n_opt : n_opt + n_val]:
            assignments[(category, group)] = "validation"
        for group in normal_groups[n_opt + n_val :]:
            assignments[(category, group)] = "test"
    frame["split"] = [assignments[(row.category, row.source_id)] for row in frame.itertuples()]
    frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
