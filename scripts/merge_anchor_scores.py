from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frames = []
    for source in args.inputs:
        frame = pd.read_csv(source)[["anchor", "score"]]
        frame = frame.rename(columns={"score": Path(source).stem})
        frames.append(frame)
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="anchor", validate="one_to_one")
    score_columns = [column for column in merged.columns if column != "anchor"]
    merged["mean_score"] = merged[score_columns].mean(axis=1)
    merged = merged.sort_values("mean_score", ascending=False).reset_index(drop=True)
    merged.insert(0, "rank", merged.index + 1)
    merged.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
