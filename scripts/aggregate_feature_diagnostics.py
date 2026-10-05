from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


FEATURE_GROUPS = {
    "real_normal": "Real normal",
    "pseudo_normal": "Pseudo-normal",
    "pseudo_abnormal": "Pseudo-abnormal",
}


def aggregate_quality(inputs: list[Path], output: Path) -> None:
    frames = []
    for directory in inputs:
        path = directory / "generated_feature_quality.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["source_run"] = directory.name
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    numeric = [column for column in data.columns if column not in {"feature_group", "source_run"}]
    result = data.groupby("feature_group", sort=False)[numeric].mean().reset_index()
    result["source_runs"] = len(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)


def balanced_projection(inputs: list[Path], output: Path, count: int, seed: int) -> None:
    if count < len(inputs):
        raise ValueError("count must be at least the number of input runs")
    rng = np.random.default_rng(seed)
    allocations = np.full(len(inputs), count // len(inputs), dtype=int)
    allocations[: count % len(inputs)] += 1
    grouped: dict[str, list[np.ndarray]] = {key: [] for key in FEATURE_GROUPS}
    source_rows: dict[str, list[str]] = {key: [] for key in FEATURE_GROUPS}
    for run_index, (directory, allocation) in enumerate(zip(inputs, allocations, strict=True)):
        path = directory / "generated_features.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path) as archive:
            for key in FEATURE_GROUPS:
                values = archive[key]
                if values.shape[0] < allocation:
                    raise ValueError(
                        f"{path} contains {values.shape[0]} {key} features, "
                        f"but {allocation} are required"
                    )
                indices = rng.choice(values.shape[0], size=allocation, replace=False)
                grouped[key].append(values[indices])
                source_rows[key].extend([f"run_{run_index}"] * allocation)
    matrices = [np.concatenate(grouped[key], axis=0) for key in FEATURE_GROUPS]
    matrix = np.concatenate(matrices, axis=0)
    projection = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=min(30, max(5, matrix.shape[0] // 20)),
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(matrix)
    rows = []
    offset = 0
    for key, values in zip(FEATURE_GROUPS, matrices, strict=True):
        for point_index in range(values.shape[0]):
            rows.append(
                {
                    "feature_group": FEATURE_GROUPS[key],
                    "point_index": point_index,
                    "tsne_1": float(projection[offset + point_index, 0]),
                    "tsne_2": float(projection[offset + point_index, 1]),
                    "seed": seed,
                    "source_run": source_rows[key][point_index],
                }
            )
        offset += values.shape[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate generated-feature measurements with equal run weights"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    quality_parser = subparsers.add_parser("quality")
    quality_parser.add_argument("inputs", nargs="+", type=Path)
    quality_parser.add_argument("--output", type=Path, required=True)
    projection_parser = subparsers.add_parser("projection")
    projection_parser.add_argument("inputs", nargs="+", type=Path)
    projection_parser.add_argument("--output", type=Path, required=True)
    projection_parser.add_argument("--count", type=int, default=500)
    projection_parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.command == "quality":
        aggregate_quality(args.inputs, args.output)
    else:
        balanced_projection(args.inputs, args.output, args.count, args.seed)
    print(args.output)


if __name__ == "__main__":
    main()
