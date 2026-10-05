from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .config import load_yaml, resolve_path
from .data import ManifestDataset
from .evaluate import evaluate_once


def stratified_indices(
    labels: np.ndarray,
    stream_size: int,
    anomaly_ratio: float,
    seed: int,
) -> list[int]:
    if not 0.0 < anomaly_ratio < 1.0:
        raise ValueError("anomaly_ratio must be between zero and one")
    anomaly_count = int(round(stream_size * anomaly_ratio))
    normal_count = stream_size - anomaly_count
    normal = np.flatnonzero(labels == 0)
    anomalous = np.flatnonzero(labels == 1)
    if len(normal) < normal_count or len(anomalous) < anomaly_count:
        raise ValueError(
            f"The test split cannot supply {normal_count} normal and {anomaly_count} "
            "anomalous images without replacement"
        )
    rng = np.random.default_rng(seed)
    indices = np.concatenate(
        [
            rng.choice(normal, size=normal_count, replace=False),
            rng.choice(anomalous, size=anomaly_count, replace=False),
        ]
    )
    return indices.astype(int).tolist()


def run_study(
    config_path: str,
    checkpoint_path: str,
    output_dir: str | Path,
    ratios: list[float],
    stream_size: int,
    device: str | None,
) -> Path:
    config = load_yaml(config_path)
    dataset_cfg = config["dataset"]
    test = ManifestDataset(
        resolve_path(config, dataset_cfg["manifest"]),
        int(dataset_cfg["image_size"]),
        "test",
        categories=dataset_cfg.get("categories"),
    )
    labels = test.frame["label"].astype(int).to_numpy()
    composition_seeds = [int(value) for value in config["evaluation"]["support_seeds"]]
    order_seeds = [int(value) for value in config["evaluation"]["stream_seeds"]]
    output = Path(output_dir).resolve()
    rows = []
    for ratio in ratios:
        repetitions = []
        for support_seed in composition_seeds:
            selected = stratified_indices(labels, stream_size, ratio, support_seed)
            for order_seed in order_seeds:
                repetitions.append(
                    evaluate_once(
                        config_path,
                        checkpoint_path,
                        support_seed,
                        order_seed,
                        device,
                        output / "evaluations" / f"ratio_{ratio:.2f}",
                        selected,
                    )
                )
        capacity_steps = np.asarray(
            [item["step_to_capacity"] for item in repetitions], dtype=float
        )
        finite_steps = capacity_steps[np.isfinite(capacity_steps)]
        rows.append(
            {
                "anomaly_ratio_percent": 100.0 * ratio,
                "i_auc": 100.0 * np.mean([item["i_auc"] for item in repetitions]),
                "p_ap": 100.0 * np.mean([item["p_ap"] for item in repetitions]),
                "pro": 100.0 * np.mean([item["pro"] for item in repetitions]),
                "mean_step_to_capacity": (
                    float(finite_steps.mean()) if finite_steps.size else "not_reached"
                ),
                "capacity_reached_percent": 100.0 * finite_steps.size / len(repetitions),
                "anomalous_admission_rate_percent": 100.0
                * np.mean([item["anomalous_admission_rate"] for item in repetitions]),
                "stream_compositions": len(composition_seeds),
                "orders_per_composition": len(order_seeds),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sensitivity to stream anomaly ratio")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8])
    parser.add_argument("--stream-size", type=int, default=500)
    parser.add_argument("--device")
    args = parser.parse_args()
    print(
        run_study(
            args.config,
            args.checkpoint,
            args.output,
            args.ratios,
            args.stream_size,
            args.device,
        )
    )


if __name__ == "__main__":
    main()
