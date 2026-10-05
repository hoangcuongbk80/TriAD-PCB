from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .config import load_yaml, resolve_path
from .data import ManifestDataset
from .engine import CausalInferenceSession
from .metrics import evaluate_predictions
from .model import TriADPCB
from .prompts import load_anchors, load_prompt_templates
from .utils import device_from_arg, save_json, seed_everything


def _support_indices(dataset: ManifestDataset, support_size: int, seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    result: dict[str, list[int]] = {}
    for category, rows in dataset.frame.groupby("category"):
        candidates = rows.index.to_numpy()
        if len(candidates) < support_size:
            raise ValueError(f"Category {category} has only {len(candidates)} normal optimization images")
        result[str(category)] = rng.choice(candidates, size=support_size, replace=False).tolist()
    return result


def _protocol_records(
    optimization: ManifestDataset,
    test: ManifestDataset,
    support: dict[str, list[int]],
    order: list[int],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    support_records = []
    for category, indices in support.items():
        for support_position, index in enumerate(indices):
            row = optimization.frame.iloc[index]
            support_records.append(
                {
                    "category": category,
                    "support_position": support_position,
                    "dataset_index": int(index),
                    "image_path": str(row.image_path),
                    "source_id": str(row.source_id),
                }
            )
    stream_records = []
    for stream_position, index in enumerate(order):
        row = test.frame.iloc[index]
        stream_records.append(
            {
                "stream_position": stream_position,
                "dataset_index": int(index),
                "image_path": str(row.image_path),
                "source_id": str(row.source_id),
                "category": str(row.category),
                "label": int(row.label),
            }
        )
    support_sources = {row["source_id"] for row in support_records}
    test_sources = {row["source_id"] for row in stream_records}
    overlap = support_sources.intersection(test_sources)
    if overlap:
        raise ValueError(f"Support/test source leakage detected: {sorted(overlap)[:10]}")
    support_paths = {row["image_path"] for row in support_records}
    test_paths = {row["image_path"] for row in stream_records}
    if support_paths.intersection(test_paths):
        raise ValueError("One or more support images also occur in the test stream")
    return support_records, stream_records


def evaluate_once(
    config_path: str,
    checkpoint_path: str,
    support_seed: int,
    stream_seed: int,
    device_arg: str | None = None,
    output_dir: str | Path | None = None,
    stream_indices: list[int] | None = None,
) -> dict[str, float]:
    config = load_yaml(config_path)
    seed_everything(stream_seed)
    device = device_from_arg(device_arg)
    dataset_cfg = config["dataset"]
    manifest = resolve_path(config, dataset_cfg["manifest"])
    optimization = ManifestDataset(
        manifest,
        int(dataset_cfg["image_size"]),
        "optimization",
        normal_only=True,
        categories=dataset_cfg.get("categories"),
    )
    test = ManifestDataset(
        manifest,
        int(dataset_cfg["image_size"]),
        "test",
        categories=dataset_cfg.get("categories"),
    )
    if len(optimization) == 0:
        raise ValueError("The optimization split contains no normal support candidates")
    if len(test) == 0:
        raise ValueError("The test split is empty")
    anchors = load_anchors(resolve_path(config, config["materials"]["anchors"]))
    prompts = load_prompt_templates(resolve_path(config, config["materials"]["prompts"]))
    model = TriADPCB(config, anchors, prompts).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    support = _support_indices(optimization, int(config["evaluation"]["support_size"]), support_seed)
    sessions: dict[str, CausalInferenceSession] = {}
    for category, indices in support.items():
        session = CausalInferenceSession(model, config)
        images = torch.stack([optimization[index]["image"] for index in indices]).to(device)
        session.initialize_support(images)
        sessions[category] = session
    rng = np.random.default_rng(stream_seed)
    if stream_indices is None:
        order = rng.permutation(len(test)).tolist()
    else:
        if not stream_indices:
            raise ValueError("The requested test stream is empty")
        if len(set(stream_indices)) != len(stream_indices):
            raise ValueError("The requested test stream contains duplicate indices")
        if min(stream_indices) < 0 or max(stream_indices) >= len(test):
            raise IndexError("A requested test index is outside the test split")
        order = rng.permutation(np.asarray(stream_indices, dtype=int)).tolist()
    support_records, stream_records = _protocol_records(optimization, test, support, order)
    labels, scores, masks, maps, rows = [], [], [], [], []
    for sequence_index, dataset_index in enumerate(order):
        sample = test[dataset_index]
        category = str(sample["category"])
        if category not in sessions:
            raise ValueError(f"No support set was constructed for category {category}")
        result = sessions[category].step(sample["image"].unsqueeze(0).to(device))
        labels.append(int(sample["label"]))
        scores.append(float(result["image_score"]))
        masks.append(sample["mask"].numpy())
        maps.append(result["pixel_map"].cpu().numpy())
        rows.append(
            {
                "sequence_index": sequence_index,
                "dataset_index": dataset_index,
                "image_path": sample["image_path"],
                "category": category,
                "label": sample["label"],
                "image_score": result["image_score"],
                "accepted_into_history": result["accepted_into_history"],
                "history_size_after_update": result["history_size_after_update"],
                "lambda_text": float(result["fusion_weights"][0].item()),
                "lambda_history": float(result["fusion_weights"][1].item()),
                "lambda_reference": float(result["fusion_weights"][2].item()),
            }
        )
    metrics = evaluate_predictions(
        np.asarray(labels),
        np.asarray(scores),
        np.stack(masks),
        np.stack(maps),
        thresholds=int(config["evaluation"]["thresholds"]),
    )
    metrics.update({"support_seed": support_seed, "stream_seed": stream_seed})
    capacity = int(config["memory"]["history_capacity"])
    capacity_steps = [
        int(row["sequence_index"]) + 1
        for row in rows
        if int(row["history_size_after_update"]) >= capacity
    ]
    metrics["step_to_capacity"] = float(capacity_steps[0]) if capacity_steps else float("nan")
    anomalous_rows = [row for row in rows if int(row["label"]) == 1]
    metrics["anomalous_admission_rate"] = float(
        np.mean([bool(row["accepted_into_history"]) for row in anomalous_rows])
    ) if anomalous_rows else 0.0
    if output_dir:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        stem = f"support{support_seed}_stream{stream_seed}"
        save_json(
            {
                "metrics": metrics,
                "support_seed": support_seed,
                "stream_seed": stream_seed,
                "support": support_records,
                "stream": stream_records,
                "audit": {
                    "support_test_image_overlap": 0,
                    "support_test_source_overlap": 0,
                    "causal_update_order": "score_gate_update",
                },
            },
            output / f"{stem}.json",
        )
        with (output / f"{stem}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--support-seed", type=int)
    parser.add_argument("--stream-seed", type=int)
    parser.add_argument("--all-repetitions", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.all_repetitions and (args.support_seed is not None or args.stream_seed is not None):
        parser.error("--all-repetitions cannot be combined with an explicit seed")
    support_seeds = config["evaluation"]["support_seeds"] if args.all_repetitions else [args.support_seed or config["evaluation"]["support_seeds"][0]]
    stream_seeds = config["evaluation"]["stream_seeds"] if args.all_repetitions else [args.stream_seed or config["evaluation"]["stream_seeds"][0]]
    all_metrics = []
    for support_seed in support_seeds:
        for stream_seed in stream_seeds:
            all_metrics.append(
                evaluate_once(
                    args.config,
                    args.checkpoint,
                    int(support_seed),
                    int(stream_seed),
                    args.device,
                    args.output,
                )
            )
    if args.output:
        numeric_keys = [key for key in all_metrics[0] if key not in {"support_seed", "stream_seed", "i_threshold", "p_threshold"}]
        aggregate = {key: float(np.mean([row[key] for row in all_metrics])) for key in numeric_keys}
        aggregate["repetitions"] = len(all_metrics)
        save_json(aggregate, Path(args.output) / "aggregate.json")
    print(all_metrics[-1])


if __name__ == "__main__":
    main()
