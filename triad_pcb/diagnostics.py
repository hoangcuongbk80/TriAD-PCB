from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from .config import load_yaml, resolve_path
from .data import ManifestDataset
from .model import TriADPCB
from .prompts import load_anchors, load_prompt_templates
from .utils import device_from_arg, l2_normalize, seed_everything


def _load_model(config: dict, checkpoint_path: str, device: torch.device) -> TriADPCB:
    anchors = load_anchors(resolve_path(config, config["materials"]["anchors"]))
    templates = load_prompt_templates(resolve_path(config, config["materials"]["prompts"]))
    model = TriADPCB(config, anchors, templates).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def rbf_mmd(x: torch.Tensor, y: torch.Tensor, chunk_size: int = 1024) -> float:
    """Biased RBF MMD with a pooled median-distance bandwidth."""
    count = min(x.shape[0], y.shape[0])
    x = l2_normalize(x[:count].float())
    y = l2_normalize(y[:count].float())
    pooled = torch.cat([x, y], dim=0)
    distances = torch.pdist(pooled).square()
    positive = distances[distances > 0]
    bandwidth = positive.median() if positive.numel() else torch.tensor(1.0)
    gamma = 1.0 / bandwidth.clamp_min(1e-8)

    def kernel_mean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), device=a.device, dtype=a.dtype)
        elements = 0
        for start in range(0, a.shape[0], chunk_size):
            values = torch.exp(-gamma * torch.cdist(a[start : start + chunk_size], b).square())
            total = total + values.sum()
            elements += values.numel()
        return total / elements

    value = kernel_mean(x, x) + kernel_mean(y, y) - 2.0 * kernel_mean(x, y)
    return float(value.clamp_min(0).item())


def _feature_row(
    name: str,
    features: torch.Tensor,
    real_features: torch.Tensor,
    normal_targets: torch.Tensor,
    abnormal_targets: torch.Tensor,
    prompt_bank: torch.Tensor,
    target_indices: torch.Tensor,
    abnormal_direction: bool,
) -> dict[str, float | str]:
    features = l2_normalize(features.float())
    normal_similarity = (features * normal_targets).sum(dim=-1)
    abnormal_similarity = (features * abnormal_targets).sum(dim=-1)
    if abnormal_direction:
        margin = abnormal_similarity - normal_similarity
    else:
        margin = normal_similarity - abnormal_similarity
    nearest = features @ prompt_bank.T
    agreement = (nearest.argmax(dim=1) == target_indices).float().mean() * 100.0
    return {
        "feature_group": name,
        "cosine_to_normal_prompt": float(normal_similarity.mean().item()),
        "cosine_to_matched_abnormal_prompt": float(abnormal_similarity.mean().item()),
        "semantic_margin": float(margin.mean().item()),
        "nearest_prompt_agreement_percent": float(agreement.item()),
        "mmd_to_real_normal": 0.0 if name == "Real normal features" else rbf_mmd(real_features, features),
    }


@torch.no_grad()
def export_generated_feature_diagnostics(
    config_path: str,
    checkpoint_path: str,
    output_dir: str | Path,
    sample_count: int,
    visualization_count: int,
    seed: int,
    device_arg: str | None,
) -> None:
    if sample_count < 1 or visualization_count < 1:
        raise ValueError("sample counts must be positive")
    config = load_yaml(config_path)
    seed_everything(seed)
    device = device_from_arg(device_arg)
    dataset_cfg = config["dataset"]
    dataset = ManifestDataset(
        resolve_path(config, dataset_cfg["manifest"]),
        int(dataset_cfg["image_size"]),
        "optimization",
        normal_only=True,
        categories=dataset_cfg.get("categories"),
    )
    if len(dataset) == 0:
        raise ValueError("The optimization split contains no normal images")
    model = _load_model(config, checkpoint_path, device)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    per_image = max(1, int(np.ceil(sample_count / min(len(dataset), 128))))
    real_parts: list[torch.Tensor] = []
    anchor_parts: list[torch.Tensor] = []
    abnormal_parts: list[torch.Tensor] = []
    collected = 0
    for dataset_index in indices:
        sample = dataset[int(dataset_index)]
        patches, global_features = model.encode_image(sample["image"].unsqueeze(0).to(device))
        patch_count = min(per_image, patches.shape[1], sample_count - collected)
        selected = rng.choice(patches.shape[1], size=patch_count, replace=False)
        real_parts.append(patches[0, torch.as_tensor(selected, device=device)])
        anchor = model.prompt_bank.anchor_weights(
            global_features, float(config["model"]["anchor_temperature"])
        ).argmax(dim=1)
        anchor_parts.append(anchor.expand(patch_count))
        abnormal_parts.append(
            torch.as_tensor(
                rng.integers(0, model.prompt_bank.abnormal_seed.shape[1], size=patch_count),
                device=device,
            )
        )
        collected += patch_count
        if collected >= sample_count:
            break

    real = torch.cat(real_parts, dim=0)[:sample_count]
    anchor_index = torch.cat(anchor_parts, dim=0)[:sample_count]
    abnormal_index = torch.cat(abnormal_parts, dim=0)[:sample_count]
    _, normal_prompts, abnormal_prompts = model.prompt_bank.normalized()
    normal_targets = normal_prompts[anchor_index]
    abnormal_targets = abnormal_prompts[anchor_index, abnormal_index]
    pseudo_normal = model.generator(real, normal_targets)
    pseudo_abnormal = model.generator(real, abnormal_targets)
    unconditioned = model.generator(real, torch.zeros_like(real))

    all_prompts = torch.cat([normal_prompts, abnormal_prompts.flatten(0, 1)], dim=0)
    abnormal_target_index = (
        normal_prompts.shape[0]
        + anchor_index * abnormal_prompts.shape[1]
        + abnormal_index
    )
    rows = [
        _feature_row(
            "Real normal features", real, real, normal_targets, abnormal_targets,
            all_prompts, anchor_index, False,
        ),
        _feature_row(
            "Pseudo-normal features", pseudo_normal, real, normal_targets, abnormal_targets,
            all_prompts, anchor_index, False,
        ),
        _feature_row(
            "Pseudo-abnormal features", pseudo_abnormal, real, normal_targets, abnormal_targets,
            all_prompts, abnormal_target_index, True,
        ),
        _feature_row(
            "Unconditioned generator outputs", unconditioned, real, normal_targets,
            abnormal_targets, all_prompts, anchor_index, False,
        ),
    ]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "generated_feature_quality.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output / "generated_features.npz",
        real_normal=real.cpu().numpy(),
        pseudo_normal=pseudo_normal.cpu().numpy(),
        pseudo_abnormal=pseudo_abnormal.cpu().numpy(),
        anchor_index=anchor_index.cpu().numpy(),
        abnormal_index=abnormal_index.cpu().numpy(),
    )

    per_group = min(visualization_count, real.shape[0])
    chosen = rng.choice(real.shape[0], size=per_group, replace=False)
    groups = {
        "Real normal": real[chosen],
        "Pseudo-normal": pseudo_normal[chosen],
        "Pseudo-abnormal": pseudo_abnormal[chosen],
    }
    matrix = torch.cat(list(groups.values()), dim=0).cpu().numpy()
    projection = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=min(30, max(5, matrix.shape[0] // 20)),
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(matrix)
    projection_rows = []
    offset = 0
    for group, values in groups.items():
        for point_index in range(values.shape[0]):
            projection_rows.append(
                {
                    "feature_group": group,
                    "point_index": point_index,
                    "tsne_1": float(projection[offset + point_index, 0]),
                    "tsne_2": float(projection[offset + point_index, 1]),
                    "seed": seed,
                }
            )
        offset += values.shape[0]
    pd.DataFrame(projection_rows).to_csv(output / "generated_feature_projection.csv", index=False)


def aggregate_fusion_trajectory(
    inputs: list[str], output_path: str | Path, max_steps: int = 100
) -> None:
    paths: list[Path] = []
    for value in inputs:
        candidate = Path(value)
        if candidate.is_dir():
            paths.extend(candidate.rglob("*_predictions.csv"))
        elif candidate.is_file():
            paths.append(candidate)
    if not paths:
        raise FileNotFoundError("No prediction CSV files were found")
    frames = []
    for run_index, path in enumerate(sorted(set(paths))):
        frame = pd.read_csv(path)
        required = {
            "sequence_index", "history_size_after_update", "lambda_text",
            "lambda_history", "lambda_reference",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame[frame["sequence_index"] < max_steps].copy()
        frame["run_index"] = run_index
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    grouped = data.groupby("sequence_index", sort=True)
    result = grouped.agg(
        mean_history_bank_size=("history_size_after_update", "mean"),
        lambda_text=("lambda_text", "mean"),
        std_lambda_text=("lambda_text", "std"),
        lambda_history=("lambda_history", "mean"),
        std_lambda_history=("lambda_history", "std"),
        lambda_reference=("lambda_reference", "mean"),
        std_lambda_reference=("lambda_reference", "std"),
    ).fillna(0.0)
    result.index.name = "stream_step"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.reset_index().to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TriAD-PCB diagnostic artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    feature_parser = subparsers.add_parser("features")
    feature_parser.add_argument("--config", required=True)
    feature_parser.add_argument("--checkpoint", required=True)
    feature_parser.add_argument("--output", required=True)
    feature_parser.add_argument("--sample-count", type=int, default=10000)
    feature_parser.add_argument("--visualization-count", type=int, default=500)
    feature_parser.add_argument("--seed", type=int, default=3407)
    feature_parser.add_argument("--device")
    fusion_parser = subparsers.add_parser("fusion")
    fusion_parser.add_argument("inputs", nargs="+")
    fusion_parser.add_argument("--output", required=True)
    fusion_parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()
    if args.command == "features":
        export_generated_feature_diagnostics(
            args.config,
            args.checkpoint,
            args.output,
            args.sample_count,
            args.visualization_count,
            args.seed,
            args.device,
        )
    else:
        aggregate_fusion_trajectory(args.inputs, args.output, args.max_steps)


if __name__ == "__main__":
    main()
