from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import load_yaml, resolve_path
from .data import ManifestDataset
from .engine import CausalInferenceSession
from .evaluate import _support_indices
from .metrics import evaluate_predictions
from .model import TriADPCB
from .prompts import load_anchors, load_prompt_templates
from .utils import device_from_arg, save_json, seed_everything


def localized_validation_anomaly(image: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Apply one deterministic local corruption to a held-out normal image."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    _, height, width = image.shape
    coarse = torch.rand((1, 1, 8, 8), generator=generator)
    field = F.interpolate(coarse, size=(height, width), mode="bicubic", align_corners=False)[0, 0]
    threshold = torch.quantile(field, 0.78)
    mask = (field >= threshold).to(image.dtype)
    mode = seed % 3
    result = image.clone()
    if mode == 0:
        noise = torch.randn(image.shape, generator=generator, dtype=image.dtype) * 0.35
        altered = image + noise
        name = "local_texture_noise"
    elif mode == 1:
        gain = torch.tensor([1.35, 0.70, 1.15], dtype=image.dtype)[:, None, None]
        offset = torch.tensor([0.20, -0.15, 0.10], dtype=image.dtype)[:, None, None]
        altered = image * gain + offset
        name = "local_chromatic_shift"
    else:
        shift_y = max(1, height // 7)
        shift_x = max(1, width // 9)
        altered = torch.roll(image, shifts=(shift_y, shift_x), dims=(1, 2))
        name = "local_structure_displacement"
    result = image * (1.0 - mask[None]) + altered * mask[None]
    return result, mask, name


def _load_model(config: dict, checkpoint_path: str, device: torch.device) -> TriADPCB:
    anchors = load_anchors(resolve_path(config, config["materials"]["anchors"]))
    prompts = load_prompt_templates(resolve_path(config, config["materials"]["prompts"]))
    model = TriADPCB(config, anchors, prompts).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def evaluate_validation(
    config_path: str,
    checkpoint_path: str,
    seed: int,
    device_arg: str | None = None,
    output_path: str | Path | None = None,
) -> dict[str, float]:
    config = load_yaml(config_path)
    seed_everything(seed)
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
    validation = ManifestDataset(
        manifest,
        int(dataset_cfg["image_size"]),
        "validation",
        normal_only=True,
        categories=dataset_cfg.get("categories"),
    )
    if len(optimization) == 0:
        raise ValueError("The optimization split contains no normal support candidates")
    if len(validation) == 0:
        raise ValueError("The validation split contains no normal images")
    model = _load_model(config, checkpoint_path, device)
    support = _support_indices(optimization, int(config["evaluation"]["support_size"]), seed)
    sessions = {}
    for category, indices in support.items():
        session = CausalInferenceSession(model, config)
        session.initialize_support(
            torch.stack([optimization[index]["image"] for index in indices]).to(device)
        )
        sessions[category] = session

    examples = []
    for index in range(len(validation)):
        sample = validation[index]
        examples.append(
            {
                "image": sample["image"],
                "mask": torch.zeros_like(sample["mask"]),
                "label": 0,
                "category": sample["category"],
                "source_index": index,
                "corruption": "none",
            }
        )
        anomalous, mask, corruption = localized_validation_anomaly(
            sample["image"], seed * 1_000_003 + index
        )
        examples.append(
            {
                "image": anomalous,
                "mask": mask,
                "label": 1,
                "category": sample["category"],
                "source_index": index,
                "corruption": corruption,
            }
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(examples)).tolist()
    labels, scores, masks, maps, records = [], [], [], [], []
    for position, example_index in enumerate(order):
        example = examples[example_index]
        result = sessions[str(example["category"])].step(example["image"].unsqueeze(0).to(device))
        labels.append(example["label"])
        scores.append(result["image_score"])
        masks.append(example["mask"].numpy())
        maps.append(result["pixel_map"].cpu().numpy())
        records.append(
            {
                "stream_position": position,
                "source_index": example["source_index"],
                "label": example["label"],
                "category": example["category"],
                "corruption": example["corruption"],
                "image_score": result["image_score"],
            }
        )
    metrics = evaluate_predictions(
        np.asarray(labels),
        np.asarray(scores),
        np.stack(masks),
        np.stack(maps),
        thresholds=int(config["evaluation"]["thresholds"]),
    )
    clean_scores = [record["image_score"] for record in records if record["label"] == 0]
    metrics["clean_mean_score"] = float(np.mean(clean_scores))
    metrics["selection_score"] = 0.5 * (metrics["i_auc"] + metrics["p_auc"])
    metrics["validation_seed"] = seed
    if output_path:
        save_json(
            {
                "metrics": metrics,
                "support_indices": support,
                "validation_order": order,
                "records": records,
                "selection_rule": "maximize 0.5 * image AUROC + 0.5 * pixel AUROC; break ties by pixel AP",
                "test_data_accessed": False,
            },
            output_path,
        )
    return metrics


def select_configuration(
    candidates_path: str, seed: int, output_path: str, device_arg: str | None
) -> None:
    candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    if not candidates:
        raise ValueError("The candidate configuration list is empty")
    results = []
    for candidate in candidates:
        metrics = evaluate_validation(
            candidate["config"], candidate["checkpoint"], seed, device_arg
        )
        results.append({"name": candidate["name"], **candidate, "metrics": metrics})
    results.sort(
        key=lambda row: (row["metrics"]["selection_score"], row["metrics"]["p_ap"]),
        reverse=True,
    )
    save_json(
        {
            "validation_seed": seed,
            "selection_rule": "maximize 0.5 * image AUROC + 0.5 * pixel AUROC; break ties by pixel AP",
            "selected": results[0]["name"],
            "ranked_candidates": results,
            "test_data_accessed": False,
        },
        output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--seed", type=int, default=3407)
    evaluate_parser.add_argument("--device")
    evaluate_parser.add_argument("--output", required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--candidates", required=True)
    select_parser.add_argument("--seed", type=int, default=3407)
    select_parser.add_argument("--device")
    select_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "evaluate":
        evaluate_validation(args.config, args.checkpoint, args.seed, args.device, args.output)
    else:
        select_configuration(args.candidates, args.seed, args.output, args.device)


if __name__ == "__main__":
    main()
