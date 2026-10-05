from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import numpy as np

from .config import load_yaml, resolve_path, save_yaml
from .evaluate import evaluate_once


VARIANTS = {
    "text_only": ["text"],
    "online_only": ["online"],
    "reference_only": ["reference"],
    "online_reference": ["online", "reference"],
    "text_reference": ["text", "reference"],
    "text_online": ["text", "online"],
    "full": ["text", "online", "reference"],
}
METRICS = ["i_auc", "i_f1_max", "i_ap", "p_auc", "p_f1_max", "p_ap", "pro"]


def run_branch_study(
    config_path: str,
    checkpoint_path: str,
    output_dir: str | Path,
    device: str | None,
) -> Path:
    config = load_yaml(config_path)
    output = Path(output_dir).resolve()
    config_dir = output / "configs"
    evaluation_dir = output / "evaluations"
    config_dir.mkdir(parents=True, exist_ok=True)
    config["dataset"]["manifest"] = str(resolve_path(config, config["dataset"]["manifest"]))
    config["materials"]["anchors"] = str(resolve_path(config, config["materials"]["anchors"]))
    config["materials"]["prompts"] = str(resolve_path(config, config["materials"]["prompts"]))
    rows = []
    for name, branches in VARIANTS.items():
        variant = copy.deepcopy(config)
        variant["evaluation"]["branches"] = branches
        variant_path = config_dir / f"{name}.yaml"
        save_yaml(variant, variant_path)
        repetitions = []
        for support_seed in variant["evaluation"]["support_seeds"]:
            for stream_seed in variant["evaluation"]["stream_seeds"]:
                repetitions.append(
                    evaluate_once(
                        str(variant_path),
                        checkpoint_path,
                        int(support_seed),
                        int(stream_seed),
                        device,
                        evaluation_dir / name,
                    )
                )
        row: dict[str, object] = {
            "setting": name,
            "branches": "+".join(branches),
            "repetitions": len(repetitions),
        }
        row.update(
            {
                metric: 100.0 * float(np.mean([item[metric] for item in repetitions]))
                for metric in METRICS
            }
        )
        rows.append(row)
    summary = output / "summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all seven branch compositions")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    print(run_branch_study(args.config, args.checkpoint, args.output, args.device))


if __name__ == "__main__":
    main()
