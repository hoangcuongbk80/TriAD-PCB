from __future__ import annotations

import argparse
import copy
from pathlib import Path

from .controlled_study import aggregate_results, run_plan
from .config import load_yaml, resolve_path, save_yaml
from .utils import load_json, save_json


def make_plan(base_config: str, output_path: str) -> None:
    config = load_yaml(base_config)
    output = Path(output_path).resolve()
    generated = output.parent / "generated_sensitivity_configs"
    generated.mkdir(parents=True, exist_ok=True)
    candidates = [
        line.strip()
        for line in resolve_path(config, "materials/candidate_anchors_16.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    seeds = load_json(resolve_path(config, "materials/seeds.json"))["training_seeds"]
    default_history = int(config["memory"]["history_capacity"])
    default_support = int(config["evaluation"]["support_size"])
    conditions = [
        ("anchor_k4", candidates[:4], 4, default_history, default_support),
        ("default_k8_j4_n50_k4", candidates[:8], 4, default_history, default_support),
        ("anchor_k12", candidates[:12], 4, default_history, default_support),
        ("prompt_j2", candidates[:8], 2, default_history, default_support),
        ("prompt_j6", candidates[:8], 6, default_history, default_support),
        ("history_n25", candidates[:8], 4, 25, default_support),
        ("history_n100", candidates[:8], 4, 100, default_support),
        ("support_k1", candidates[:8], 4, default_history, 1),
        ("support_k2", candidates[:8], 4, default_history, 2),
        ("support_k8", candidates[:8], 4, default_history, 8),
    ]
    jobs = []
    for condition, anchors, abnormal_count, history_capacity, support_size in conditions:
        anchor_file = generated / f"anchors_{condition}.yaml"
        save_yaml({"anchors": anchors}, anchor_file)
        condition_config = copy.deepcopy(config)
        condition_config["dataset"]["manifest"] = str(resolve_path(config, config["dataset"]["manifest"]))
        condition_config["materials"]["anchors"] = str(anchor_file)
        condition_config["materials"]["prompts"] = str(
            resolve_path(config, config["materials"]["prompts"])
        )
        condition_config["model"]["abnormal_prompts_per_anchor"] = abnormal_count
        condition_config["memory"]["history_capacity"] = history_capacity
        condition_config["evaluation"]["support_size"] = support_size
        condition_config["output"]["root"] = str(output.parent / "runs")
        config_file = generated / f"{condition}.yaml"
        save_yaml(condition_config, config_file)
        for seed in seeds:
            jobs.append(
                {
                    "condition": condition,
                    "report_group": condition,
                    "seed": int(seed),
                    "config": str(config_file),
                    "run_name": f"{condition}_seed{seed}",
                }
            )
    save_json(
        {
            "base_config": str(Path(base_config).resolve()),
            "nested_anchor_source": str(resolve_path(config, "materials/candidate_anchors_16.txt")),
            "jobs": jobs,
        },
        output,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--base-config", required=True)
    plan_parser.add_argument("--output", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--plan", required=True)
    run_parser.add_argument("--device")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input", required=True)
    aggregate_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        make_plan(args.base_config, args.output)
    elif args.command == "run":
        run_plan(args.plan, args.device)
    else:
        aggregate_results(args.input, args.output)


if __name__ == "__main__":
    main()
