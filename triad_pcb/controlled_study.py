from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .config import load_yaml, resolve_path, save_yaml
from .evaluate import evaluate_once
from .train import train
from .utils import load_json, save_json


def make_plan(base_config: str, output_path: str) -> None:
    config = load_yaml(base_config)
    output = Path(output_path).resolve()
    generated = output.parent / "generated_configs"
    generated.mkdir(parents=True, exist_ok=True)
    controls = yaml.safe_load(resolve_path(config, "materials/anchor_controls.yaml").read_text(encoding="utf-8"))
    seeds = load_json(resolve_path(config, "materials/seeds.json"))["training_seeds"]
    default_prompts = str(resolve_path(config, "materials/prompts_default.yaml"))
    conditions = [
        ("clip_ranked", controls["clip_ranked"], default_prompts, "clip_ranked"),
        ("manual", controls["manual_balanced"], default_prompts, "manual"),
    ]
    conditions.extend(
        (name, anchors, default_prompts, "random")
        for name, anchors in controls["random_subsets"].items()
    )
    for variant in ["normal", "abnormal", "both"]:
        conditions.append(
            (
                f"paraphrase_{variant}",
                controls["clip_ranked"],
                str(resolve_path(config, f"materials/prompts_paraphrase_{variant}.yaml")),
                f"paraphrase_{variant}",
            )
        )
    jobs = []
    for condition, anchors, prompts, report_group in conditions:
        anchor_file = generated / f"anchors_{condition}.yaml"
        save_yaml({"anchors": anchors}, anchor_file)
        condition_config = copy.deepcopy(config)
        condition_config["dataset"]["manifest"] = str(resolve_path(config, config["dataset"]["manifest"]))
        condition_config["materials"]["anchors"] = str(anchor_file)
        condition_config["materials"]["prompts"] = prompts
        condition_config["output"]["root"] = str(output.parent / "runs")
        config_file = generated / f"{condition}.yaml"
        save_yaml(condition_config, config_file)
        for seed in seeds:
            jobs.append(
                {
                    "condition": condition,
                    "report_group": report_group,
                    "seed": int(seed),
                    "config": str(config_file),
                    "run_name": f"{condition}_seed{seed}",
                }
            )
    save_json({"base_config": str(Path(base_config).resolve()), "jobs": jobs}, output)


def run_plan(plan_path: str, device: str | None) -> None:
    plan = load_json(plan_path)
    plan_root = Path(plan_path).resolve().parent
    for job in plan["jobs"]:
        checkpoint = train(job["config"], job["seed"], device, job["run_name"])
        config = load_yaml(job["config"])
        repetition_dir = plan_root / "evaluations" / job["condition"] / f"seed{job['seed']}"
        repetitions = []
        for support_seed in config["evaluation"]["support_seeds"]:
            for stream_seed in config["evaluation"]["stream_seeds"]:
                repetitions.append(
                    evaluate_once(
                        job["config"],
                        str(checkpoint),
                        int(support_seed),
                        int(stream_seed),
                        device,
                        repetition_dir,
                    )
                )
        metric_keys = ["i_auc", "i_f1_max", "i_ap", "p_auc", "p_f1_max", "p_ap", "pro"]
        aggregate = {key: float(np.mean([row[key] for row in repetitions])) for key in metric_keys}
        aggregate.update(
            {
                "condition": job["condition"],
                "report_group": job["report_group"],
                "training_seed": job["seed"],
                "repetitions": len(repetitions),
            }
        )
        save_json(aggregate, repetition_dir / "aggregate.json")


def aggregate_results(input_dir: str, output_path: str) -> None:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in Path(input_dir).rglob("aggregate.json")]
    if not rows:
        raise FileNotFoundError("No per-model aggregate.json files were found")
    metric_keys = ["i_auc", "i_f1_max", "i_ap", "p_auc", "p_f1_max", "p_ap", "pro"]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["report_group"], []).append(row)
    output_rows = []
    for group, group_rows in grouped.items():
        result = {"condition": group, "models": len(group_rows)}
        for metric in metric_keys:
            values = np.asarray([row[metric] for row in group_rows]) * 100.0
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output_rows.append(result)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)


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
