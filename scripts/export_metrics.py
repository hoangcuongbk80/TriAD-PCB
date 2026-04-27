from __future__ import annotations
import argparse
import json
from pathlib import Path

from src.eval.metrics import roc_auc_score, average_precision_score, f1_max

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-json", required=True)
    p.add_argument("--labels-json", required=True, help="JSON list of {path, label}")
    p.add_argument("--out", default="outputs/metrics.json")
    args = p.parse_args()

    with open(args.pred_json, "r", encoding="utf-8") as f:
        preds = json.load(f)
    with open(args.labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)

    pred_map = {x["path"]: x["score"] for x in preds}
    y_true = []
    y_score = []
    for item in labels:
        y_true.append(int(item["label"]))
        y_score.append(float(pred_map[item["path"]]))

    metrics = {
        "I-AUC": roc_auc_score(y_true, y_score),
        "I-AP": average_precision_score(y_true, y_score),
        "I-F1-max": f1_max(y_true, y_score)[0],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
