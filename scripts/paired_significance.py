from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REQUIRED_COLUMNS = {"dataset", "run_id", "method", "metric", "value"}


def paired_tests(
    input_path: Path,
    reference: str,
    baseline: str,
    confidence: float,
) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    selected = data[data["method"].isin([reference, baseline])].copy()
    if selected.empty:
        raise ValueError("Neither requested method occurs in the input table")
    duplicates = selected.duplicated(["dataset", "metric", "run_id", "method"])
    if duplicates.any():
        raise ValueError("Each dataset, metric, run, and method tuple must be unique")

    rows: list[dict[str, object]] = []
    for (dataset, metric), group in selected.groupby(["dataset", "metric"], sort=True):
        paired = group.pivot(index="run_id", columns="method", values="value")
        if reference not in paired or baseline not in paired:
            raise ValueError(f"Missing method for {dataset}, {metric}")
        incomplete = paired[[reference, baseline]].isna().any(axis=1)
        if incomplete.any():
            missing_runs = paired.index[incomplete].astype(str).tolist()
            raise ValueError(
                f"Unpaired runs for {dataset}, {metric}: {missing_runs[:10]}"
            )
        difference = (paired[reference] - paired[baseline]).to_numpy(dtype=float)
        if difference.size < 2:
            raise ValueError(f"At least two paired runs are required for {dataset}, {metric}")
        mean = float(difference.mean())
        sd = float(difference.std(ddof=1))
        standard_error = sd / np.sqrt(difference.size)
        critical = float(stats.t.ppf((1.0 + confidence) / 2.0, difference.size - 1))
        statistic, p_value = stats.ttest_rel(
            paired[reference].to_numpy(dtype=float),
            paired[baseline].to_numpy(dtype=float),
        )
        rows.append(
            {
                "dataset": dataset,
                "metric": metric,
                "reference_method": reference,
                "baseline_method": baseline,
                "paired_runs": difference.size,
                "mean_difference": mean,
                "paired_difference_sd": sd,
                "ci_lower": mean - critical * standard_error,
                "ci_upper": mean + critical * standard_error,
                "t_statistic": float(statistic),
                "degrees_of_freedom": difference.size - 1,
                "p_value_two_sided": float(p_value),
            }
        )
    if not rows:
        raise ValueError("No paired dataset and metric groups were found")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paired confidence intervals and two-sided paired t-tests"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = paired_tests(args.input, args.reference, args.baseline, args.confidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
