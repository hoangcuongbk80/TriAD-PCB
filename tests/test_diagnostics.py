import pandas as pd
import torch

from triad_pcb.diagnostics import aggregate_fusion_trajectory, rbf_mmd


def test_rbf_mmd_distinguishes_shifted_features():
    generator = torch.Generator().manual_seed(7)
    reference = torch.randn(128, 16, generator=generator)
    close = reference.clone()
    shifted = reference + 1.5
    assert rbf_mmd(reference, close) < 1e-6
    assert rbf_mmd(reference, shifted) > 0.05


def test_fusion_trajectory_uses_run_level_measurements(tmp_path):
    columns = {
        "sequence_index": [0, 1],
        "history_size_after_update": [0, 1],
        "lambda_text": [0.45, 0.40],
        "lambda_history": [0.10, 0.20],
        "lambda_reference": [0.45, 0.40],
    }
    first = tmp_path / "first_predictions.csv"
    second = tmp_path / "second_predictions.csv"
    pd.DataFrame(columns).to_csv(first, index=False)
    second_data = pd.DataFrame(columns)
    second_data["lambda_text"] = [0.43, 0.38]
    second_data["lambda_history"] = [0.14, 0.24]
    second_data["lambda_reference"] = [0.43, 0.38]
    second_data.to_csv(second, index=False)
    output = tmp_path / "trajectory.csv"

    aggregate_fusion_trajectory([str(tmp_path)], output)
    result = pd.read_csv(output)

    assert result.loc[0, "lambda_text"] == 0.44
    assert result.loc[1, "lambda_history"] == 0.22
    assert result.loc[0, "std_lambda_text"] > 0
