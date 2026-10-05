import numpy as np

from triad_pcb.metrics import evaluate_predictions


def test_perfect_predictions_have_unit_detection_metrics():
    labels = np.array([0, 1])
    scores = np.array([0.0, 1.0])
    masks = np.array([[[0, 0], [0, 0]], [[0, 1], [0, 0]]])
    maps = masks.astype(float)
    metrics = evaluate_predictions(labels, scores, masks, maps, thresholds=20)
    assert metrics["i_auc"] == 1.0
    assert metrics["i_ap"] == 1.0
    assert metrics["p_auc"] == 1.0
    assert metrics["p_ap"] == 1.0
