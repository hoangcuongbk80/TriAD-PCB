from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def maximum_f1(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(labels.astype(np.uint8), scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.nanargmax(f1))
    threshold = float(thresholds[min(index, len(thresholds) - 1)]) if len(thresholds) else 0.5
    return float(f1[index]), threshold


def pro_auc(masks: np.ndarray, maps: np.ndarray, max_fpr: float = 0.3, steps: int = 200) -> float:
    masks = masks.astype(bool)
    normal_pixels = ~masks
    thresholds = np.linspace(float(maps.max()), float(maps.min()), steps)
    fprs, pros = [], []
    components = [ndimage.label(mask)[0] for mask in masks]
    for threshold in thresholds:
        prediction = maps >= threshold
        false_positive_rate = prediction[normal_pixels].mean() if normal_pixels.any() else 0.0
        if false_positive_rate > max_fpr:
            continue
        overlap = []
        for prediction_i, component_i in zip(prediction, components):
            for region_id in range(1, int(component_i.max()) + 1):
                region = component_i == region_id
                overlap.append(float(prediction_i[region].mean()))
        if overlap:
            fprs.append(float(false_positive_rate))
            pros.append(float(np.mean(overlap)))
    if len(fprs) < 2:
        return 0.0
    order = np.argsort(fprs)
    x = np.asarray(fprs)[order]
    y = np.asarray(pros)[order]
    unique_x, unique_index = np.unique(x, return_index=True)
    y = y[unique_index]
    return float(np.trapezoid(y, unique_x) / max_fpr)


def evaluate_predictions(
    image_labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    pixel_maps: np.ndarray,
    thresholds: int = 200,
) -> dict[str, float]:
    pixel_labels = masks.astype(np.uint8).ravel()
    pixel_scores = pixel_maps.ravel()
    image_f1, image_threshold = maximum_f1(image_labels, image_scores)
    pixel_f1, pixel_threshold = maximum_f1(pixel_labels, pixel_scores)
    return {
        "i_auc": float(roc_auc_score(image_labels, image_scores)),
        "i_f1_max": image_f1,
        "i_ap": float(average_precision_score(image_labels, image_scores)),
        "p_auc": float(roc_auc_score(pixel_labels, pixel_scores)),
        "p_f1_max": pixel_f1,
        "p_ap": float(average_precision_score(pixel_labels, pixel_scores)),
        "pro": pro_auc(masks, pixel_maps, steps=thresholds),
        "i_threshold": image_threshold,
        "p_threshold": pixel_threshold,
    }
