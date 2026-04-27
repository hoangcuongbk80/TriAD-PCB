from __future__ import annotations
import numpy as np

def _rankdata(a):
    temp = np.argsort(a)
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(a), dtype=float) + 1
    return ranks

def roc_auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

def average_precision_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true == 1)
    fp = np.cumsum(y_true == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max((y_true == 1).sum(), 1)
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * max(r - prev_r, 0.0)
        prev_r = r
    return float(ap)

def f1_max(y_true, y_score, num_thresholds: int = 200):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return float("nan"), 0.5
    thresholds = np.linspace(y_score.min(), y_score.max(), num_thresholds)
    best = (0.0, thresholds[0])
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = np.logical_and(pred == 1, y_true == 1).sum()
        fp = np.logical_and(pred == 1, y_true == 0).sum()
        fn = np.logical_and(pred == 0, y_true == 1).sum()
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best[0]:
            best = (f1, t)
    return float(best[0]), float(best[1])

def pro_score(masks_true, masks_pred):
    scores = []
    for yt, yp in zip(masks_true, masks_pred):
        yt = np.asarray(yt).astype(bool)
        yp = np.asarray(yp).astype(bool)
        inter = np.logical_and(yt, yp).sum()
        denom = yt.sum()
        scores.append(inter / denom if denom > 0 else 1.0)
    return float(np.mean(scores)) if scores else float("nan")
