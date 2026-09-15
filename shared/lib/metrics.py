"""Shared multi-class metrics for every benchmark method."""
import numpy as np


def multiclass_metrics(labels, preds, probs=None) -> dict:
    """Return the benchmark's exact six-metric multiclass schema."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, top_k_accuracy_score)
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    probs = np.asarray(probs) if probs is not None else None
    if probs is None or probs.ndim != 2 or probs.shape[1] <= 5:
        raise ValueError(
            "Multiclass benchmark metrics require a 2D probability matrix with "
            "at least six class columns."
        )
    out = {
        "accuracy": round(float(accuracy_score(labels, preds)), 6),
        "balanced_acc": round(float(balanced_accuracy_score(labels, preds)), 6),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 6),
        "micro_f1": round(float(f1_score(labels, preds, average="micro", zero_division=0)), 6),
    }
    lab_range = np.arange(probs.shape[1])
    for k in (3, 5):
        out[f"top{k}_acc"] = round(float(
            top_k_accuracy_score(labels, probs, k=k, labels=lab_range)), 6)
    return out
