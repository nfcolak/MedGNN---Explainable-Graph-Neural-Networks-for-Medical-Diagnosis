"""Shared multi-class metrics so both analyses report identical numbers.

Mirrors `_multiclass_metrics` used by protgnn_analysis/train.py. Kept here so
graphcare_analysis can reuse the exact same metric definitions for a fair
comparison. (TODO: migrate protgnn_analysis/train.py to import from here too,
removing the duplicate definition.)
"""
import numpy as np


def multiclass_metrics(labels, preds, probs=None) -> dict:
    """Balanced accuracy, macro/micro-F1, and top-3/top-5 accuracy.

    Plain accuracy is expected to be added by the caller.
    """
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 top_k_accuracy_score)
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    out = {
        "balanced_acc": round(float(balanced_accuracy_score(labels, preds)), 6),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 6),
        "micro_f1": round(float(f1_score(labels, preds, average="micro", zero_division=0)), 6),
    }
    if probs is not None and getattr(probs, "ndim", 0) == 2:
        n_cls = probs.shape[1]
        lab_range = np.arange(n_cls)
        for k in (3, 5):
            if n_cls > k:
                out[f"top{k}_acc"] = round(float(
                    top_k_accuracy_score(labels, probs, k=k, labels=lab_range)), 6)
    return out
