"""Behavioral contract tests shared by benchmark method entry points."""


import numpy as np

from shared.lib.metrics import multiclass_metrics


def _thirty_class_predictions():
    labels = np.arange(6, dtype=np.int64)
    probs = np.zeros((6, 30), dtype=np.float64)
    rankings = [
        [0],
        [2, 1],
        [2],
        [4, 0, 1, 3],
        [4],
        [0, 1, 2, 3, 4, 5],
    ]
    for row, ranking in enumerate(rankings):
        for rank, class_id in enumerate(ranking):
            probs[row, class_id] = 1.0 - rank * 0.1
    preds = probs.argmax(axis=1)
    return labels, preds, probs


def test_shared_multiclass_metrics_has_exact_six_metric_schema_and_values():
    labels, preds, probs = _thirty_class_predictions()

    metrics = multiclass_metrics(labels, preds, probs)

    assert metrics == {
        "accuracy": 0.5,
        "balanced_acc": 0.5,
        "macro_f1": 0.333333,
        "micro_f1": 0.5,
        "top3_acc": 0.666667,
        "top5_acc": 0.833333,
    }
