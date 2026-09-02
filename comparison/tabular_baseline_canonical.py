"""Tabular baselines on the CANONICAL split, so their numbers sit in the same
table as ProtGNN and GraphCare.

baselines/disease_baseline.py uses its own random 80/20 split, which is not
comparable with comparison/canonical_split.json. This script reuses the same
feature construction (build_xy) but assigns rows to train/val/test by the
canonical subject fold, and reports:

  * majority-class baseline (predict the most frequent training class)
  * XGBoost (tuned, class-weighted, early stopping on the canonical val fold)

Metrics match shared/lib/metrics.py: accuracy, balanced accuracy, macro-F1,
micro-F1, top-3 and top-5 accuracy.

Run:  PYTHONPATH=. python3 comparison/tabular_baseline_canonical.py
Out:  comparison/tabular/metrics.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             top_k_accuracy_score)
from sklearn.utils.class_weight import compute_sample_weight

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "baselines"))
from disease_baseline import build_xy  # noqa: E402

SPLIT = REPO / "comparison" / "canonical_split.json"
OUT_DIR = REPO / "comparison" / "tabular"


def _metrics(y_true, pred, proba, n_classes):
    labels = np.arange(n_classes)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "top3_acc": float(top_k_accuracy_score(y_true, proba, k=3, labels=labels)),
        "top5_acc": float(top_k_accuracy_score(y_true, proba, k=5, labels=labels)),
    }


def main():
    split = json.load(open(SPLIT))
    fold, classes = split["fold"], split["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    df = pd.read_csv(REPO / "data" / "merged_ed.csv", low_memory=False)
    # build_xy drops rows without a disease_1 label; mirror that filter here so
    # the subject ids stay aligned with the returned X.
    kept = df[df["disease_1"].fillna("").astype(str).str.len() > 0]
    X, y_raw, _ = build_xy(df)
    assert len(X) == len(kept), (len(X), len(kept))

    f = kept["subject_id"].astype(str).map(fold).values
    y = np.array([c2i.get(str(v), -1) for v in y_raw])
    keep = (~pd.isna(f)) & (y >= 0)
    X, y, f = X[keep], y[keep], f[keep].astype(int)

    tr, va, te = f == 0, f == 1, f == 2
    print(f"  canonical folds: train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"classes={n_classes} features={X.shape[1]}")

    results = {}

    # --- majority-class baseline -------------------------------------------
    prior = np.bincount(y[tr], minlength=n_classes) / tr.sum()
    maj_proba = np.tile(prior, (int(te.sum()), 1))
    maj_pred = np.full(int(te.sum()), int(prior.argmax()))
    results["majority"] = _metrics(y[te], maj_pred, maj_proba, n_classes)
    results["majority"]["class"] = classes[int(prior.argmax())]
    print("  majority:", json.dumps(results["majority"]))

    # --- XGBoost ------------------------------------------------------------
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=800, max_depth=7, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=3, reg_lambda=2.0, reg_alpha=0.5, gamma=0.1,
        objective="multi:softprob", num_class=n_classes,
        tree_method="hist", n_jobs=-1, eval_metric="mlogloss",
        early_stopping_rounds=30,
    )
    print("  training xgboost on the canonical train fold ...")
    clf.fit(X[tr], y[tr],
            sample_weight=compute_sample_weight("balanced", y[tr]),
            eval_set=[(X[va], y[va])], verbose=50)
    best = getattr(clf, "best_iteration", None)
    proba = clf.predict_proba(X[te])
    results["xgboost"] = _metrics(y[te], proba.argmax(1), proba, n_classes)
    results["xgboost"]["best_iteration"] = int(best) if best is not None else None
    print("  xgboost:", json.dumps(results["xgboost"]))

    results["meta"] = {
        "split": "comparison/canonical_split.json",
        "n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()),
        "n_classes": n_classes, "n_features": int(X.shape[1]),
        "csv": "data/merged_ed.csv",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT_DIR / "metrics.json", "w"), indent=2)
    print("  saved ->", OUT_DIR / "metrics.json")


if __name__ == "__main__":
    main()
