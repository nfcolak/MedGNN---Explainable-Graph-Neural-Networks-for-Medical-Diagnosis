"""
Class-confusion analysis for the disease target.

Trains a quick XGBoost on a merged CSV and reports WHICH diseases get confused
with which — to (a) validate that confusions are clinically coherent and (b)
surface candidate classes to merge. Works on any merged_ed*_sample CSV, so it
can study the standard top-30 set OR a broader all-label set produced with
    TOP_N_DISEASES=150 python3 scripts/merge_ed.py

Outputs (to outputs/results/confusion/):
  confusion_matrix.csv        full true×pred counts (disease names)
  most_confused_pairs.csv     ranked (true -> pred) off-diagonal counts
  per_class_metrics.csv       precision/recall/F1/support, worst-recall first

Usage:
    python3 scripts/confusion_analysis.py
    python3 scripts/confusion_analysis.py --csv data/merged_ed_top150_sample_60k.csv
    python3 scripts/confusion_analysis.py --top_pairs 40
"""

import os
import sys
import argparse
from collections import Counter

import numpy as np
import pandas as pd

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for p in (_THIS, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from disease_baseline import build_xy

OUT_DIR = os.path.join(_ROOT, "outputs", "results", "confusion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(_ROOT, "data", "merged_ed.csv"))
    ap.add_argument("--top_pairs", type=int, default=30, help="# most-confused pairs to list")
    ap.add_argument("--min_class", type=int, default=5, help="drop classes with < this many rows")
    args = ap.parse_args()

    print(f"Loading {args.csv}")
    df = pd.read_csv(args.csv, low_memory=False)
    X, y_raw, feat = build_xy(df)

    vc = pd.Series(y_raw).value_counts()
    keep = set(vc[vc >= args.min_class].index)
    m = np.array([v in keep for v in y_raw])
    X, y_raw = X[m], y_raw[m]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n = len(le.classes_)
    print(f"  rows={len(X)}  classes={n}  features={X.shape[1]}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    import xgboost as xgb
    clf = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            objective="multi:softprob", num_class=n,
                            tree_method="hist", n_jobs=-1, eval_metric="mlogloss")
    print("Training xgb ...")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float((pred == yte).mean())
    print(f"  test accuracy: {acc:.4f}  (n={len(yte)})")

    names = le.classes_
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- full confusion matrix (names) ---
    cm = confusion_matrix(yte, pred, labels=np.arange(n))
    cm_df = pd.DataFrame(cm, index=names, columns=names)
    cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))

    # --- most-confused (true -> pred) pairs ---
    conf = Counter()
    for t, p in zip(yte, pred):
        if t != p:
            conf[(names[t], names[p])] += 1
    pairs = [{"true": t, "pred": p, "count": c,
              "pct_of_true_class": round(100 * c / max((yte == le.transform([t])[0]).sum(), 1), 1)}
             for (t, p), c in conf.most_common()]
    pd.DataFrame(pairs).to_csv(os.path.join(OUT_DIR, "most_confused_pairs.csv"), index=False)

    # --- per-class metrics (worst recall first) ---
    pr, rc, f1, sup = precision_recall_fscore_support(yte, pred, labels=np.arange(n),
                                                      zero_division=0)
    pc = pd.DataFrame({"disease": names, "precision": pr.round(3), "recall": rc.round(3),
                       "f1": f1.round(3), "support": sup}).sort_values("recall")
    pc.to_csv(os.path.join(OUT_DIR, "per_class_metrics.csv"), index=False)

    # --- console summary ---
    print(f"\n=== TOP {args.top_pairs} MOST-CONFUSED PAIRS (true -> pred) ===")
    for r in pairs[:args.top_pairs]:
        print(f"{r['count']:4d} ({r['pct_of_true_class']:4.1f}%)  "
              f"{r['true'][:34]:34s} -> {r['pred'][:34]}")

    print(f"\n=== 12 HARDEST CLASSES (lowest recall) ===")
    for _, r in pc.head(12).iterrows():
        print(f"  recall={r['recall']:.2f}  f1={r['f1']:.2f}  n={int(r['support']):4d}  {r['disease'][:44]}")

    print(f"\nSaved to {OUT_DIR}/  (confusion_matrix.csv, most_confused_pairs.csv, per_class_metrics.csv)")


if __name__ == "__main__":
    main()
