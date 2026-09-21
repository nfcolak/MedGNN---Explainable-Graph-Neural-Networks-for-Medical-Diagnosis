"""Decisive test: is the GNN-to-XGBoost gap interaction ORDER, or capacity?

Same model family, same features, same split. Only tree depth varies.
depth=1 stumps cannot form ANY feature conjunction -> pure additive model.
depth>=2 can form concept x hub conjunctions.

If depth-1 lands near the GNN family (0.544-0.564) and depth-4 jumps to ~0.58,
the gap is interaction ORDER across the concept/hub blocks, not capacity.

Read-only against the pinned native artifact. Test fold never touched.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from comparison.standardized.native_reference_v1.data import DEFAULT, Reference, save
from comparison.standardized.xgboost_native_baseline import build_tabular_matrix
from shared.lib.metrics import multiclass_metrics

HERE = Path(__file__).resolve().parent
PINNED = "2a2566e662bf87b10d97e58700f868d8d6d6b036582ec046dd5453e6dcd9f7c0"


def fit(Xtr, ytr, Xva, yva, depth, seed, label, n_estimators=800):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=30, n_estimators=n_estimators,
        learning_rate=0.05, max_depth=depth, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=2.0, min_child_weight=3.0, tree_method="hist",
        eval_metric="mlogloss", random_state=seed, n_jobs=6,
        early_stopping_rounds=50)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    proba = model.predict_proba(Xva)
    m = multiclass_metrics(yva, proba.argmax(1), proba)
    m["best_iteration"] = int(model.best_iteration)
    m["max_depth"] = depth
    print(f"  {label:38} macroF1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
          f"bal={m['balanced_acc']:.4f} iter={m['best_iteration']}", flush=True)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, default=DEFAULT)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not a.execute:
        print("dry-run: pass --execute"); return

    t0 = time.time()
    ref = Reference(a.artifact, expected=PINNED)
    X, y, folds, names, meta = build_tabular_matrix(a.artifact)
    tr = np.flatnonzero(folds == 0)
    va = np.flatnonzero(folds == 1)
    print(f"artifact ok | contract {ref.fingerprint[:12]} | train {len(tr)} val {len(va)}")

    print("\nInteraction order sweep on concepts+hub (324 features):")
    results = {}
    for d in (1, 2, 3, 4, 6):
        results[f"depth_{d}"] = fit(X[tr], y[tr], X[va], y[va], d, a.seed,
                                    f"XGBoost depth={d}"
                                    + ("  (additive, no conjunctions)" if d == 1 else ""))

    out = {
        "scope": "diagnostic_interaction_order_probe_not_benchmark",
        "contract_sha256": ref.fingerprint,
        "input_sha256": ref.contract["input_sha256"],
        "selected_counts": [int(len(tr)), int(len(va))],
        "seed": a.seed,
        "test_evaluated": False,
        "results": results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    save(HERE / "depth_results.json", out)
    print(f"\nwrote depth_results.json  ({out['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
