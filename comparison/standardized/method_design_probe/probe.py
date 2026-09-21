"""Where does the GNN family lose to XGBoost? Isolated diagnostic, validation only.

Read-only against the pinned native artifact. Writes nothing outside this directory.
Test fold is never touched. These are attribution probes, NOT benchmark entries.

Question: the 5 GNNs all compress 331 native input slots through a narrow linear
encoder before any interaction is computed, then aggregate additively. XGBoost sees
324 raw features with depth-4 conjunctions. Is the gap (a) graph topology, (b) the
width bottleneck, or (c) missing multiplicative concept x hub interaction?
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from comparison.standardized.native_reference_v1.data import DEFAULT, Reference, save
from comparison.standardized.xgboost_native_baseline import build_tabular_matrix
from shared.lib.metrics import multiclass_metrics

HERE = Path(__file__).resolve().parent
PINNED = "2a2566e662bf87b10d97e58700f868d8d6d6b036582ec046dd5453e6dcd9f7c0"


def fit_mlp(Xtr, ytr, Xva, yva, width, depth, epochs, seed, lr=1e-3, bs=128,
            label=""):
    """Plain MLP on the same graph-level tabular view. No graph, no pooling."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    layers, d = [], Xtr.shape[1]
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(0.3)]
        d = width
    layers += [nn.Linear(d, 30)]
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva)
    n = len(ytr)
    best = {"macro_f1": -1.0}
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(n)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(Xva_t)
            proba = torch.softmax(logits, dim=1).numpy()
        m = multiclass_metrics(yva, proba.argmax(1), proba)
        if m["macro_f1"] > best["macro_f1"]:
            best = dict(m)
            best["epoch"] = ep + 1
    best["params"] = sum(p.numel() for p in model.parameters())
    print(f"  {label:44} macroF1={best['macro_f1']:.4f} acc={best['accuracy']:.4f} "
          f"bal={best['balanced_acc']:.4f} ep={best['epoch']} params={best['params']}",
          flush=True)
    return best


def fit_xgb(Xtr, ytr, Xva, yva, seed, label="", n_estimators=600):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=30, n_estimators=n_estimators,
        learning_rate=0.03, max_depth=4, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=2.0, min_child_weight=3.0, tree_method="hist",
        eval_metric="mlogloss", random_state=seed, n_jobs=6,
        early_stopping_rounds=50)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    proba = model.predict_proba(Xva)
    m = multiclass_metrics(yva, proba.argmax(1), proba)
    m["best_iteration"] = int(model.best_iteration)
    print(f"  {label:44} macroF1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} "
          f"bal={m['balanced_acc']:.4f} iter={m['best_iteration']}", flush=True)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, default=DEFAULT)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if not a.execute:
        print("dry-run: pass --execute"); return

    t0 = time.time()
    ref = Reference(a.artifact, expected=PINNED)
    X, y, folds, names, meta = build_tabular_matrix(a.artifact)
    tr = np.flatnonzero(folds == 0)
    va = np.flatnonzero(folds == 1)
    assert len(tr) == 59607 and len(va) == 7448, (len(tr), len(va))
    print(f"artifact ok | contract {ref.fingerprint[:12]} | train {len(tr)} val {len(va)}")

    concept = slice(0, 192)
    hub = slice(192, 324)
    # Hub is already train-fit z-scored in the artifact; concepts are binary.
    results = {}

    print("\n[1] Signal attribution — which block carries the discriminative signal?")
    results["xgb_concepts_only"] = fit_xgb(
        X[tr][:, concept], y[tr], X[va][:, concept], y[va], a.seed,
        "XGBoost  concepts only (192)")
    results["xgb_hub_only"] = fit_xgb(
        X[tr][:, hub], y[tr], X[va][:, hub], y[va], a.seed,
        "XGBoost  hub only (132)")
    results["xgb_full"] = fit_xgb(
        X[tr], y[tr], X[va], y[va], a.seed,
        "XGBoost  concepts+hub (324)")

    print("\n[2] Is it the graph, or the width bottleneck? Same features, no graph.")
    for w in (64, 256):
        results[f"mlp_w{w}"] = fit_mlp(
            X[tr], y[tr], X[va], y[va], w, 2, a.epochs, a.seed,
            label=f"MLP      width {w}, 2 layers, no graph")

    print("\n[3] Does explicit concept x hub multiplicative interaction help?")
    results["mlp_w256_d3"] = fit_mlp(
        X[tr], y[tr], X[va], y[va], 256, 3, a.epochs, a.seed,
        label="MLP      width 256, 3 layers, no graph")

    out = {
        "scope": "diagnostic_attribution_probe_not_benchmark",
        "contract_sha256": ref.fingerprint,
        "input_sha256": ref.contract["input_sha256"],
        "counts": meta["counts"],
        "selected_counts": [int(len(tr)), int(len(va))],
        "seed": a.seed,
        "mlp_epochs": a.epochs,
        "test_evaluated": False,
        "results": results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    save(HERE / "probe_results.json", out)
    print(f"\nwrote {HERE/'probe_results.json'}  ({out['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
