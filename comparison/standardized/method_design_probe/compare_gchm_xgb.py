"""Where exactly does GCHM-sqrt beat XGBoost? Per-class and per-subgroup, same cohort.

Read-only. Validation fold only; test fold is never opened. No refitting: both models'
stored validation predictions are reloaded from their run artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from comparison.standardized.native_reference_v1.data import DEFAULT, Reference, save

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PINNED = "2a2566e662bf87b10d97e58700f868d8d6d6b036582ec046dd5453e6dcd9f7c0"
SEEDS = (1234, 1235, 1236)


def gchm_probs(run_root: Path, seed: int):
    """Reload the stored validation logits of the best-balanced-accuracy epoch."""
    d = run_root / f"seed{seed}"
    manifest = json.loads((d / "run_manifest.json").read_text())
    assert manifest["status"] == "completed", manifest["status"]
    assert manifest["test_evaluated"] is False
    assert manifest["binding"]["weight_policy"] == "sqrt_inverse"
    history = json.loads((d / "history.json").read_text())
    best = max(history, key=lambda e: e["validation"]["balanced_acc"])
    npz = np.load(d / f"validation_{best['epoch']:03d}.npz")
    logits = npz["logits"].astype(np.float64)
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True), npz["y"], npz["ordinals"], best["epoch"]


def metrics(y, pred, proba):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, top_k_accuracy_score)
    lab = np.arange(proba.shape[1])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "top3_acc": float(top_k_accuracy_score(y, proba, k=3, labels=lab)),
    }


def main():
    ref = Reference(DEFAULT, expected=PINNED)
    labels = ref.contract["labels"]

    xgb = np.load(REPO / "comparison/standardized/native_runs/xgboost_v1/validation.npz")
    x_proba, x_y, x_ord = xgb["proba"].astype(np.float64), xgb["y"], xgb["ordinals"]

    root = REPO / "comparison/standardized/native_runs/gchm_sqrt_v2"
    stack, g_y, g_ord, eps = [], None, None, []
    for s in SEEDS:
        p, y, o, ep = gchm_probs(root, s)
        stack.append(p); eps.append(ep + 1)
        if g_y is None:
            g_y, g_ord = y, o
        else:
            assert np.array_equal(g_y, y) and np.array_equal(g_ord, o)
    assert np.array_equal(g_ord, x_ord), "cohort mismatch between GCHM and XGBoost"
    assert np.array_equal(g_y, x_y), "label mismatch"
    y = g_y

    print(f"cohort {len(y)} patients | GCHM best epochs {eps} | test_evaluated False\n")

    # Per-seed, then the seed-averaged ensemble as a secondary view.
    per_seed = [metrics(y, p.argmax(1), p) for p in stack]
    g_mean = np.mean(stack, axis=0)

    x_pred = x_proba.argmax(1)
    x_m = metrics(y, x_pred, x_proba)

    print("HEADLINE (validation, same 7448 patients)")
    print(f"{'model':26} {'bal-acc':>9} {'macroF1':>9} {'accuracy':>9} {'top3':>9}")
    for s, m in zip(SEEDS, per_seed):
        print(f"{'GCHM-sqrt seed'+str(s):26} {m['balanced_acc']:9.4f} {m['macro_f1']:9.4f} "
              f"{m['accuracy']:9.4f} {m['top3_acc']:9.4f}")
    mm = metrics(y, g_mean.argmax(1), g_mean)
    print(f"{'GCHM-sqrt 3-seed mean':26} {mm['balanced_acc']:9.4f} {mm['macro_f1']:9.4f} "
          f"{mm['accuracy']:9.4f} {mm['top3_acc']:9.4f}")
    print(f"{'XGBoost (official)':26} {x_m['balanced_acc']:9.4f} {x_m['macro_f1']:9.4f} "
          f"{x_m['accuracy']:9.4f} {x_m['top3_acc']:9.4f}")

    # Per-class recall: this is what balanced accuracy actually averages.
    g_pred = g_mean.argmax(1)
    rows = []
    for c in range(30):
        mask = y == c
        n = int(mask.sum())
        gr = float((g_pred[mask] == c).mean()) if n else 0.0
        xr = float((x_pred[mask] == c).mean()) if n else 0.0
        rows.append({"class": labels[c], "n": n, "gchm_recall": gr,
                     "xgb_recall": xr, "delta": gr - xr})

    rows.sort(key=lambda r: r["n"])
    print("\nPER-CLASS RECALL, rarest first (recall = what balanced accuracy averages)")
    print(f"{'class':46} {'n':>5} {'GCHM':>7} {'XGB':>7} {'delta':>8}")
    for r in rows:
        flag = "  <<" if r["delta"] > 0.05 else ("  !!" if r["delta"] < -0.05 else "")
        print(f"{r['class'][:46]:46} {r['n']:5d} {r['gchm_recall']:7.3f} "
              f"{r['xgb_recall']:7.3f} {r['delta']:+8.3f}{flag}")

    wins = [r for r in rows if r["delta"] > 0]
    small = [r for r in rows if r["n"] <= 100]
    big = [r for r in rows if r["n"] > 400]
    print(f"\nGCHM recall higher in {len(wins)}/30 classes")
    print(f"  rare classes (n<=100, {len(small)} of them): "
          f"GCHM {np.mean([r['gchm_recall'] for r in small]):.3f} vs "
          f"XGB {np.mean([r['xgb_recall'] for r in small]):.3f}")
    print(f"  common classes (n>400, {len(big)} of them): "
          f"GCHM {np.mean([r['gchm_recall'] for r in big]):.3f} vs "
          f"XGB {np.mean([r['xgb_recall'] for r in big]):.3f}")

    # Subgroup view: does graph size matter?
    counts = (np.diff(ref.arrays["node_ptr"]) - 1)[g_ord]
    print("\nBY CONCEPT COUNT (graph size)")
    print(f"{'group':18} {'n':>6} {'GCHM bal':>9} {'XGB bal':>9} {'GCHM acc':>9} {'XGB acc':>9}")
    for name, sel in [("0-1 concept", counts <= 1), ("2-4", (counts >= 2) & (counts <= 4)),
                      ("5-9", (counts >= 5) & (counts <= 9)), ("10+", counts >= 10)]:
        if sel.sum() < 30:
            continue
        gm = metrics(y[sel], g_pred[sel], g_mean[sel])
        xm = metrics(y[sel], x_pred[sel], x_proba[sel])
        print(f"{name:18} {int(sel.sum()):6d} {gm['balanced_acc']:9.4f} {xm['balanced_acc']:9.4f} "
              f"{gm['accuracy']:9.4f} {xm['accuracy']:9.4f}")

    save(HERE / "gchm_vs_xgboost.json", {
        "scope": "validation_only_no_refit",
        "contract_sha256": ref.fingerprint,
        "cohort": int(len(y)),
        "test_evaluated": False,
        "gchm_best_epochs": eps,
        "gchm_per_seed": per_seed,
        "gchm_3seed_mean": mm,
        "xgboost": x_m,
        "per_class": rows,
    })
    print(f"\nwrote {HERE/'gchm_vs_xgboost.json'}")


if __name__ == "__main__":
    main()
