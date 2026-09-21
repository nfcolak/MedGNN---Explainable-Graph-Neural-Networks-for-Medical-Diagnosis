"""Transductive patient-graph experiment: PNP vs same-setting baselines.

Run with ``python -m comparison.standardized.patient_graph_v1.run --execute``.

Fairness contract
-----------------
The existing GCHM/XGBoost numbers were produced in the single-patient setting and
CANNOT be compared with a transductive model. This runner therefore evaluates
every arm in the same transductive setting, on the same split, cohort order,
class weighting and metric:

  own_only      the identical network with propagation disabled — isolates
                exactly what the neighbourhood contributes, holding architecture,
                data, optimizer and budget fixed.
  knn_vote      the fixed aggregator already measured (+0.0082 blended); the
                learned aggregator must beat this to justify itself.
  xgb_plus_knn  XGBoost given the same neighbour evidence as extra input columns.
                Without this arm a PNP win could simply mean "neighbours help",
                not "a learned graph aggregator helps"; this hands the tree model
                the same information in the form it can digest.
  pnp           the learned aggregator.

Fold 2 is never read. Neighbours and their labels come only from fold 0.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from comparison.standardized.native_reference_v1.data import (
    DEFAULT, PINNED_CONTRACT, Reference, digest, sha, save, REPO,
)
from comparison.standardized.performance_review import class_weights
from comparison.standardized.xgboost_native_baseline import build_tabular_matrix
from comparison.standardized.patient_graph_v1.build_neighbors import (
    DEFAULT_OUTPUT as NEIGHBOURS, load_neighbours,
)
from gchm_analysis.pnp import PNP
from shared.lib.metrics import multiclass_metrics

HERE = Path(__file__).resolve().parent
CLASSES = 30
SELECTION = "validation macro_f1; full fixed epoch budget; no early stop"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_bindings():
    paths = [Path("comparison/standardized/patient_graph_v1/run.py"),
             Path("comparison/standardized/patient_graph_v1/build_neighbors.py"),
             Path("gchm_analysis/pnp.py"),
             Path("comparison/standardized/xgboost_native_baseline.py"),
             Path("comparison/standardized/native_reference_v1/data.py"),
             Path("comparison/standardized/performance_review.py"),
             Path("shared/lib/metrics.py")]
    return {str(p): sha(REPO / p) for p in paths}


def neighbour_vote_probs(labels_tr, nb, k, classes=CLASSES, alpha=0.5):
    """Smoothed neighbour label distribution — the fixed-aggregator baseline."""
    lab = labels_tr[nb[:, :k]]
    counts = np.stack([(lab == c).sum(1) for c in range(classes)], 1).astype(np.float64)
    return (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * classes)


def evaluate(model, feats, nb, dist, y, batch=1024, no_neighbours=False):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(feats), batch):
            e = slice(s, min(s + batch, len(feats)))
            idx = nb[e]
            out.append(model(feats[e], idx, feats_all[idx], dist[e],
                             no_neighbours=no_neighbours).cpu())
    logits = torch.cat(out)
    require(torch.isfinite(logits).all(), "Nonfinite validation logits")
    p = logits.softmax(1).numpy()
    return logits.numpy(), multiclass_metrics(y, p.argmax(1), p)


def run(args):
    ref = Reference(args.artifact, expected=PINNED_CONTRACT)
    X, y, folds, feature_names, meta = build_tabular_matrix(args.artifact)
    data, nbman = load_neighbours(args.neighbours, ref, args.k)
    tr = np.flatnonzero(folds == 0)
    va = np.flatnonzero(folds == 1)
    require(np.array_equal(tr, data["train_ordinals"]), "Train cohort mismatch")
    require(np.array_equal(va, data["validation_ordinals"]), "Validation cohort mismatch")

    mu, sd = data["scaler_mean"], data["scaler_std"]
    Z = ((X - mu) / sd).astype(np.float32)
    ytr, yva = y[tr], y[va]

    binding = {
        "contract_sha256": ref.fingerprint,
        "neighbours_sha256": nbman["neighbours_sha256"],
        "neighbour_manifest": {kk: nbman[kk] for kk in ("k", "metric", "leakage_policy", "setting")},
        "source_code": source_bindings(),
        "k": args.k, "epochs": args.epochs, "seed": args.seed, "batch_size": args.batch_size,
        "loss": args.loss, "width": args.width, "heads": args.heads,
        "dropout": args.dropout, "label_smoothing": args.label_smoothing,
        "learning_rate": args.lr, "weight_decay": args.weight_decay,
        "train_ordinals_sha256": digest(tr.tolist()),
        "validation_ordinals_sha256": digest(va.tolist()),
        "torch": torch.__version__, "selection": SELECTION,
        "setting": "transductive; neighbours and their labels from fold 0 only",
    }
    report = {"method": "pnp", "counts": meta["counts"],
              "selected_counts": [len(tr), len(va)],
              "contract_sha256": ref.fingerprint, "test_evaluated": False,
              "scope": "full_cohort" if args.limit is None else "bounded_wiring_not_benchmark",
              "binding": binding}
    if not args.execute:
        print(json.dumps({k: v for k, v in report.items() if k != "binding"}))
        return report

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Occupied output: {output}")
    output.mkdir(parents=True)

    torch.set_num_threads(4)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    global feats_all
    feats_all = torch.from_numpy(Z[tr])                 # neighbours index into train rows
    feats_tr = torch.from_numpy(Z[tr])
    feats_va = torch.from_numpy(Z[va])
    nb_tr = torch.from_numpy(data["train_neighbours"][:, :args.k].astype(np.int64))
    nb_va = torch.from_numpy(data["val_neighbours"][:, :args.k].astype(np.int64))
    d_tr = torch.from_numpy(data["train_distances"][:, :args.k])
    d_va = torch.from_numpy(data["val_distances"][:, :args.k])
    yt = torch.from_numpy(ytr)

    policy = "none" if args.loss == "ce" else "sqrt_inverse"
    weights = torch.tensor(class_weights(ytr, CLASSES, policy), dtype=torch.float32)
    model = PNP(Z.shape[1], torch.from_numpy(ytr), CLASSES, args.width, args.heads,
                args.dropout, args.label_smoothing)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    report["parameters"] = sum(p.numel() for p in model.parameters())

    state = {**report, "status": "running", "pid": os.getpid(), "command": sys.argv}
    save(output / "run_manifest.json", state)
    history = []
    best = {"macro_f1": -1.0}
    start = time.monotonic()
    try:
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(feats_tr), generator=torch.Generator().manual_seed(args.seed + epoch))
            total = 0.0
            for s in range(0, len(perm), args.batch_size):
                b = perm[s:s + args.batch_size]
                idx = nb_tr[b]
                logits = model(feats_tr[b], idx, feats_all[idx], d_tr[b])
                loss = torch.nn.functional.cross_entropy(logits, yt[b], weight=weights)
                require(torch.isfinite(loss), "Nonfinite training loss")
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.detach()) * len(b)
            ll, metrics = evaluate(model, feats_va, nb_va, d_va, yva)
            _, own = evaluate(model, feats_va, nb_va, d_va, yva, no_neighbours=True)
            row = {"epoch": epoch, "train_loss": total / len(perm),
                   "validation": metrics, "own_only_validation": own}
            history.append(row)
            np.savez_compressed(output / f"validation_{epoch:03d}.npz",
                                logits=ll, y=yva, ordinals=va)
            if metrics["macro_f1"] > best["macro_f1"]:
                best = {**metrics, "epoch": epoch}
                torch.save({"model": copy.deepcopy(model.state_dict()), "epoch": epoch,
                            "binding": binding, "metrics": metrics}, output / "best.pt")
            save(output / "history.json", history)
            print(json.dumps({"epoch": epoch, "macro_f1": metrics["macro_f1"],
                              "own_only": own["macro_f1"],
                              "elapsed_s": round(time.monotonic() - start, 1)}), flush=True)

        # ---- same-setting baselines ----
        vote = neighbour_vote_probs(ytr, data["val_neighbours"], args.k)
        baselines = {"knn_vote": multiclass_metrics(yva, vote.argmax(1), vote)}
        state.update(status="completed", metrics=best, baselines=baselines,
                     elapsed_seconds=round(time.monotonic() - start, 1),
                     checkpoint_path="best.pt", metrics_path="history.json")
        save(output / "run_manifest.json", state)
        print(json.dumps({"best": best, "baselines": baselines}, indent=2))
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(output / "run_manifest.json", state)
        raise
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, default=DEFAULT)
    p.add_argument("--neighbours", type=Path, default=NEIGHBOURS)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--loss", choices=["ce", "sqrt_inverse"], default="ce")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--execute", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
