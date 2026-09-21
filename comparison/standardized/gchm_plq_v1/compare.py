"""Artifact-only comparison: GCHM-PLQ vs GCHM incumbent vs XGBoost.

Run with ``python -m comparison.standardized.gchm_plq_v1.compare``.

Never instantiates a model and never reads fold 2. Every number comes from a
completed run manifest plus its saved validation predictions, re-verified here.

Three-way, matched on seed, split, label order, sqrt-inverse weighting and the
pinned native contract. The compute budgets are NOT matched (GCHM 30 epochs,
XGBoost 1200 boosting rounds); that asymmetry is inherited from the existing
matched protocol and is reported, not hidden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from comparison.standardized.native_reference_v1.data import (
    DEFAULT, PINNED_CONTRACT, Reference, digest, sha,
)
from shared.lib.metrics import multiclass_metrics

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NATIVE = REPO / "comparison/standardized/native_runs"
MATCHED = REPO / "comparison/standardized/matched_gchm_xgb_v1/runs"
SEEDS = (1234, 1235, 1236)
BUDGET = 30
GCHM_SELECTION = "validation macro_f1; full fixed epoch budget; no early stop"
METRICS = ("macro_f1", "balanced_acc", "accuracy", "micro_f1", "top3_acc", "top5_acc")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def context(ref):
    tr, va = ref.fold(0), ref.fold(1)
    labels = ref.contract["labels"]
    y = ref.arrays["y"][va]
    require(len(va) > 0 and np.all((y >= 0) & (y < len(labels))), "Invalid validation cohort")
    return {"train": tr, "validation": va, "y": y, "labels": labels,
            "train_ordinals_sha256": digest(tr.tolist()),
            "validation_ordinals_sha256": digest(va.tolist())}


def load_native(path, method, seed, ctx, ref):
    """Validate a completed GCHM-family run and return selected-epoch probabilities."""
    import torch

    path = Path(path)
    m = json.loads((path / "run_manifest.json").read_text())
    require(m["status"] == "completed", f"Incomplete run: {path}")
    require(m["test_evaluated"] is False, "Run evaluated test")
    require(m["scope"] == "full_cohort", "Not a full-cohort run")
    require(m["contract_sha256"] == ref.fingerprint == PINNED_CONTRACT, "Contract mismatch")
    require(m["selected_counts"] == [len(ctx["train"]), len(ctx["validation"])], "Partial cohort")
    b = m["binding"]
    require(b["method"] == method, f"Method mismatch: {b['method']} != {method}")
    require(b["seed"] == seed, "Seed mismatch")
    require(b["epochs"] == BUDGET and b["batch_size"] == 128, "Budget/batch mismatch")
    require(b["loss"] == "sqrt_inverse" and b["weight_policy"] == "sqrt_inverse", "Loss mismatch")
    require(b["limit"] is None, "Bounded wiring run is not benchmark evidence")
    require(b["selection"] == GCHM_SELECTION, "Selection policy changed")
    require(b["train_ordinals_sha256"] == ctx["train_ordinals_sha256"], "Train cohort mismatch")
    require(b["validation_ordinals_sha256"] == ctx["validation_ordinals_sha256"],
            "Validation cohort mismatch")
    if method == "gchm_boost":
        raise ValueError("gchm_boost was withdrawn: consuming XGBoost predictions "
                         "makes the comparison a hybrid, not a rival architecture")
    history = json.loads((path / "history.json").read_text())
    require(len(history) == BUDGET and [r["epoch"] for r in history] == list(range(BUDGET)),
            "Incomplete or disordered history")
    values = np.array([r["validation"]["macro_f1"] for r in history], dtype=float)
    epoch = int(np.argmax(np.round(values, 6)))
    proof = json.loads((path / "replay.json").read_text())
    require(proof == m["replay"] and proof["exact_logits"] is True, "Missing exact replay")
    require(proof["selected_epoch"] == epoch, "Replay/selection disagreement")
    require(sha(path / "best.pt") == proof["checkpoint_sha256"], "Checkpoint SHA mismatch")
    with np.load(path / f"validation_{epoch:03d}.npz", allow_pickle=False) as z:
        logits, y, ordinals = z["logits"], z["y"], z["ordinals"]
    require(np.array_equal(ordinals, ctx["validation"]) and np.array_equal(y, ctx["y"]),
            "Prediction cohort/order mismatch")
    require(np.isfinite(logits).all(), "Nonfinite logits")
    p = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    metrics = multiclass_metrics(ctx["y"], p.argmax(1), p)
    require(metrics == history[epoch]["validation"], "Recomputed metrics mismatch")
    return {"proba": p, "metrics": metrics, "selected_epoch": epoch, "binding": b,
            "parameters": m["parameters"], "final": history[-1]["validation"],
            "curve": values.tolist(), "train_loss": [r["train_loss"] for r in history]}


def load_xgboost(path, seed, ctx):
    path = Path(path)
    m = json.loads((path / "run_manifest.json").read_text())
    require(m["status"] == "completed" and m["test_evaluated"] is False, "Bad XGBoost run")
    require(m["contract_sha256"] == PINNED_CONTRACT, "Contract mismatch")
    b = m["binding"]
    require(b["seed"] == seed and b["weight_policy"] == "sqrt_inverse", "Seed/weight mismatch")
    require(b["train_ordinals_sha256"] == ctx["train_ordinals_sha256"], "Train cohort mismatch")
    require(b["validation_ordinals_sha256"] == ctx["validation_ordinals_sha256"],
            "Validation cohort mismatch")
    with np.load(path / "validation.npz", allow_pickle=False) as z:
        proba, y, ordinals = z["proba"], z["y"], z["ordinals"]
    require(np.array_equal(ordinals, ctx["validation"]) and np.array_equal(y, ctx["y"]),
            "XGBoost cohort/order mismatch")
    metrics = multiclass_metrics(ctx["y"], proba.argmax(1), proba)
    require(metrics == m["metrics"], "Recomputed XGBoost metrics mismatch")
    return {"proba": proba, "metrics": metrics, "rounds": b["rounds"],
            "selected_iteration": m["selected_iteration"]}


def summarize(runs):
    out = {}
    for k in METRICS:
        v = np.array([r["metrics"][k] for r in runs], dtype=float)
        out[k] = {"mean": round(float(v.mean()), 6),
                  "sample_sd": round(float(v.std(ddof=1)), 6) if len(v) > 1 else 0.0,
                  "per_seed": [round(float(x), 6) for x in v]}
    return out


def ensemble(runs, ctx):
    """Mean of the per-seed probability simplex, same rule for every method."""
    p = np.mean([r["proba"] for r in runs], axis=0)
    p = p / p.sum(1, keepdims=True)
    return multiclass_metrics(ctx["y"], p.argmax(1), p), p


def per_class(y, pred, labels):
    rows = []
    for c, name in enumerate(labels):
        truth, hit = y == c, pred == c
        tp = int((truth & hit).sum()); fp = int((~truth & hit).sum()); fn = int((truth & ~hit).sum())
        rows.append({"name": name, "support": int(truth.sum()), "tp": tp, "fp": fp, "fn": fn,
                     "recall": round(tp / truth.sum(), 6) if truth.sum() else 0.0,
                     "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
                     "f1": round(2 * tp / (2 * tp + fp + fn), 6) if tp else 0.0})
    return rows


def paired_delta(a_runs, b_runs):
    """Per-seed paired differences: the seeds are matched, so pair before averaging."""
    out = {}
    for k in METRICS:
        d = np.array([x["metrics"][k] - y["metrics"][k] for x, y in zip(a_runs, b_runs)])
        out[k] = {"mean": round(float(d.mean()), 6),
                  "sample_sd": round(float(d.std(ddof=1)), 6) if len(d) > 1 else 0.0,
                  "per_seed": [round(float(v), 6) for v in d],
                  "wins": int((d > 0).sum()), "n": len(d)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=Path, default=DEFAULT)
    ap.add_argument("--output", type=Path, default=HERE / "report.json")
    args = ap.parse_args(argv)

    ref = Reference(args.artifact, expected=PINNED_CONTRACT)
    ctx = context(ref)

    plq = [load_native(NATIVE / f"gchm_plq_v1/seed{s}", "gchm_plq", s, ctx, ref) for s in SEEDS]
    gchm = [load_native(NATIVE / f"gchm_sqrt_v2/seed{s}", "gchm", s, ctx, ref) for s in SEEDS]
    xgb = [load_xgboost(MATCHED / f"sqrt_inverse/seed{s}", s, ctx) for s in SEEDS]

    # Parity: the two GCHM arms may differ only by method name.
    for a, b in zip(plq, gchm):
        differing = sorted(k for k in set(a["binding"]) | set(b["binding"])
                           if a["binding"].get(k) != b["binding"].get(k))
        require(differing == ["hub_encoder", "method"],
                f"Unexpected binding differences between GCHM arms: {differing}")
        changed = [f for f in set(a["binding"]["source_code"]) | set(b["binding"]["source_code"])
                   if a["binding"]["source_code"].get(f) != b["binding"]["source_code"].get(f)]
        require(changed == [], f"Source tree differs between arms: {changed}")

    plq_ens, plq_p = ensemble(plq, ctx)
    gchm_ens, gchm_p = ensemble(gchm, ctx)
    xgb_ens, xgb_p = ensemble(xgb, ctx)

    labels = ctx["labels"]
    pc_plq = per_class(ctx["y"], plq_p.argmax(1), labels)
    pc_xgb = per_class(ctx["y"], xgb_p.argmax(1), labels)
    pc_gchm = per_class(ctx["y"], gchm_p.argmax(1), labels)

    report = {
        "scope": "validation_only_matched_seeds",
        "contract_sha256": ref.fingerprint,
        "validation_count": int(len(ctx["y"])),
        "seeds": list(SEEDS),
        "test_evaluated": False,
        "encoder": plq[0]["binding"]["hub_encoder"],
        "parameters": {"gchm_plq": plq[0]["parameters"], "gchm": gchm[0]["parameters"]},
        "single_model": {
            "gchm_plq": summarize(plq), "gchm": summarize(gchm), "xgboost": summarize(xgb),
        },
        "ensemble": {"gchm_plq": plq_ens, "gchm": gchm_ens, "xgboost": xgb_ens},
        "paired_delta": {
            "plq_minus_gchm": paired_delta(plq, gchm),
            "plq_minus_xgboost": paired_delta(plq, xgb),
            "gchm_minus_xgboost": paired_delta(gchm, xgb),
        },
        "selected_epochs": {"gchm_plq": [r["selected_epoch"] for r in plq],
                            "gchm": [r["selected_epoch"] for r in gchm]},
        "xgboost_budget": {"rounds": xgb[0]["rounds"],
                           "selected_iteration": [r["selected_iteration"] for r in xgb]},
        "ensemble_per_class": [
            {"name": a["name"], "support": a["support"],
             "plq_recall": a["recall"], "xgb_recall": b["recall"], "gchm_recall": c["recall"],
             "plq_precision": a["precision"], "xgb_precision": b["precision"],
             "plq_fp": a["fp"], "xgb_fp": b["fp"],
             "plq_minus_xgb_f1": round(a["f1"] - b["f1"], 6)}
            for a, b, c in zip(pc_plq, pc_xgb, pc_gchm)
        ],
        "limitations": [
            "Validation only; test_evaluated=false on every run.",
            "Compute budgets are not matched: GCHM 30 epochs vs XGBoost 1200 boosting rounds.",
            "Checkpoint selection and reporting share the validation split for all methods.",
            "Three seeds; sample sd over three runs is a weak dispersion estimate.",
            "No confidence intervals or multiple-comparison correction in this report.",
            "Upstream temporal_clean=false limitations of the pinned artifact are unchanged.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    brief = {
        "single_model_macro_f1": {k: report["single_model"][k]["macro_f1"]
                                  for k in ("gchm_plq", "gchm", "xgboost")},
        "ensemble_macro_f1": {k: report["ensemble"][k]["macro_f1"]
                              for k in ("gchm_plq", "gchm", "xgboost")},
        "paired_delta_macro_f1": {k: report["paired_delta"][k]["macro_f1"]
                                  for k in report["paired_delta"]},
        "output": str(args.output.relative_to(REPO)),
    }
    print(json.dumps(brief, indent=2))
    return report


if __name__ == "__main__":
    main()
