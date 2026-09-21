"""Artifact-only pilot analysis: GCHM-sqrt control vs concept-dropout candidate.

Run with ``python -m comparison.standardized.gchm_dropout_pilot_v1.analyse``.

Single decision variable: ``concept_dropout`` 0.0 -> 0.15. Everything else
(artifact, split, label order, seed, epochs, batch size, sqrt-inverse loss,
optimizer, learning rate, weight decay, topology) is frozen and re-verified
from the run bindings before any number is reported.

This module never instantiates a model and never touches fold 2 (test).
It reads completed run manifests, histories and saved validation logits only.
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
RUNS = REPO / "comparison/standardized/native_runs/gchm_dropout_pilot_v1"
SELECTION = "validation macro_f1; full fixed epoch budget; no early stop"
BUDGET = 30
# Frozen before inspecting any candidate metric.
FROZEN = {
    "decision_variable": "concept_dropout",
    "control": {"method": "gchm", "concept_dropout": 0.0},
    "candidate": {"method": "gchm_concept_dropout", "concept_dropout": 0.15},
    "shared": {"seed": 1234, "epochs": BUDGET, "batch_size": 128, "loss": "sqrt_inverse"},
    "primary_metric": "macro_f1",
    "secondary": ["balanced_acc", "accuracy", "micro_f1", "top3_acc", "top5_acc"],
    "rare_class_watch": "per-class recall, precision and false positives reported together",
    "test_evaluated": False,
}
# Classes named in the matched GCHM/XGBoost report as precision-cost cases.
WATCH = ("Non-ST elevation (NSTEMI) myocardial infarction", "Unsp intestnl obst")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def context(ref):
    """Train/validation only. Fold 2 is never requested."""
    tr, va = ref.fold(0), ref.fold(1)
    require(len(tr) > 0 and len(va) > 0, "Empty train/validation cohort")
    labels = ref.contract["labels"]
    y = ref.arrays["y"][va]
    require(np.all((y >= 0) & (y < len(labels))), "Invalid validation labels")
    return {"train": tr, "validation": va, "y": y, "labels": labels,
            "train_support": np.bincount(ref.arrays["y"][tr], minlength=len(labels)),
            "validation_support": np.bincount(y, minlength=len(labels)),
            "train_ordinals_sha256": digest(tr.tolist()),
            "validation_ordinals_sha256": digest(va.tolist())}


def first_maximum(history):
    require(len(history) == BUDGET, "Incomplete fixed-budget history")
    require([r["epoch"] for r in history] == list(range(BUDGET)), "History order/gaps/duplicates")
    values = np.array([r["validation"]["macro_f1"] for r in history], dtype=float)
    require(np.isfinite(values).all(), "Nonfinite selection metric")
    return history[int(np.argmax(np.round(values, 6)))]


def load_run(path, expected, ctx, ref):
    """Validate one completed native run and return its selected-epoch predictions."""
    import torch

    path = Path(path)
    m = json.loads((path / "run_manifest.json").read_text())
    require(m["status"] == "completed", f"Incomplete run: {path.name}")
    require(m["test_evaluated"] is False, "Run evaluated test")
    require(m["contract_sha256"] == ref.fingerprint == PINNED_CONTRACT, "Contract mismatch")
    require(m["scope"] == "full_cohort", "Not a full-cohort run")
    require(m["selected_counts"] == [len(ctx["train"]), len(ctx["validation"])], "Partial cohort")
    b = m["binding"]
    require(b["method"] == expected["method"], f"Method mismatch: {b['method']}")
    for key, value in FROZEN["shared"].items():
        require(b[key] == value, f"Frozen setting changed: {key}={b[key]!r}")
    require(b["limit"] is None, "Bounded wiring run is not benchmark evidence")
    require(b["weight_policy"] == "sqrt_inverse", "sqrt loss weighting changed")
    require(b["selection"] == SELECTION, "Selection policy changed")
    require(b["train_ordinals_sha256"] == ctx["train_ordinals_sha256"], "Train cohort mismatch")
    require(b["validation_ordinals_sha256"] == ctx["validation_ordinals_sha256"],
            "Validation cohort mismatch")
    with np.load(path / "cohort.npz", allow_pickle=False) as z:
        require(np.array_equal(z["train_ordinals"], ctx["train"]), "Train ordinals mismatch")
        require(np.array_equal(z["validation_ordinals"], ctx["validation"]),
                "Validation ordinals mismatch")
    history = json.loads((path / "history.json").read_text())
    selected = first_maximum(history)
    epoch = selected["epoch"]
    proof = json.loads((path / "replay.json").read_text())
    require(proof == m["replay"], "Replay file/manifest mismatch")
    require(proof["exact_logits"] is True and proof["selected_epoch"] == epoch,
            "Missing exact replay of the selected checkpoint")
    require(sha(path / "best.pt") == proof["checkpoint_sha256"], "Checkpoint SHA mismatch")
    with np.load(path / f"validation_{epoch:03d}.npz", allow_pickle=False) as z:
        logits, y, ordinals = z["logits"], z["y"], z["ordinals"]
    require(np.array_equal(ordinals, ctx["validation"]), "Prediction order mismatch")
    require(np.array_equal(y, ctx["y"]), "Prediction label mismatch")
    require(np.isfinite(logits).all(), "Nonfinite validation logits")
    p = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    metrics = multiclass_metrics(ctx["y"], p.argmax(1), p)
    require(metrics == selected["validation"], "Recomputed selected metrics mismatch")
    return {"path": str(path.relative_to(REPO)), "binding": b, "history": history,
            "selected_epoch": epoch, "selected": metrics,
            "final": history[-1]["validation"],
            "train_loss": [r["train_loss"] for r in history],
            "macro_f1_curve": [r["validation"]["macro_f1"] for r in history],
            "parameters": m["parameters"], "pred": p.argmax(1)}


def binding_delta(control, candidate):
    """Prove exactly one binding field differs, and that it is the source hash set."""
    a, b = control["binding"], candidate["binding"]
    differing = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    require(differing == ["method"],
            f"Only the method name may differ; observed: {differing}")
    files = sorted(set(a["source_code"]) | set(b["source_code"]))
    changed = [f for f in files if a["source_code"].get(f) != b["source_code"].get(f)]
    require(changed == [], f"Source tree changed between the two runs: {changed}")
    return {"differing_binding_fields": differing, "identical_source_hashes": True,
            "source_file_count": len(files)}


def per_class(y, pred, labels):
    rows = []
    for c, name in enumerate(labels):
        truth = y == c
        hit = pred == c
        tp = int((truth & hit).sum())
        fp = int((~truth & hit).sum())
        fn = int((truth & ~hit).sum())
        rows.append({"class": c, "name": name, "support": int(truth.sum()),
                     "tp": tp, "fp": fp, "fn": fn,
                     "recall": round(tp / truth.sum(), 6) if truth.sum() else 0.0,
                     "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
                     "f1": round(2 * tp / (2 * tp + fp + fn), 6) if tp else 0.0})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--artifact", type=Path, default=DEFAULT)
    ap.add_argument("--output", type=Path, default=HERE / "report.json")
    args = ap.parse_args(argv)

    ref = Reference(args.artifact, expected=PINNED_CONTRACT)
    ctx = context(ref)
    control = load_run(args.runs / "control_seed1234", FROZEN["control"], ctx, ref)
    candidate = load_run(args.runs / "dropout_seed1234", FROZEN["candidate"], ctx, ref)
    require(control["parameters"] == candidate["parameters"],
            "Capacity changed; this is no longer a regularization-only comparison")

    cc = per_class(ctx["y"], control["pred"], ctx["labels"])
    dc = per_class(ctx["y"], candidate["pred"], ctx["labels"])
    deltas = [{"name": a["name"], "support": a["support"],
               "recall_control": a["recall"], "recall_candidate": b["recall"],
               "precision_control": a["precision"], "precision_candidate": b["precision"],
               "fp_control": a["fp"], "fp_candidate": b["fp"],
               "tp_control": a["tp"], "tp_candidate": b["tp"],
               "f1_delta": round(b["f1"] - a["f1"], 6)}
              for a, b in zip(cc, dc)]
    rare = sorted(deltas, key=lambda r: r["support"])[:10]

    report = {
        "protocol": FROZEN,
        "contract_sha256": ref.fingerprint,
        "validation_count": int(len(ctx["y"])),
        "parameters": control["parameters"],
        "parity": binding_delta(control, candidate),
        "control": {k: control[k] for k in
                    ("path", "selected_epoch", "selected", "final", "macro_f1_curve", "train_loss")},
        "candidate": {k: candidate[k] for k in
                      ("path", "selected_epoch", "selected", "final", "macro_f1_curve", "train_loss")},
        "headline": {
            "macro_f1_selected_delta": round(candidate["selected"]["macro_f1"]
                                             - control["selected"]["macro_f1"], 6),
            "macro_f1_final_delta": round(candidate["final"]["macro_f1"]
                                          - control["final"]["macro_f1"], 6),
            "selected_minus_final_control": round(control["selected"]["macro_f1"]
                                                  - control["final"]["macro_f1"], 6),
            "selected_minus_final_candidate": round(candidate["selected"]["macro_f1"]
                                                    - candidate["final"]["macro_f1"], 6),
            "total_false_positives_control": sum(r["fp"] for r in cc),
            "total_false_positives_candidate": sum(r["fp"] for r in dc),
        },
        "watch_classes": [r for r in deltas if r["name"] in WATCH],
        "rarest_ten_classes": rare,
        "per_class": deltas,
        "limitations": [
            "Single seed (1234). One run is not sufficient to replace the incumbent.",
            "Checkpoint selection and reporting share the same validation split; no held-out"
            " estimate of the selection gain is available here.",
            "No confidence intervals in this pilot; per-class counts are point estimates.",
            "test_evaluated=false; fold 2 was never read.",
            "concept_dropout=0.15 is a prespecified candidate value, not a tuned optimum.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"parity": report["parity"], "headline": report["headline"],
                      "control_selected": control["selected"],
                      "candidate_selected": candidate["selected"],
                      "control_final": control["final"],
                      "candidate_final": candidate["final"],
                      "watch_classes": report["watch_classes"],
                      "output": str(args.output.relative_to(REPO))}, indent=2))
    return report


if __name__ == "__main__":
    main()
