"""AUROC / AUPRC for a finished run, from the probabilities it already saved.

The task description asks for clinical predictive quality as AUROC/AUPRC; the run
reports accuracy / F1 / top-k only. `validation.npz` holds the probabilities and
labels, so nothing is retrained. One-vs-rest per class, macro-averaged over the
classes that are defined on this fold (at least one positive and one negative), both
plain and patient-equal (each visit weighted 1 / that patient's evaluated visits, the
same weighting as the run's own headline metric).

    python3 -m core.discrimination --run <run dir> --out <NEW json path>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def patient_equal_weights(subjects) -> np.ndarray:
    counts = Counter(subjects.tolist())
    return np.asarray([1.0 / counts[s] for s in subjects.tolist()], dtype=float)


def discrimination_report(proba: np.ndarray, y: np.ndarray, subjects=None) -> dict:
    proba, y = np.asarray(proba, dtype=float), np.asarray(y).astype(int)
    if proba.ndim != 2 or proba.shape[0] != y.shape[0]:
        raise ValueError("proba must be [rows, classes] and aligned with y")
    weights = None if subjects is None else patient_equal_weights(np.asarray(subjects))
    per_class, undefined = {}, []
    for c in range(proba.shape[1]):
        positive = (y == c).astype(int)
        if positive.sum() == 0 or positive.sum() == positive.size:
            undefined.append(c)
            continue
        entry = {"positives": int(positive.sum()),
                 "auroc": float(roc_auc_score(positive, proba[:, c])),
                 "auprc": float(average_precision_score(positive, proba[:, c]))}
        if weights is not None:
            entry["auroc_patient_equal"] = float(roc_auc_score(positive, proba[:, c], sample_weight=weights))
            entry["auprc_patient_equal"] = float(average_precision_score(positive, proba[:, c], sample_weight=weights))
        per_class[c] = entry
    if not per_class:
        raise ValueError("no class is defined on this fold")
    keys = ["auroc", "auprc"] + (["auroc_patient_equal", "auprc_patient_equal"] if weights is not None else [])
    return {
        "rows": int(y.size),
        "classes_defined": sorted(per_class),
        "classes_undefined": undefined,
        "macro": {k: float(np.mean([v[k] for v in per_class.values()])) for k in keys},
        "per_class": {str(c): v for c, v in per_class.items()},
        "note": "one-vs-rest, macro over classes defined on this fold; patient-equal = each "
                "visit weighted 1/patient's evaluated visits",
    }


def run_report(run_dir) -> dict:
    path = Path(run_dir) / "validation.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing (a run that never scored validation has no AUROC to report)")
    data = np.load(path, allow_pickle=False)
    return discrimination_report(data["proba"], data["y"], data["subjects"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    report = run_report(args.run)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["macro"]))


if __name__ == "__main__":
    main()
