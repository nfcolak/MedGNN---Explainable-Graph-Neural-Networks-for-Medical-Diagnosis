"""XGBoost baseline on the same native ProtGNN/GSAT reference artifact.

This is not a graph model. It consumes only information already present in the
native identical-input artifact: binary concept presence for the 192 medication /
chief-complaint concept nodes plus the 132 native patient-hub fields. Test fold is
never evaluated here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np

from comparison.standardized.native_reference_v1.data import DEFAULT, Reference, digest, sha, save
from shared.lib.metrics import multiclass_metrics


PINNED_CONTRACT = "2a2566e662bf87b10d97e58700f868d8d6d6b036582ec046dd5453e6dcd9f7c0"


def _concept_names(contract: Dict[str, Any]) -> list[str]:
    names = contract["feature_names"]
    meds = [n for n in names if n.startswith("med_id[")]
    complaints = [n for n in names if n.startswith("cc_id[")]
    if len(meds) != 104 or len(complaints) != 88:
        raise ValueError("Native concept feature names changed")
    return meds + complaints


def build_tabular_matrix(artifact: Path = DEFAULT) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str], Dict[str, Any]]:
    """Return graph-level tabular features from the authoritative native artifact.

    Feature layout:
    - 0:192 binary concept-present indicators derived from node_ids.
    - 192:324 native hub fields, exactly as stored in inputs.npz.
    """
    ref = Reference(artifact, expected=PINNED_CONTRACT)
    arrays = ref.arrays
    n = int(arrays["y"].shape[0])
    concept = np.zeros((n, 192), dtype=np.float32)
    node_ids = arrays["node_ids"]
    node_ptr = arrays["node_ptr"]
    for i in range(n):
        ids = node_ids[node_ptr[i] : node_ptr[i + 1]]
        ids = ids[ids < 192]
        concept[i, ids] = 1.0
    hub = arrays["hub"].astype(np.float32, copy=True)
    if hub.shape != (74511, 132):
        raise ValueError(f"Native hub shape changed: {hub.shape}")
    X = np.concatenate([concept, hub], axis=1).astype(np.float32, copy=False)
    if X.shape != (74511, 324):
        raise ValueError(f"XGBoost tabular shape changed: {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("Nonfinite XGBoost features")
    y = arrays["y"].astype(np.int64, copy=True)
    folds = arrays["folds"].astype(np.int64, copy=True)
    feature_names = [f"concept_present[{n}]" for n in _concept_names(ref.contract)] + list(ref.contract["hub_fields"])
    meta = {
        "version": "native-xgboost-tabular-v1",
        "source_artifact": str(Path(artifact).resolve()),
        "contract_sha256": ref.fingerprint,
        "input_sha256": ref.contract["input_sha256"],
        "counts": ref.contract["counts"],
        "feature_layout": {"concept_presence": 192, "native_hub": 132, "total": 324},
        "hub_fields": list(ref.contract["hub_fields"]),
        "labels": list(ref.contract["labels"]),
        "test_evaluated": False,
    }
    return X, y, folds, feature_names, meta


def split_indices(folds: np.ndarray, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = np.flatnonzero(folds == 0)
    val_idx = np.flatnonzero(folds == 1)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        train_idx = train_idx[:limit]
        val_idx = val_idx[:limit]
    return train_idx, val_idx


def _proba_fingerprint(proba: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(proba).tobytes())
    return h.hexdigest()


def train_xgboost(
    output: Path,
    artifact: Path = DEFAULT,
    seed: int = 1234,
    limit: int | None = None,
    n_estimators: int = 1200,
    learning_rate: float = 0.03,
    max_depth: int = 4,
    early_stopping_rounds: int = 50,
    execute: bool = False,
) -> Dict[str, Any]:
    X, y, folds, feature_names, meta = build_tabular_matrix(artifact)
    train_idx, val_idx = split_indices(folds, limit)
    scope = "bounded_wiring_not_benchmark" if limit is not None else "full_cohort"
    report = {
        "method": "xgboost",
        "counts": meta["counts"],
        "selected_counts": [int(len(train_idx)), int(len(val_idx))],
        "contract_sha256": meta["contract_sha256"],
        "input_sha256": meta["input_sha256"],
        "scope": scope,
        "feature_count": int(X.shape[1]),
        "test_evaluated": False,
        "params": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "early_stopping_rounds": early_stopping_rounds,
            "seed": seed,
            "objective": "multi:softprob",
            "tree_method": "hist",
        },
    }
    if not execute:
        return report
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Occupied output: {output}")
    output.mkdir(parents=True)

    import xgboost as xgb

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=30,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=3.0,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=4,
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(
        X[train_idx],
        y[train_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        verbose=False,
    )
    proba = model.predict_proba(X[val_idx])
    if proba.shape != (len(val_idx), 30):
        raise ValueError(f"Unexpected probability shape: {proba.shape}")
    if not np.isfinite(proba).all():
        raise ValueError("Nonfinite validation probabilities")
    pred = proba.argmax(axis=1)
    metrics = multiclass_metrics(y[val_idx], pred, proba)
    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    model.save_model(output / "model.json")
    np.savez_compressed(output / "validation.npz", proba=proba, y=y[val_idx], ordinals=val_idx)
    save(output / "feature_names.json", feature_names)
    manifest = {
        **report,
        "status": "completed",
        "metrics": metrics,
        "best_iteration": None if best_iteration is None else int(best_iteration),
        "best_score": None if best_score is None else float(best_score),
        "validation_proba_sha256": _proba_fingerprint(proba),
        "xgboost_version": xgb.__version__,
        "artifact_files": {
            "model": "model.json",
            "validation": "validation.npz",
            "feature_names": "feature_names.json",
        },
    }
    save(output / "run_manifest.json", manifest)

    # Read-back verification: model reload reproduces validation probabilities exactly
    # enough for xgboost JSON serialization; do not evaluate test split.
    loaded = xgb.XGBClassifier()
    loaded.load_model(output / "model.json")
    replay = loaded.predict_proba(X[val_idx])
    np.testing.assert_allclose(replay, proba, rtol=0, atol=1e-7)
    save(output / "replay.json", {"validation_count": int(len(val_idx)), "test_evaluated": False, "max_abs_diff": float(np.max(np.abs(replay - proba)))})
    return manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, default=DEFAULT)
    p.add_argument("--output", type=Path, default=Path("comparison/standardized/native_runs/xgboost_v1"))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-estimators", type=int, default=1200)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--early-stopping-rounds", type=int, default=50)
    p.add_argument("--execute", action="store_true")
    return p


def main(argv: Iterable[str] | None = None) -> Dict[str, Any]:
    args = parser().parse_args(argv)
    result = train_xgboost(
        output=args.output,
        artifact=args.artifact,
        seed=args.seed,
        limit=args.limit,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        early_stopping_rounds=args.early_stopping_rounds,
        execute=args.execute,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
