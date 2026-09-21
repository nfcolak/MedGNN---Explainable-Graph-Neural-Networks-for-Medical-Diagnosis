"""Matched EventGCHM versus XGBoost runner for ``event_graph_v1``.

This is a production entrypoint, not a claim that a run has already happened.
Both methods consume the same target-bound sample IDs, subject split and ordered
labels. XGBoost consumes a deterministic graph-level projection; EventGCHM
consumes the typed/temporal PyG graph. Test inference is forbidden unless the
caller explicitly passes ``--allow-test`` in a future final-evaluation phase.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader

from comparison.standardized.event_graph_gchm_xgb_v1.features import load_binding
from comparison.standardized.event_graph_gchm_xgb_v1.labels import (
    DEFAULT_GRAPH_ROOT, sha256,
)
from comparison.standardized.performance_review import class_weights
from event_graph_analysis.data import EventGraphArtifact
from event_graph_analysis.model import EventGCHM
from event_graph_analysis.tensorize import EventGraphTensorizer
from shared.lib.metrics import multiclass_metrics

HERE = Path(__file__).resolve().parent
DEFAULT_BINDING = DEFAULT_GRAPH_ROOT.parent / "first_recorded_lab_all_visits_v2_targets_v1"
DEFAULT_FEATURES = DEFAULT_GRAPH_ROOT.parent / "first_recorded_lab_all_visits_v2_features_v1"
DEFAULT_OUTPUT = HERE / "runs"
SEEDS = (1234, 1235, 1236)
WEIGHT_POLICIES = ("none", "sqrt_inverse")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    path.chmod(0o600)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _metric(proba: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if proba.ndim != 2 or proba.shape[0] != len(y) or not np.isfinite(proba).all():
        raise ValueError("Invalid probability matrix")
    if not np.allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Probability rows do not sum to one")
    return multiclass_metrics(y, proba.argmax(axis=1), proba)


def _selection_metric(predictions: np.ndarray, labels: np.ndarray) -> float:
    pred = np.asarray(predictions).reshape(-1, 30)
    return round(float(f1_score(labels, pred.argmax(1), average="macro", zero_division=0)), 6)


def _check_split_metadata(binding: dict[str, dict[str, Any]]) -> None:
    subjects_by_split: dict[str, set[str]] = {split: set() for split in ("train", "validation", "test")}
    for row in binding.values():
        if row["target"] >= 0:
            subjects_by_split[row["split"]].add(row["subject_id"])
    if subjects_by_split["train"] & subjects_by_split["validation"]:
        raise ValueError("Train/validation subject overlap")
    if subjects_by_split["train"] & subjects_by_split["test"]:
        raise ValueError("Train/test subject overlap")
    if subjects_by_split["validation"] & subjects_by_split["test"]:
        raise ValueError("Validation/test subject overlap")


def _feature_arrays(feature_root: Path, binding_manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    feature_root = Path(feature_root).resolve()
    manifest = _load(feature_root / "features_manifest.json")
    if manifest.get("status") != "completed" or manifest.get("schema_version") != "event_graph_xgb_features_v1":
        raise ValueError("Incomplete or incompatible XGBoost feature artifact")
    if manifest.get("binding_targets_sha256") != binding_manifest.get("artifact_files", {}).get("targets.csv"):
        raise ValueError("XGBoost features and target binding differ")
    for name, fingerprint in manifest.get("artifact_files", {}).items():
        if sha256(feature_root / name) != fingerprint:
            raise ValueError(f"Feature artifact checksum mismatch: {name}")
    arrays = np.load(feature_root / "features.npz", allow_pickle=False)
    names = json.loads((feature_root / "feature_names.json").read_text())
    X, y, folds = arrays["X"], arrays["y"], arrays["folds"]
    if X.ndim != 2 or y.shape != (len(X),) or folds.shape != (len(X),) or not np.isfinite(X).all():
        raise ValueError("Invalid feature artifact arrays")
    if len(names) != X.shape[1]:
        raise ValueError("Feature-name width mismatch")
    return X.astype(np.float32, copy=False), y.astype(np.int64, copy=False), folds.astype(np.int8, copy=False), names


def train_xgboost(
    output: Path,
    feature_root: Path = DEFAULT_FEATURES,
    binding_root: Path = DEFAULT_BINDING,
    seed: int = 1234,
    weight_policy: str = "sqrt_inverse",
    rounds: int = 1200,
    execute: bool = False,
) -> dict[str, Any]:
    """Train the matched external baseline on graph-derived features."""
    import xgboost as xgb

    binding, binding_manifest = load_binding(binding_root)
    X, y, folds, feature_names = _feature_arrays(feature_root, binding_manifest)
    if weight_policy not in WEIGHT_POLICIES or seed not in SEEDS:
        raise ValueError("Seed or weight policy is outside the frozen protocol")
    train_idx = np.flatnonzero(folds == 0)
    val_idx = np.flatnonzero(folds == 1)
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Training and validation cohorts must be nonempty")
    weights = class_weights(y[train_idx], 30, weight_policy).astype(np.float32)
    report = {
        "method": "xgboost_event_projection",
        "seed": seed,
        "weight_policy": weight_policy,
        "scope": "full_labelled_cohort",
        "counts": {"train": int(len(train_idx)), "validation": int(len(val_idx)),
                   "test": int((folds == 2).sum()), "features": int(X.shape[1])},
        "graph_feature_manifest": str(Path(feature_root).resolve() / "features_manifest.json"),
        "target_binding_manifest": str(Path(binding_root).resolve() / "binding_manifest.json"),
        "test_evaluated": False,
        "temporal_clean": False,
        "params": {"num_boost_round": rounds, "learning_rate": 0.03, "max_depth": 4,
                   "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3.0,
                   "reg_lambda": 2.0, "tree_method": "hist", "objective": "multi:softprob"},
    }
    if not execute:
        return {**report, "status": "dry_run"}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Occupied output: {output}")
    output.mkdir(parents=True)
    _set_seed(seed)
    train = xgb.DMatrix(X[train_idx], label=y[train_idx], weight=weights[y[train_idx]])
    validation = xgb.DMatrix(X[val_idx], label=y[val_idx])
    params = {
        "objective": "multi:softprob", "num_class": 30, "eta": 0.03,
        "max_depth": 4, "subsample": 0.9, "colsample_bytree": 0.9,
        "min_child_weight": 3.0, "lambda": 2.0, "tree_method": "hist",
        "seed": seed, "disable_default_eval_metric": 1,
    }

    def custom_metric(predictions, data):
        return "macro_f1", _selection_metric(predictions, data.get_label().astype(np.int64))

    history: dict[str, Any] = {}
    started = time.monotonic()
    booster = xgb.train(params, train, num_boost_round=rounds,
                        evals=[(validation, "validation")], custom_metric=custom_metric,
                        maximize=True, evals_result=history, verbose_eval=False)
    values = np.asarray(history["validation"]["macro_f1"], dtype=float)
    best = int(np.argmax(values))
    chosen = booster[:best + 1]
    proba = chosen.predict(validation)
    final_proba = booster.predict(validation)
    metrics = _metric(proba, y[val_idx])
    final_metrics = _metric(final_proba, y[val_idx])
    chosen.save_model(output / "model.ubj")
    booster.save_model(output / "final_model.ubj")
    np.savez_compressed(output / "validation.npz", proba=proba, y=y[val_idx], ordinals=val_idx)
    np.savez_compressed(output / "final_validation.npz", proba=final_proba, y=y[val_idx], ordinals=val_idx)
    _json(output / "feature_names.json", feature_names)
    _json(output / "history.json", [{"iteration": i, "validation_macro_f1": float(v)} for i, v in enumerate(values)])
    loaded = xgb.Booster()
    loaded.load_model(output / "model.ubj")
    replay = loaded.predict(validation)
    if not np.array_equal(replay, proba):
        raise ValueError("XGBoost selected checkpoint replay differs")
    state = {**report, "status": "completed", "metrics": metrics,
             "final_metrics": final_metrics, "selected_iteration": best,
             "elapsed_seconds": time.monotonic() - started,
             "artifact_files": {name: sha256(output / name) for name in (
                 "model.ubj", "final_model.ubj", "validation.npz", "final_validation.npz",
                 "feature_names.json", "history.json")}}
    _json(output / "run_manifest.json", state)
    return state


class BoundEventDataset(torch.utils.data.Dataset):
    """Lazy labelled view over graph JSONL; unmatched stays never enter a loader."""

    def __init__(self, artifact: EventGraphArtifact, offsets: list[int], targets: dict[str, dict[str, Any]], adapter):
        self.artifact = artifact
        self.offsets = tuple(offsets)
        self.targets = targets
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int):
        graph = self.artifact._read(self.offsets[index])
        row = self.targets.get(graph["sample_id"])
        if row is None or row["target"] < 0:
            raise ValueError("Dataset offset is not target-bound")
        graph = dict(graph)
        graph["target"] = int(row["target"])
        return self.adapter.transform(graph)


def _labelled_offsets(artifact: EventGraphArtifact, binding: dict[str, dict[str, Any]], split: str) -> list[int]:
    offsets = []
    for offset in artifact._offsets[split]:
        graph = artifact._read(offset)
        row = binding.get(graph["sample_id"])
        if row is None or row["target"] < 0:
            continue
        if row["split"] != split or graph["split"] != split:
            raise ValueError("Graph and target binding split mismatch")
        offsets.append(offset)
    return offsets


def _fit_event_adapter(artifact: EventGraphArtifact, binding: dict[str, dict[str, Any]],
                       train_offsets: list[int]) -> tuple[dict[str, Any], EventGraphTensorizer]:
    """Fit tensorization and PNA degree state on labelled train graphs only."""
    adapter = EventGraphTensorizer(add_reverse_edges=True)
    adapter.fit(
        artifact._read(offset)
        for offset in train_offsets
    )
    state = {
        "artifact_sha256": artifact.fingerprint,
        "labels": artifact.manifest.get("labels"),
        "fit_split": "train",
        # EventGraphArtifact.restore_adapter uses the artifact fold count as an
        # integrity check; fit_graph_count records the narrower actual scope.
        "train_count": len(artifact._offsets["train"]),
        "fit_graph_count": len(train_offsets),
        "fit_graph_scope": "labelled_train_only",
        "adapter": adapter.to_dict(),
    }
    return state, adapter


def _fit_degree_histogram(artifact: EventGraphArtifact, adapter: EventGraphTensorizer,
                          train_offsets: list[int]) -> torch.Tensor:
    """Fit PNA's in-degree histogram from the same labelled train graphs."""
    histogram = np.zeros(1, dtype=np.int64)
    for offset in train_offsets:
        data = adapter.transform(artifact._read(offset))
        degrees = np.bincount(data.edge_index[1].cpu().numpy(), minlength=data.num_nodes)
        if len(degrees) > len(histogram):
            histogram = np.pad(histogram, (0, len(degrees) - len(histogram)))
        histogram[:len(degrees)] += degrees
    if not histogram.any():
        histogram[0] = 1
    return torch.tensor(histogram, dtype=torch.float32)


@torch.no_grad()
def _predict_gchm(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probabilities.append(logits.softmax(-1).cpu().numpy())
        labels.append(batch.y.view(-1).cpu().numpy())
    if not probabilities:
        raise ValueError("Empty evaluation loader")
    return np.concatenate(probabilities), np.concatenate(labels)


def train_event_gchm(
    output: Path,
    graph_root: Path = DEFAULT_GRAPH_ROOT,
    binding_root: Path = DEFAULT_BINDING,
    seed: int = 1234,
    weight_policy: str = "sqrt_inverse",
    epochs: int = 30,
    batch_size: int = 8,
    execute: bool = False,
) -> dict[str, Any]:
    """Train EventGCHM on the same labelled graph cohort as XGBoost."""
    if weight_policy not in WEIGHT_POLICIES or seed not in SEEDS:
        raise ValueError("Seed or weight policy is outside the frozen protocol")
    binding, binding_manifest = load_binding(binding_root)
    artifact = EventGraphArtifact(graph_root)
    _check_split_metadata(binding)
    train_offsets = _labelled_offsets(artifact, binding, "train")
    val_offsets = _labelled_offsets(artifact, binding, "validation")
    test_count = sum(1 for row in binding.values() if row["split"] == "test" and row["target"] >= 0)
    report = {
        "method": "event_gchm",
        "seed": seed,
        "weight_policy": weight_policy,
        "scope": "full_labelled_cohort",
        "counts": {"train": len(train_offsets), "validation": len(val_offsets), "test": test_count},
        "graph_root": str(Path(graph_root).resolve()),
        "target_binding_manifest": str(Path(binding_root).resolve() / "binding_manifest.json"),
        "epochs": epochs, "batch_size": batch_size, "test_evaluated": False,
        "temporal_clean": False,
    }
    if not execute:
        return {**report, "status": "dry_run"}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Occupied output: {output}")
    output.mkdir(parents=True)
    _set_seed(seed)
    adapter_state, adapter = _fit_event_adapter(artifact, binding, train_offsets)
    degree_histogram = _fit_degree_histogram(artifact, adapter, train_offsets)
    model = EventGCHM(adapter.num_tokens, adapter.num_relations, 30,
                       degree_histogram=degree_histogram, hidden_dim=64,
                       relation_dim=16, num_layers=3, dropout=0.1)
    device = torch.device("cpu")
    model.to(device)
    train_ds = BoundEventDataset(artifact, train_offsets, binding, adapter)
    val_ds = BoundEventDataset(artifact, val_offsets, binding, adapter)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    train_targets = np.asarray([binding[artifact._read(offset)["sample_id"]]["target"] for offset in train_offsets], dtype=np.int64)
    weights = torch.tensor(class_weights(train_targets, 30, weight_policy), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = -1
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y.view(-1), weight=weights.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite EventGCHM loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        proba, val_y = _predict_gchm(model, val_loader, device)
        metrics = _metric(proba, val_y)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "validation": metrics}
        history.append(row)
        score = round(metrics["macro_f1"], 6)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "config": model.config,
                        "adapter": adapter_state, "epoch": epoch}, output / "best.pt")
            np.savez_compressed(output / "validation.npz", proba=proba, y=val_y)
    if best_epoch < 0:
        raise ValueError("No EventGCHM checkpoint selected")
    _json(output / "adapter.json", adapter_state)
    _json(output / "history.json", history)
    state = {**report, "status": "completed", "best_epoch": best_epoch,
             "metrics": history[best_epoch]["validation"], "elapsed_seconds": time.monotonic() - started,
             "artifact_files": {name: sha256(output / name) for name in (
                 "best.pt", "validation.npz", "adapter.json", "history.json")}}
    _json(output / "run_manifest.json", state)
    return state


def write_comparison_report(output: Path, gchm_state: dict[str, Any], xgb_state: dict[str, Any]) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    g = gchm_state["metrics"]
    x = xgb_state["metrics"]
    result = {
        "schema_version": "event_graph_gchm_xgb_comparison_v1",
        "scope": "validation_only",
        "test_evaluated": False,
        "event_gchm": g,
        "xgboost_event_projection": x,
        "delta_event_gchm_minus_xgboost": {key: float(g[key] - x[key]) for key in g if key in x},
        "counts": {"event_gchm": gchm_state["counts"], "xgboost": xgb_state["counts"]},
        "limitations": [
            "GCHM is an event-graph adaptation of the native receiver-conditioned mechanism, not the legacy 331-slot star model run unchanged.",
            "XGBoost is a non-GNN graph-level projection baseline; it receives no raw columns outside the event graph.",
            "The event graph and target binding remain temporal_clean=false; storetime is an availability proxy.",
            "Validation checkpoint selection is reported; the test fold is not opened.",
        ],
    }
    _json(output / "comparison.json", result)
    lines = [
        "# EventGCHM–XGBoost comparison",
        "",
        "Validation only; test_evaluated=false.",
        "",
        "| Method | Macro-F1 | Balanced accuracy | Accuracy |",
        "|---|---:|---:|---:|",
        f"| EventGCHM | {g['macro_f1']:.6f} | {g['balanced_acc']:.6f} | {g['accuracy']:.6f} |",
        f"| XGBoost graph projection | {x['macro_f1']:.6f} | {x['balanced_acc']:.6f} | {x['accuracy']:.6f} |",
        "",
        "The XGBoost row is an external tabular baseline, not a GNN method.",
        "The event graph is not the old native 331-slot star artifact.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    (output / "comparison.md").chmod(0o600)
    return output / "comparison.md"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPH_ROOT)
    parser.add_argument("--binding-root", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=("xgboost", "event_gchm", "both"), default="both")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--weight-policy", choices=WEIGHT_POLICIES)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=1200)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.all and (args.seed is not None or args.weight_policy is not None):
        parser.error("--all cannot be combined with --seed/--weight-policy")
    if not args.all and (args.seed is None or args.weight_policy is None):
        parser.error("Supply --all or both --seed and --weight-policy")
    if args.all:
        cells = [(weight, seed) for weight in WEIGHT_POLICIES for seed in SEEDS]
    else:
        if args.seed is None or args.weight_policy is None:
            parser.error("Supply --all or both --seed/--weight-policy")
        cells = [(args.weight_policy, args.seed)]
    for weight, seed in cells:
        root = Path(args.output_root) / f"{weight}/seed{seed}"
        results = {}
        if args.method in ("xgboost", "both"):
            results["xgboost"] = train_xgboost(root / "xgboost", args.features_root, args.binding_root,
                                                seed, weight, args.rounds, args.execute)
        if args.method in ("event_gchm", "both"):
            results["event_gchm"] = train_event_gchm(root / "event_gchm", args.graph_root,
                                                       args.binding_root, seed, weight,
                                                       args.epochs, args.batch_size, args.execute)
        if args.method == "both" and args.execute and all(state.get("status") == "completed" for state in results.values()):
            write_comparison_report(root / "comparison", results["event_gchm"], results["xgboost"])
        print(json.dumps({name: {key: value.get(key) for key in ("status", "counts", "metrics", "elapsed_seconds")}
                          for name, value in results.items()}, allow_nan=False))


if __name__ == "__main__":
    main()
