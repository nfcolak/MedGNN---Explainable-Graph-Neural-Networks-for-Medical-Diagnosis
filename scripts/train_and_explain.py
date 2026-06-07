#train_and_explain.py
"""
ProtGNN — Train & Explain with GraphXAI
========================================
This script trains the ProtGNN model on the configured dataset, then runs
three GraphXAI explainers on the test set and saves every artefact under
the ``outputs/results/`` directory.

Output layout
-------------
outputs/results/
├── model_config.json          hyperparameters used for this run
├── training_metrics.csv       per-epoch train / eval metrics
├── test_metrics.json          final test-set performance
├── explanations/
│   ├── graph_<i>.json         per-graph node-importance for all explainers
│   └── explanations_summary.csv  min / max / mean per graph per explainer
└── report.txt                 human-readable end-to-end summary

Usage
-----
    python scripts/train_and_explain.py [--clst 0.1] [--sep 0.1] [--explain_n 10] [--no_prot]

Arguments
---------
--clst       cluster-loss weight   (default: 0.1)
--sep        separation-loss weight (default: 0.1)
--explain_n  number of test graphs to explain (default: 10, -1 = all)
--no_prot    disable prototype layers; train as a standard GCN / GIN
             (recommended for stable convergence — prototype training
             requires careful warm-up that can cause loss explosion)
"""

import os
import sys
import json
import csv
import argparse
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Data

# ---------------------------------------------------------------------------
# Path setup — make GraphXAI importable
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_EXTERNAL_GRAPHXAI_ROOT = os.path.join(_PROJECT_ROOT, "external", "GraphXAI-main")

for _path in (_PROJECT_ROOT, _SRC_DIR, _EXTERNAL_GRAPHXAI_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ---------------------------------------------------------------------------
# ProtGNN imports
# ---------------------------------------------------------------------------
from configs.config import OUTPUTS_DIR, data_args, model_args, train_args
from prot_gnn.models import GnnNets
from prot_gnn.explainability.graphxai_wrapper import ProtGNNWrapper
from prot_gnn.load_dataset import get_dataset, get_dataloader
from prot_gnn.my_mcts import mcts
from archive_results import archive_results

# ---------------------------------------------------------------------------
# GraphXAI imports
# ---------------------------------------------------------------------------
try:
    from graphxai.explainers.grad import GradExplainer
    from graphxai.explainers.integrated_grad import IntegratedGradExplainer
    from graphxai.explainers.gnn_explainer import GNNExplainer
    GRAPHXAI_OK = True
except Exception as _e:
    print(f"[WARNING] GraphXAI could not be imported: {_e}")
    GRAPHXAI_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_BASE = os.path.join(str(OUTPUTS_DIR), "results")  # fixed parent dir
# RESULTS_DIR is redirected to a timestamped per-run subfolder by
# _prepare_results_dir() so runs never overwrite each other.
RESULTS_DIR = RESULTS_BASE
EXPLAINER_NAMES = ["GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]
FEATURE_TOP_K = 10
NODE_TOP_K = 5


# ===========================================================================
# Helpers
# ===========================================================================

def _mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _prepare_results_dir():
    """Redirect RESULTS_DIR to a fresh, timestamped + titled per-run subfolder.
    NEVER deletes previous runs — each run is self-contained under its own
    folder, e.g.  outputs/results/2026-06-02_17-30_disease_gcn/ ."""
    global RESULTS_DIR
    # small, readable title from the dataset + model
    title = (data_args.dataset_name
             .replace("mimic_intra_patient_", "")
             .replace("mimic_patient_sim_", "patsim_")
             .replace("mimic_", ""))
    stamp = time.strftime("%Y-%m-%d_%H-%M")
    RESULTS_DIR = os.path.join(RESULTS_BASE, f"{stamp}_{title}_{model_args.model_name}")
    _mkdir(RESULTS_DIR)
    _mkdir(os.path.join(RESULTS_DIR, "explanations"))
    # leave a pointer to the most recent run so summary scripts can find it
    try:
        with open(os.path.join(RESULTS_BASE, "latest_run.txt"), "w") as f:
            f.write(RESULTS_DIR)
    except Exception:
        pass
    print(f"  Run results dir: {RESULTS_DIR}")


def _compute_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    nonzero = counts > 0
    weights = np.ones(num_classes, dtype=np.float32)
    weights[nonzero] = len(labels) / (num_classes * counts[nonzero])
    return torch.tensor(weights, dtype=torch.float32, device=model_args.device), counts


def _average_precision(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y_true.sum()
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _binary_metrics(labels, preds, pos_scores):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    spec      = tn / max(tn + fp, 1)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 6),
        "recall":    round(recall,    6),
        "specificity": round(spec,    6),
        "f1":        round(f1,        6),
        "balanced_acc": round((recall + spec) / 2, 6),
        "pr_auc":    round(_average_precision(labels, pos_scores), 6),
    }


def _multiclass_metrics(labels, preds, probs):
    """Multi-class (e.g. 30-disease) metrics: accuracy is added by the caller;
    here we add balanced accuracy, macro/micro-F1, and top-3/top-5 accuracy."""
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 top_k_accuracy_score)
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    out = {
        "balanced_acc": round(float(balanced_accuracy_score(labels, preds)), 6),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 6),
        "micro_f1": round(float(f1_score(labels, preds, average="micro", zero_division=0)), 6),
    }
    if probs is not None and probs.ndim == 2:
        n_cls = probs.shape[1]
        lab_range = np.arange(n_cls)
        for k in (3, 5):
            if n_cls > k:
                out[f"top{k}_acc"] = round(float(
                    top_k_accuracy_score(labels, probs, k=k, labels=lab_range)), 6)
    return out


def _confusion_counts(labels, preds):
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    return {
        "tn": int(np.sum((preds == 0) & (labels == 0))),
        "fp": int(np.sum((preds == 1) & (labels == 0))),
        "fn": int(np.sum((preds == 0) & (labels == 1))),
        "tp": int(np.sum((preds == 1) & (labels == 1))),
    }


def _label_distribution(labels, num_classes=None):
    labels = np.asarray(labels, dtype=np.int64)
    if num_classes is None:
        num_classes = int(labels.max()) + 1 if labels.size else 0
    counts = np.bincount(labels, minlength=num_classes).astype(np.int64)
    total = int(counts.sum())
    return {
        f"class_{idx}": {
            "count": int(count),
            "rate": round(float(count / total), 6) if total else 0.0,
        }
        for idx, count in enumerate(counts)
    }


def _threshold_sensitivity(labels, pos_scores, thresholds):
    rows = []
    for threshold in sorted({round(float(t), 6) for t in thresholds}):
        preds = (pos_scores >= threshold).astype(np.int64)
        metrics = _binary_metrics(labels, preds, pos_scores)
        row = {
            "threshold": threshold,
            **_confusion_counts(labels, preds),
            "predicted_admitted": int(np.sum(preds == 1)),
            "predicted_home": int(np.sum(preds == 0)),
            **metrics,
        }
        rows.append(row)
    return rows


def _collect_predictions(dataloader, model, criterion):
    model.eval()
    losses, accs, labels_all, preds_all, probs_all = [], [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            logits, probs, _, _, _ = model(batch)
            loss = criterion(logits, batch.y)
            _, pred = torch.max(logits, -1)
            losses.append(loss.item())
            accs.append(pred.eq(batch.y).cpu().numpy())
            labels_all.append(batch.y.cpu())
            preds_all.append(pred.cpu())
            probs_all.append(probs.cpu())
    return (
        float(np.mean(losses)),
        torch.cat(labels_all).numpy(),
        torch.cat(preds_all).numpy(),
        torch.cat(probs_all).numpy(),
    )


def _state_from_predictions(loss, labels_all, argmax_preds, probs_all, threshold=None):
    preds_all = argmax_preds
    threshold_used = None
    if threshold is not None and probs_all.ndim == 2 and probs_all.shape[1] == 2:
        preds_all = (probs_all[:, 1] >= threshold).astype(np.int64)
        threshold_used = float(threshold)

    state = {
        "loss": round(loss, 6),
        "acc":  round(float((preds_all == labels_all).mean()), 6),
    }
    if probs_all.ndim == 2 and probs_all.shape[1] == 2:
        state.update(_binary_metrics(labels_all, preds_all, probs_all[:, 1]))
        state["confusion_matrix"] = _confusion_counts(labels_all, preds_all)
        if threshold_used is not None:
            state["threshold"] = round(threshold_used, 6)
    else:
        # multi-class (e.g. disease, 30 classes)
        state.update(_multiclass_metrics(labels_all, preds_all, probs_all))
    return state


def _find_best_threshold(labels, pos_scores, objective="balanced_acc"):
    labels = np.asarray(labels, dtype=np.int64)
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    best_threshold = 0.5
    best_score = float("-inf")
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (pos_scores >= threshold).astype(np.int64)
        metrics = _binary_metrics(labels, preds, pos_scores)
        score = metrics.get(objective, metrics["f1"])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def _evaluate(dataloader, model, criterion, threshold=None):
    loss, labels_all, argmax_preds, probs_all = _collect_predictions(dataloader, model, criterion)
    return _state_from_predictions(loss, labels_all, argmax_preds, probs_all, threshold=threshold)


def _tensor_to_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().view(-1).tolist()
    return value


def _first_or_none(value):
    values = _tensor_to_list(value)
    if values is None or len(values) == 0:
        return None
    return int(values[0])


def _top_items(values, names=None, k=10, use_abs=True):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    order_values = np.abs(arr) if use_abs else arr
    order = np.argsort(-order_values)[: min(k, len(arr))]
    items = []
    for idx in order:
        item = {
            "index": int(idx),
            "score": float(arr[idx]),
            "abs_score": float(abs(arr[idx])),
        }
        if names is not None and idx < len(names):
            item["name"] = names[idx]
        items.append(item)
    return items


def _grad_feature_attribution(wrapper, x, edge_index, label, forward_kwargs, criterion):
    x_attr = x.clone().detach().requires_grad_(True)
    output = wrapper(x_attr, edge_index, **forward_kwargs)
    loss = criterion(output, label)
    wrapper.zero_grad(set_to_none=True)
    loss.backward()
    return x_attr.grad.detach()


def _integrated_grad_feature_attribution(wrapper, x, edge_index, label, forward_kwargs, criterion, steps=40):
    baseline = torch.zeros_like(x)
    grads = torch.zeros(steps + 1, *x.shape, device=x.device)
    for step in range(steps + 1):
        temp_x = (baseline + (float(step) / steps) * (x - baseline)).detach().requires_grad_(True)
        output = wrapper(temp_x, edge_index, **forward_kwargs)
        loss = criterion(output, label)
        wrapper.zero_grad(set_to_none=True)
        loss.backward()
        grads[step] = temp_x.grad.detach()
    avg_grads = ((grads[:-1] + grads[1:]) / 2.0).mean(dim=0)
    return (x - baseline) * avg_grads


def _feature_record(feature_attr, feature_names, top_k=FEATURE_TOP_K):
    arr = feature_attr.detach().cpu().numpy()
    feature_scores = np.abs(arr).sum(axis=0)
    return {
        "attribution": arr.tolist(),
        "feature_scores": feature_scores.tolist(),
        "top_features": _top_items(feature_scores, names=feature_names, k=top_k, use_abs=False),
    }


def _dataset_metadata(dataset):
    return {
        "feature_cols": getattr(dataset, "feature_cols", []),
        "feature_metadata": getattr(dataset, "feature_metadata", {}),
        "label_mapping": getattr(dataset, "label_mapping", {"0": "HOME", "1": "ADMITTED"}),
    }


def _prototype_record(gnn_nets, batch, output_dim, pred_label=None, top_k=5):
    model = getattr(gnn_nets, "model", None)
    if model is None or not getattr(model, "enable_prot", False):
        return {"enabled": False}

    with torch.no_grad():
        logits, probs, _node_emb, _graph_emb, min_distances = gnn_nets(batch)

    if min_distances is None or not torch.is_tensor(min_distances) or min_distances.numel() == 0:
        return {"enabled": True, "available": False}

    distances = min_distances.detach().cpu().reshape(-1).numpy()
    epsilon = float(getattr(model, "epsilon", 1e-4))
    activations = np.log((distances + 1.0) / (distances + epsilon))
    pred = int(pred_label) if pred_label is not None else int(torch.argmax(logits, dim=-1).item())
    num_per_class = max(1, int(len(distances) / max(output_dim, 1)))
    weights = model.last_layer.weight.detach().cpu().numpy()
    contributions = activations * weights[pred]

    def proto_item(proto_idx):
        proto_idx = int(proto_idx)
        return {
            "prototype_index": proto_idx,
            "prototype_class": int(proto_idx // num_per_class),
            "distance": float(distances[proto_idx]),
            "activation": float(activations[proto_idx]),
            "contribution_to_pred": float(contributions[proto_idx]),
        }

    closest = [proto_item(i) for i in np.argsort(distances)[: min(top_k, len(distances))]]
    strongest = [
        proto_item(i)
        for i in np.argsort(-np.abs(contributions))[: min(top_k, len(contributions))]
    ]
    closest_by_class = {}
    for class_idx in range(output_dim):
        start = class_idx * num_per_class
        end = min(start + num_per_class, len(distances))
        if start >= end:
            continue
        class_local = int(np.argmin(distances[start:end]))
        closest_by_class[str(class_idx)] = proto_item(start + class_local)

    return {
        "enabled": True,
        "available": True,
        "num_prototypes": int(len(distances)),
        "num_prototypes_per_class": int(num_per_class),
        "top_closest": closest,
        "top_contributors": strongest,
        "closest_by_class": closest_by_class,
    }


# ===========================================================================
# Training
# ===========================================================================

def train_model(clst: float, sep: float, use_prot: bool = False,
                margin: float = 1.0) -> tuple:
    """
    Train ProtGNN (or a plain GCN when use_prot=False) and save checkpoints.

    Parameters
    ----------
    clst      : cluster-loss weight (only active when use_prot=True)
    sep       : separation-loss weight (only active when use_prot=True)
    use_prot  : whether to enable prototype layers.  When False, the model
                is trained as a standard GCN/GIN which converges reliably.

    Returns
    -------
    gnn_nets, epoch_rows, test_state, output_dim, epoch_header, diagnostics
    """
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    if not use_prot:
        print("  [INFO] Prototype layers DISABLED — training as a standard GCN.")
        model_args.enable_prot = False
    else:
        print("  [INFO] Prototype layers ENABLED.")

    # --- Data ---
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    input_dim  = dataset.num_node_features
    output_dim = int(dataset.num_classes)
    dataloader = get_dataloader(
        dataset,
        train_args.batch_size,
        random_split_flag=data_args.random_split,
        data_split_ratio=data_args.data_split_ratio,
        seed=data_args.seed,
    )

    avg_nodes = sum(dataset[i].x.shape[0] for i in range(len(dataset))) / len(dataset)
    avg_edges = sum(dataset[i].edge_index.shape[1] for i in range(len(dataset))) / len(dataset)
    print(f"  Dataset : {data_args.dataset_name}")
    print(f"  Graphs  : {len(dataset)}  |  avg nodes: {avg_nodes:.1f}  |  avg edges: {avg_edges/2:.1f}")
    print(f"  Classes : {output_dim}  |  input_dim: {input_dim}")

    # --- Model ---
    gnn_nets = GnnNets(input_dim, output_dim, model_args)
    gnn_nets.to_device()

    train_indices = dataloader["train"].dataset.indices
    eval_indices = dataloader["eval"].dataset.indices
    test_indices = dataloader["test"].dataset.indices
    train_labels  = np.array([int(dataset[i].y.view(-1)[0].item()) for i in train_indices])
    eval_split_labels = np.array([int(dataset[i].y.view(-1)[0].item()) for i in eval_indices])
    test_split_labels = np.array([int(dataset[i].y.view(-1)[0].item()) for i in test_indices])
    class_weights, class_counts = _compute_class_weights(train_labels, output_dim)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(gnn_nets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    print(f"  Class counts (train): {class_counts.astype(int).tolist()}")
    print(f"  Class counts (eval) : {np.bincount(eval_split_labels, minlength=output_dim).astype(int).tolist()}")
    print(f"  Class counts (test) : {np.bincount(test_split_labels, minlength=output_dim).astype(int).tolist()}")
    print(f"  Class weights       : {[round(w, 4) for w in class_weights.tolist()]}")
    print(f"  clst={clst}  sep={sep}  lr={train_args.learning_rate}")
    print()

    ckpt_dir = os.path.join(model_args.checkpoint, data_args.dataset_name)
    _mkdir(ckpt_dir)

    best_metric       = float("-inf")   # best ever (for checkpoint)
    best_for_patience = float("-inf")   # reference for early-stop min_delta
    early_stop_cnt    = 0
    epoch_rows        = []

    header = ["epoch", "train_loss", "train_acc",
              "eval_loss", "eval_acc", "eval_f1", "eval_pr_auc",
              "eval_recall", "eval_precision",
              "eval_macro_f1", "eval_balanced_acc", "eval_top3_acc", "eval_top5_acc"]

    t0 = time.time()
    for epoch in range(train_args.max_epochs):

        # --- Prototype projection (every proj_interval epochs after proj_epochs) ---
        _proj_interval = getattr(train_args, "proj_interval", 10)
        if use_prot and epoch >= train_args.proj_epochs and (epoch - train_args.proj_epochs) % _proj_interval == 0:
            gnn_nets.eval()
            # Shuffle train indices once per projection pass so every prototype
            # sees a different set of candidates instead of the deterministic
            # `proto_i * 10` offset slice (which biased later prototypes toward
            # the tail of the training set).
            proj_candidates = list(train_indices)
            random.shuffle(proj_candidates)
            n_prot = output_dim * model_args.num_prototypes_per_class
            proj_t0 = time.time()
            print(f"\n  Prototype projection (epoch {epoch}): {n_prot} prototypes "
                  f"(MCTS subgraph search — this is the slow step)...", flush=True)
            for proto_i in range(n_prot):
                label_cls = proto_i // model_args.num_prototypes_per_class
                count = 0
                best_sim = 0.0
                proj_prot = None
                for j in proj_candidates:
                    d = dataset[j]
                    if int(d.y.view(-1)[0].item()) == label_cls:
                        count += 1
                        coalition, sim, prot = mcts(d, gnn_nets, gnn_nets.model.prototype_vectors[proto_i])
                        if sim > best_sim:
                            best_sim = sim
                            proj_prot = prot
                        if count >= train_args.nearest_graphs:
                            break
                if proj_prot is not None:
                    gnn_nets.model.prototype_vectors.data[proto_i] = proj_prot
                # --- live progress (updates in place) ---
                done = proto_i + 1
                elapsed = time.time() - proj_t0
                eta = elapsed / done * (n_prot - done)
                print(f"\r    P{done:3d}/{n_prot} | class {label_cls:2d} | "
                      f"best_sim={best_sim:.3f} | {elapsed:5.0f}s elapsed | "
                      f"ETA ~{eta:4.0f}s   ", end="", flush=True)
            print(f"\n  Projection done in {time.time() - proj_t0:.0f}s.", flush=True)

        # --- Warm-up vs. joint training ---
        gnn_nets.train()
        if use_prot:
            if epoch < train_args.warm_epochs:
                for p in gnn_nets.model.gnn_layers.parameters():  p.requires_grad = True
                gnn_nets.model.prototype_vectors.requires_grad = True
                for p in gnn_nets.model.last_layer.parameters():  p.requires_grad = False
            else:
                for p in gnn_nets.parameters():  p.requires_grad = True

        batch_losses, batch_accs = [], []
        for batch in dataloader["train"]:
            logits, probs, _, _, min_distances = gnn_nets(batch)
            loss = criterion(logits, batch.y)

            if use_prot and len(min_distances) > 0:
                # Cluster loss
                prototype_class_identity = gnn_nets.model.prototype_class_identity.to(min_distances.device)
                batch_labels = batch.y.view(-1).to(prototype_class_identity.device)
                prot_correct = torch.t(
                    prototype_class_identity[:, batch_labels].bool()
                ).to(min_distances.device)
                cluster_cost = torch.mean(
                    torch.min(min_distances[prot_correct]
                              .reshape(-1, model_args.num_prototypes_per_class), dim=1)[0]
                )

                # Margin-based separation loss: penalise wrong-class prototypes
                # only when they are closer than `margin`. Bounded in [0, margin]
                # per sample → far more stable than the unbounded original
                # `-mean(min_wrong_dist)` formulation (which previously forced
                # users to set sep=0 to avoid blow-up).
                prot_wrong = ~prot_correct
                wrong_min_dist = torch.min(
                    min_distances[prot_wrong]
                    .reshape(-1, (output_dim - 1) * model_args.num_prototypes_per_class), dim=1
                )[0]
                sep_cost = torch.mean(torch.clamp(margin - wrong_min_dist, min=0.0))

                # Sparsity (L1 on cross-class weights)
                l1_mask = 1 - torch.t(prototype_class_identity).to(min_distances.device)
                l1 = (gnn_nets.model.last_layer.weight * l1_mask).norm(p=1)

                # Diversity loss
                ld = torch.tensor(0.0, device=model_args.device)
                for k in range(output_dim):
                    pv = gnn_nets.model.prototype_vectors[
                        k * model_args.num_prototypes_per_class:(k + 1) * model_args.num_prototypes_per_class
                    ]
                    pv = F.normalize(pv, p=2, dim=1)
                    m1 = torch.mm(pv, pv.t()) - torch.eye(pv.shape[0], device=model_args.device) - 0.3
                    m2 = torch.zeros_like(m1)
                    ld = ld + torch.sum(torch.where(m1 > 0, m1, m2))

                total_loss = loss + clst * cluster_cost + sep * sep_cost + 5e-4 * l1 + 0.0 * ld

                # Guard against exploding prototype losses
                if not torch.isfinite(total_loss):
                    total_loss = loss
            else:
                total_loss = loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(gnn_nets.parameters(), clip_value=2.0)
            optimizer.step()

            _, pred = torch.max(logits, -1)
            batch_losses.append(total_loss.item() if torch.isfinite(total_loss) else loss.item())
            batch_accs.append(pred.eq(batch.y).cpu().numpy())

        train_loss = float(np.mean(batch_losses))
        train_acc  = float(np.concatenate(batch_accs).mean())
        eval_state = _evaluate(dataloader["eval"], gnn_nets, criterion)

        # selection metric: PR-AUC (binary) else macro-F1 (multi-class) else acc
        sel = eval_state.get("pr_auc")
        if sel is None:
            sel = eval_state.get("macro_f1", eval_state["acc"])
        sel = eval_state["acc"] if (isinstance(sel, float) and np.isnan(sel)) else sel

        # Early stopping: only a MEANINGFUL improvement (> min_delta) resets the
        # patience counter — tiny 0.0001 noise bumps no longer keep it alive
        # forever once the metric has plateaued.
        min_delta = getattr(train_args, "early_stop_min_delta", 0.0)
        if sel > best_for_patience + min_delta:
            best_for_patience = sel
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        # Save checkpoint whenever we hit a new best at all (independent of delta)
        is_best = (sel > best_metric)
        if is_best:
            best_metric = sel
        state = {"net": gnn_nets.state_dict(), "epoch": epoch, "acc": sel}
        latest = os.path.join(ckpt_dir, f"{model_args.model_name}_latest.pth")
        best   = os.path.join(ckpt_dir, f"{model_args.model_name}_best.pth")
        torch.save(state, latest)
        if is_best or not os.path.isfile(best):
            shutil.copy(latest, best)

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc, 6),
            "eval_loss":  eval_state["loss"],
            "eval_acc":   eval_state["acc"],
            "eval_f1":        eval_state.get("f1", ""),
            "eval_pr_auc":    eval_state.get("pr_auc", ""),
            "eval_recall":    eval_state.get("recall", ""),
            "eval_precision": eval_state.get("precision", ""),
            "eval_macro_f1":     eval_state.get("macro_f1", ""),
            "eval_balanced_acc": eval_state.get("balanced_acc", ""),
            "eval_top3_acc":     eval_state.get("top3_acc", ""),
            "eval_top5_acc":     eval_state.get("top5_acc", ""),
        }
        epoch_rows.append(row)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:4d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Eval Loss: {eval_state['loss']:.4f}  Acc: {eval_state['acc']:.4f}"
            + (f"  PR-AUC: {eval_state['pr_auc']:.4f}" if "pr_auc" in eval_state else "")
            + (f"  macroF1: {eval_state['macro_f1']:.4f}"
               + (f"  top5: {eval_state['top5_acc']:.3f}" if "top5_acc" in eval_state else "")
               if "macro_f1" in eval_state else "")
            + f"  [{elapsed:.0f}s]"
        )

        if early_stop_cnt > train_args.early_stopping:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    # --- Load selected model ---
    selected_ckpt_path = best
    selected_ckpt_label = "best validation"
    if use_prot and epoch_rows and epoch_rows[-1]["epoch"] >= train_args.proj_epochs:
        selected_ckpt_path = latest
        selected_ckpt_label = "latest post-projection"

    ckpt = torch.load(selected_ckpt_path, map_location=model_args.device)
    gnn_nets.update_state_dict(ckpt["net"])
    gnn_nets.eval()
    print(
        f"\n  Selected checkpoint ({selected_ckpt_label}): "
        f"epoch={ckpt['epoch']}, metric={ckpt['acc']:.4f}"
    )

    # --- Calibrate binary decision threshold on validation data ---
    calibrated_threshold = None
    diagnostics = {
        "label_distribution": {
            "train": _label_distribution(train_labels, output_dim),
            "eval": _label_distribution(eval_split_labels, output_dim),
            "test": _label_distribution(test_split_labels, output_dim),
        }
    }
    if output_dim == 2:
        _eval_loss, eval_labels, _eval_preds, eval_probs = _collect_predictions(
            dataloader["eval"], gnn_nets, criterion
        )
        calibrated_threshold, threshold_score = _find_best_threshold(
            eval_labels, eval_probs[:, 1], objective="balanced_acc"
        )
        print(
            f"\n  Calibrated decision threshold: {calibrated_threshold:.3f} "
            f"(validation balanced_acc={threshold_score:.4f})"
        )

    # --- Test evaluation ---
    test_loss, test_labels, test_argmax_preds, test_probs = _collect_predictions(
        dataloader["test"], gnn_nets, criterion
    )
    if output_dim == 2:
        threshold_values = [0.3, 0.36, 0.4, 0.5, 0.6, 0.7]
        if calibrated_threshold is not None:
            threshold_values.append(calibrated_threshold)
        diagnostics["threshold_sensitivity"] = _threshold_sensitivity(
            test_labels, test_probs[:, 1], threshold_values
        )
        diagnostics["confusion_matrices"] = {
            "argmax": _confusion_counts(test_labels, test_argmax_preds),
        }
        for item in diagnostics["threshold_sensitivity"]:
            name = "threshold_" + str(item["threshold"]).replace(".", "_")
            diagnostics["confusion_matrices"][name] = {
                key: item[key] for key in ("tn", "fp", "fn", "tp")
            }

    test_state = _state_from_predictions(
        test_loss, test_labels, test_argmax_preds, test_probs, threshold=calibrated_threshold
    )
    print("\n  TEST RESULTS")
    for k, v in test_state.items():
        print(f"    {k:<18}: {v}")

    if "threshold_sensitivity" in diagnostics:
        print("\n  THRESHOLD SENSITIVITY (test set)")
        for item in diagnostics["threshold_sensitivity"]:
            print(
                f"    t={item['threshold']:.3f} | "
                f"precision={item['precision']:.3f} recall={item['recall']:.3f} "
                f"specificity={item['specificity']:.3f} f1={item['f1']:.3f} | "
                f"TP={item['tp']} FP={item['fp']} TN={item['tn']} FN={item['fn']}"
            )

    return gnn_nets, epoch_rows, test_state, output_dim, header, calibrated_threshold, diagnostics


# ===========================================================================
# GraphXAI Explanation
# ===========================================================================

def explain_test_set(
    gnn_nets: GnnNets,
    output_dim: int,
    explain_n: int,
    threshold: float = None,
) -> list:
    """
    Run GradExplainer, IntegratedGradExplainer, and GNNExplainer on the test set.

    Returns
    -------
    List of dicts, one per graph, with node importance arrays for each explainer.
    """
    if explain_n == 0:
        print("\n[INFO] explain_n=0 — skipping GraphXAI explanation step.")
        return []

    if not GRAPHXAI_OK:
        print("[WARNING] GraphXAI not available — skipping explanation step.")
        return []

    print("\n" + "=" * 60)
    print("GRAPHXAI EXPLANATIONS")
    print("=" * 60)

    # Explanation device: CPU by default. GNNExplainer moves model weights to
    # CPU internally, which corrupts the device state for subsequent graphs on
    # MPS ("weight is on cpu but expected on mps") and crashes the whole pass.
    # CPU is the reliable path; set EXPLAIN_DEVICE=mps to (try to) use the GPU.
    _exp_dev = os.environ.get("EXPLAIN_DEVICE", "cpu")
    explain_device = torch.device(_exp_dev)
    print(f"  Explanation device: {explain_device} "
          f"(set EXPLAIN_DEVICE=mps to try GPU)")
    previous_gnn_device = getattr(gnn_nets, "device", model_args.device)
    previous_model_device = getattr(getattr(gnn_nets, "model", None), "device", None)
    gnn_nets.device = explain_device
    if hasattr(gnn_nets, "model") and hasattr(gnn_nets.model, "device"):
        gnn_nets.model.device = explain_device
    gnn_nets.to(explain_device)

    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    dataloader = get_dataloader(
        dataset, batch_size=1,
        random_split_flag=data_args.random_split,
        data_split_ratio=data_args.data_split_ratio,
        seed=data_args.seed,
    )

    wrapper   = ProtGNNWrapper(gnn_nets)
    wrapper.eval()
    criterion = nn.CrossEntropyLoss()

    grad_exp  = GradExplainer(wrapper, criterion=criterion)
    integ_exp = IntegratedGradExplainer(wrapper, criterion=criterion)
    gnn_exp   = GNNExplainer(wrapper)

    print(f"  GNN layers detected (L): {grad_exp.L}")
    print(f"  Explaining {'all' if explain_n < 0 else explain_n} test graphs\n")

    feature_names = getattr(dataset, "feature_cols", [])
    records = []
    n_total = len(dataloader["test"])
    limit   = n_total if explain_n < 0 else min(explain_n, n_total)

    for idx, batch in enumerate(dataloader["test"]):
        if idx >= limit:
            break

        x          = batch.x.to(explain_device)
        edge_index = batch.edge_index.to(explain_device)
        label      = batch.y.to(explain_device)
        null_batch = torch.zeros(x.size(0), dtype=torch.long, device=explain_device)
        fwd_kwargs = {"batch": null_batch}

        with torch.no_grad():
            logits = wrapper(x, edge_index, null_batch)
            if threshold is not None and logits.shape[-1] == 2:
                probs = torch.softmax(logits, dim=-1)
                pred = int(probs[:, 1].item() >= threshold)
            else:
                pred = logits.argmax(dim=-1).item()

        # Explain the model's DECISION → attribute w.r.t. the PREDICTED class,
        # not the ground-truth label. (For correct predictions they coincide;
        # for wrong ones this shows "why the model said pred", which is what a
        # self-explaining diagnosis tool should surface.)
        explain_label = torch.tensor([int(pred)], dtype=torch.long,
                                     device=explain_device)

        record = {
            "graph_idx":   idx,
            "dataset_index": _first_or_none(getattr(batch, "dataset_index", None)),
            "patient_index": _first_or_none(getattr(batch, "patient_index", None)),
            "node_patient_indices": _tensor_to_list(getattr(batch, "node_patient_indices", None)),
            "true_label":  int(label.item()),
            "pred_label":  int(pred),
            "correct":     bool(pred == label.item()),
            "num_nodes":   int(x.size(0)),
            "num_edges":   int(edge_index.size(1)),
            "feature_names": feature_names,
            "prototype_evidence": _prototype_record(gnn_nets, batch, output_dim, pred_label=pred),
            "explanations": {},
        }

        # ---- GradExplainer ----
        try:
            exp = grad_exp.get_explanation_graph(
                x=x, edge_index=edge_index, label=explain_label, forward_kwargs=fwd_kwargs
            )
            imp = exp.node_imp.detach().cpu().numpy()
            feature_attr = _grad_feature_attribution(wrapper, x, edge_index, explain_label, fwd_kwargs, criterion)
            record["explanations"]["GradExplainer"] = {
                "node_importance": imp.tolist(),
                "top_nodes": _top_items(imp, k=NODE_TOP_K),
                "feature_importance": _feature_record(feature_attr, feature_names),
                "min": float(imp.min()),
                "max": float(imp.max()),
                "mean": float(imp.mean()),
                "std": float(imp.std()),
            }
        except Exception as e:
            record["explanations"]["GradExplainer"] = {"error": str(e)}

        # ---- IntegratedGradExplainer ----
        try:
            exp = integ_exp.get_explanation_graph(
                x=x, edge_index=edge_index, label=explain_label, forward_kwargs=fwd_kwargs
            )
            imp = exp.node_imp.detach().cpu().numpy()
            feature_attr = _integrated_grad_feature_attribution(
                wrapper, x, edge_index, explain_label, fwd_kwargs, criterion
            )
            record["explanations"]["IntegratedGradExplainer"] = {
                "node_importance": imp.tolist(),
                "top_nodes": _top_items(imp, k=NODE_TOP_K),
                "feature_importance": _feature_record(feature_attr, feature_names),
                "min": float(imp.min()),
                "max": float(imp.max()),
                "mean": float(imp.mean()),
                "std": float(imp.std()),
            }
        except Exception as e:
            record["explanations"]["IntegratedGradExplainer"] = {"error": str(e)}

        # ---- GNNExplainer ----
        try:
            exp = gnn_exp.get_explanation_graph(
                x=x, edge_index=edge_index, forward_kwargs=fwd_kwargs
            )
            imp = exp.node_imp.detach().cpu().numpy()
            feature_imp = exp.feature_imp.detach().cpu().numpy()
            record["explanations"]["GNNExplainer"] = {
                "node_importance": imp.tolist(),
                "top_nodes": _top_items(imp, k=NODE_TOP_K, use_abs=False),
                "feature_importance": {
                    "feature_scores": feature_imp.tolist(),
                    "top_features": _top_items(feature_imp, names=feature_names, k=FEATURE_TOP_K, use_abs=False),
                },
                "min": float(imp.min()),
                "max": float(imp.max()),
                "mean": float(imp.mean()),
                "std": float(imp.std()),
            }
        except Exception as e:
            record["explanations"]["GNNExplainer"] = {"error": str(e)}

        # Summary print
        status = "CORRECT" if record["correct"] else "WRONG  "
        print(
            f"  [{status}] Graph {idx:4d} | "
            f"true={record['true_label']} pred={record['pred_label']} | "
            f"nodes={record['num_nodes']:3d} edges={record['num_edges']:3d}"
        )
        for name, res in record["explanations"].items():
            if "error" in res:
                print(f"    {name:<28}: ERROR — {res['error']}")
            else:
                print(
                    f"    {name:<28}: "
                    f"min={res['min']:+.4f}  max={res['max']:+.4f}  "
                    f"mean={res['mean']:+.4f}  std={res['std']:.4f}"
                )

        records.append(record)

    gnn_nets.device = previous_gnn_device
    if hasattr(gnn_nets, "model") and previous_model_device is not None:
        gnn_nets.model.device = previous_model_device
    gnn_nets.to(previous_gnn_device)

    return records


# ===========================================================================
# Save results
# ===========================================================================

def save_results(
    epoch_rows: list,
    epoch_header: list,
    test_state: dict,
    diagnostics: dict,
    explanation_records: list,
    clst: float,
    sep: float,
    output_dim: int,
    use_prot: bool = False,
    threshold: float = None,
):
    """Write all artefacts under outputs/results/."""
    _prepare_results_dir()

    # ---- model_config.json ----
    config = {
        "dataset":                 data_args.dataset_name,
        "model":                   model_args.model_name,
        "input_dim":               "from dataset",
        "output_dim":              output_dim,
        "latent_dim":              model_args.latent_dim,
        "mlp_hidden":              model_args.mlp_hidden,
        "readout":                 model_args.readout,
        "dropout":                 model_args.dropout,
        "adj_normalize":           model_args.adj_normlize,
        "emb_normalize":           model_args.emb_normlize,
        "enable_prototypes":       use_prot,
        "num_prototypes_per_class": model_args.num_prototypes_per_class,
        "learning_rate":           train_args.learning_rate,
        "batch_size":              train_args.batch_size,
        "weight_decay":            train_args.weight_decay,
        "max_epochs":              train_args.max_epochs,
        "early_stopping":          train_args.early_stopping,
        "warm_epochs":             train_args.warm_epochs,
        "proj_epochs":             train_args.proj_epochs,
        "clst_weight":             clst,
        "sep_weight":              sep,
        "decision_threshold":       threshold,
        "data_split_ratio":        data_args.data_split_ratio,
        "seed":                    data_args.seed,
    }
    with open(os.path.join(RESULTS_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    with open(os.path.join(RESULTS_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(_dataset_metadata(dataset), f, indent=2)

    # ---- training_metrics.csv ----
    csv_path = os.path.join(RESULTS_DIR, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_header)
        writer.writeheader()
        writer.writerows(epoch_rows)

    # ---- test_metrics.json ----
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(test_state, f, indent=2)

    # ---- evaluation_diagnostics.json ----
    diagnostics = diagnostics or {}
    with open(os.path.join(RESULTS_DIR, "evaluation_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    # ---- threshold_sensitivity.csv ----
    threshold_rows = diagnostics.get("threshold_sensitivity", [])
    if threshold_rows:
        threshold_path = os.path.join(RESULTS_DIR, "threshold_sensitivity.csv")
        threshold_header = [
            "threshold", "tn", "fp", "fn", "tp",
            "predicted_home", "predicted_admitted",
            "acc", "precision", "recall", "specificity", "f1", "balanced_acc", "pr_auc",
        ]
        with open(threshold_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=threshold_header)
            writer.writeheader()
            writer.writerows(threshold_rows)

    # ---- per-graph explanation JSON files ----
    for rec in explanation_records:
        fname = os.path.join(RESULTS_DIR, "explanations", f"graph_{rec['graph_idx']}.json")
        with open(fname, "w") as f:
            json.dump(rec, f, indent=2)

    # ---- explanations_summary.csv ----
    summary_path = os.path.join(RESULTS_DIR, "explanations", "explanations_summary.csv")
    summary_header = [
        "graph_idx", "true_label", "pred_label", "correct", "num_nodes", "num_edges",
    ]
    for name in EXPLAINER_NAMES:
        for stat in ("min", "max", "mean", "std"):
            summary_header.append(f"{name}_{stat}")
    summary_header.append("any_error")

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_header)
        writer.writeheader()
        for rec in explanation_records:
            row = {
                "graph_idx":  rec["graph_idx"],
                "true_label": rec["true_label"],
                "pred_label": rec["pred_label"],
                "correct":    rec["correct"],
                "num_nodes":  rec["num_nodes"],
                "num_edges":  rec["num_edges"],
                "any_error":  False,
            }
            for name in EXPLAINER_NAMES:
                res = rec["explanations"].get(name, {})
                if "error" in res:
                    for stat in ("min", "max", "mean", "std"):
                        row[f"{name}_{stat}"] = ""
                    row["any_error"] = True
                else:
                    for stat in ("min", "max", "mean", "std"):
                        row[f"{name}_{stat}"] = res.get(stat, "")
            writer.writerow(row)

    # ---- report.txt ----
    total_epochs = len(epoch_rows)

    def _epoch_sel(r):
        # PR-AUC (binary) > macro-F1 (multi-class) > accuracy
        if r.get("eval_pr_auc", "") not in ("", None):
            return r["eval_pr_auc"] or 0
        if r.get("eval_macro_f1", "") not in ("", None):
            return r["eval_macro_f1"] or 0
        return r["eval_acc"]
    best_epoch   = max(epoch_rows, key=_epoch_sel)
    n_explained  = len(explanation_records)
    n_correct    = sum(1 for r in explanation_records if r["correct"])
    exp_success  = {
        name: sum(1 for r in explanation_records if "error" not in r["explanations"].get(name, {"error": ""}))
        for name in EXPLAINER_NAMES
    }

    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write("=" * 64 + "\n")
        f.write("ProtGNN + GraphXAI — Run Report\n")
        f.write("=" * 64 + "\n\n")

        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        for k, v in config.items():
            f.write(f"  {k:<30}: {v}\n")
        f.write("\n")

        f.write("TRAINING SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total epochs run         : {total_epochs}\n")
        f.write(f"  Best epoch               : {best_epoch['epoch']}\n")
        f.write(f"  Best eval acc            : {best_epoch['eval_acc']}\n")
        f.write(f"  Best eval PR-AUC         : {best_epoch.get('eval_pr_auc', 'N/A')}\n")
        f.write("\n")

        f.write("TEST SET PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        for k, v in test_state.items():
            f.write(f"  {k:<24}: {v}\n")
        f.write("\n")

        label_distribution = diagnostics.get("label_distribution", {})
        if label_distribution:
            f.write("LABEL DISTRIBUTION\n")
            f.write("-" * 40 + "\n")
            label_map = config.get("label_mapping", {}) or {}
            if output_dim == 2:
                for split_name in ("train", "eval", "test"):
                    split_dist = label_distribution.get(split_name, {})
                    class_0 = split_dist.get("class_0", {})
                    class_1 = split_dist.get("class_1", {})
                    f.write(
                        f"  {split_name:<8}: "
                        f"HOME={class_0.get('count', 0)} ({class_0.get('rate', 0.0):.3f})  "
                        f"ADMITTED={class_1.get('count', 0)} ({class_1.get('rate', 0.0):.3f})\n"
                    )
            else:
                # multi-class: list per-class train counts (with names if available)
                train_dist = label_distribution.get("train", {})
                f.write(f"  {output_dim} classes (train counts):\n")
                for idx in range(output_dim):
                    cd = train_dist.get(f"class_{idx}", {})
                    name = label_map.get(str(idx), f"class_{idx}")
                    f.write(f"    [{idx:>2}] {name:<40} "
                            f"{cd.get('count', 0)} ({cd.get('rate', 0.0):.3f})\n")
            f.write("\n")

        confusion_matrices = diagnostics.get("confusion_matrices", {})
        if confusion_matrices:
            f.write("CONFUSION MATRICES\n")
            f.write("-" * 40 + "\n")
            for name, counts in confusion_matrices.items():
                f.write(
                    f"  {name:<18}: "
                    f"TN={counts.get('tn', 0)}  FP={counts.get('fp', 0)}  "
                    f"FN={counts.get('fn', 0)}  TP={counts.get('tp', 0)}\n"
                )
            f.write("\n")

        if threshold_rows:
            f.write("THRESHOLD SENSITIVITY\n")
            f.write("-" * 40 + "\n")
            f.write("  threshold  precision  recall  specificity  f1      balanced_acc  TP  FP  TN  FN\n")
            for row in threshold_rows:
                f.write(
                    f"  {row['threshold']:<9.3f}  "
                    f"{row['precision']:<9.3f}  {row['recall']:<6.3f}  "
                    f"{row['specificity']:<11.3f}  {row['f1']:<6.3f}  "
                    f"{row['balanced_acc']:<12.3f}  "
                    f"{row['tp']:<2}  {row['fp']:<2}  {row['tn']:<2}  {row['fn']:<2}\n"
                )
            f.write("\n")

        f.write("GRAPHXAI EXPLANATION SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Graphs explained         : {n_explained}\n")
        f.write(f"  Model correct (on those) : {n_correct} / {n_explained}\n")
        for name, cnt in exp_success.items():
            f.write(f"  {name:<28}: {cnt} / {n_explained} successful\n")
        f.write("\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"  outputs/results/model_config.json\n")
        f.write(f"  outputs/results/dataset_metadata.json\n")
        f.write(f"  outputs/results/training_metrics.csv          ({total_epochs} rows)\n")
        f.write(f"  outputs/results/test_metrics.json\n")
        f.write(f"  outputs/results/evaluation_diagnostics.json\n")
        if threshold_rows:
            f.write(f"  outputs/results/threshold_sensitivity.csv\n")
        f.write(f"  outputs/results/explanations/graph_<i>.json   ({n_explained} files)\n")
        f.write(f"  outputs/results/explanations/explanations_summary.csv\n")
        f.write(f"  outputs/results/report.txt\n")

    print(f"\n  Saved report to {os.path.join(RESULTS_DIR, 'report.txt')}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ProtGNN and explain with GraphXAI")
    parser.add_argument("--dataset",   default=None, help="Dataset name, e.g. mimic_patient_sim_no_los_k20")
    parser.add_argument("--clst",      type=float, default=0.1,  help="Cluster loss weight")
    parser.add_argument("--sep",       type=float, default=0.1,  help="Separation loss weight (margin-based)")
    parser.add_argument("--margin",    type=float, default=1.0,  help="Separation-loss margin (penalty if wrong-class prototype distance < margin)")
    parser.add_argument("--seed",      type=int,   default=None, help="Override data-split seed for multi-seed runs")
    parser.add_argument("--explain_n", type=int,   default=10,   help="Graphs to explain (-1 = all)")
    parser.add_argument("--no_prot",   action="store_true",      help="Disable prototype layers (standard GCN training)")
    parser.add_argument("--no_archive", action="store_true",     help="Do not copy this run into outputs/runs")
    parser.add_argument("--archive_tag", default="auto", help="Tag appended to archived run folder")
    args = parser.parse_args()

    if args.dataset:
        data_args.dataset_name = args.dataset
    if args.seed is not None:
        data_args.seed = args.seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    use_prot = not args.no_prot

    # Train
    gnn_nets, epoch_rows, test_state, output_dim, epoch_header, threshold, diagnostics = train_model(
        args.clst, args.sep, use_prot, margin=args.margin
    )

    # Explain
    explanation_records = explain_test_set(gnn_nets, output_dim, args.explain_n, threshold=threshold)

    # Save
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    save_results(epoch_rows, epoch_header, test_state, diagnostics, explanation_records,
                 args.clst, args.sep, output_dim, use_prot, threshold=threshold)

    print(f"\n  All results written to: {RESULTS_DIR}/")
    if not args.no_archive:
        command = " ".join([sys.executable, *sys.argv])
        archive_dir = archive_results(
            results_dir=RESULTS_DIR,
            runs_dir=os.path.join(str(OUTPUTS_DIR), "runs"),
            tag=args.archive_tag,
            command=command,
        )
        print(f"  Archived run to: {archive_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
