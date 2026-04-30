#train_and_explain.py
"""
ProtGNN — Train & Explain with GraphXAI
========================================
This script trains the ProtGNN model on the configured dataset, then runs
three GraphXAI explainers on the test set and saves every artefact under
the ``results/`` directory.

Output layout
-------------
results/
├── model_config.json          hyperparameters used for this run
├── training_metrics.csv       per-epoch train / eval metrics
├── test_metrics.json          final test-set performance
├── explanations/
│   ├── graph_<i>.json         per-graph node-importance for all explainers
│   └── explanations_summary.csv  min / max / mean per graph per explainer
└── report.txt                 human-readable end-to-end summary

Usage
-----
    python train_and_explain.py [--clst 0.1] [--sep 0.1] [--explain_n 10] [--no_prot]

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
_GRAPHXAI_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir, "GraphXAI"))
if _GRAPHXAI_ROOT not in sys.path:
    sys.path.insert(0, _GRAPHXAI_ROOT)

# ---------------------------------------------------------------------------
# ProtGNN imports
# ---------------------------------------------------------------------------
from Configures import data_args, model_args, train_args
from models import GnnNets
from models.graphxai_wrapper import ProtGNNWrapper
from load_dataset import get_dataset, get_dataloader
from my_mcts import mcts

# ---------------------------------------------------------------------------
# GraphXAI imports
# ---------------------------------------------------------------------------
try:
    from graphxai.explainers import GradExplainer, IntegratedGradExplainer, GNNExplainer
    GRAPHXAI_OK = True
except Exception as _e:
    print(f"[WARNING] GraphXAI could not be imported: {_e}")
    GRAPHXAI_OK = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
EXPLAINER_NAMES = ["GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]


# ===========================================================================
# Helpers
# ===========================================================================

def _mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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


def _evaluate(dataloader, model, criterion):
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
    state = {
        "loss": round(float(np.mean(losses)), 6),
        "acc":  round(float(np.concatenate(accs).mean()), 6),
    }
    labels_all = torch.cat(labels_all).numpy()
    preds_all  = torch.cat(preds_all).numpy()
    probs_all  = torch.cat(probs_all).numpy()
    if probs_all.ndim == 2 and probs_all.shape[1] == 2:
        state.update(_binary_metrics(labels_all, preds_all, probs_all[:, 1]))
    return state


# ===========================================================================
# Training
# ===========================================================================

def train_model(clst: float, sep: float, use_prot: bool = False) -> tuple:
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
    gnn_nets, epoch_rows, test_state, output_dim, epoch_header
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
    train_labels  = np.array([int(dataset[i].y.view(-1)[0].item()) for i in train_indices])
    class_weights, class_counts = _compute_class_weights(train_labels, output_dim)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(gnn_nets.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    print(f"  Class counts (train): {class_counts.astype(int).tolist()}")
    print(f"  Class weights       : {[round(w, 4) for w in class_weights.tolist()]}")
    print(f"  clst={clst}  sep={sep}  lr={train_args.learning_rate}")
    print()

    ckpt_dir = os.path.join("checkpoint", data_args.dataset_name)
    _mkdir(ckpt_dir)

    best_metric     = float("-inf")
    early_stop_cnt  = 0
    epoch_rows      = []

    header = ["epoch", "train_loss", "train_acc",
              "eval_loss", "eval_acc", "eval_f1", "eval_pr_auc",
              "eval_recall", "eval_precision"]

    t0 = time.time()
    for epoch in range(train_args.max_epochs):

        # --- Prototype projection (every 10 epochs after proj_epochs) ---
        if use_prot and epoch >= train_args.proj_epochs and epoch % 10 == 0:
            gnn_nets.eval()
            for proto_i in range(output_dim * model_args.num_prototypes_per_class):
                label_cls = proto_i // model_args.num_prototypes_per_class
                count = 0
                best_sim = 0.0
                proj_prot = None
                for j in range(proto_i * 10, len(train_indices)):
                    d = dataset[train_indices[j]]
                    if d.y == label_cls:
                        count += 1
                        coalition, sim, prot = mcts(d, gnn_nets, gnn_nets.model.prototype_vectors[proto_i])
                        if sim > best_sim:
                            best_sim = sim
                            proj_prot = prot
                    if count >= train_args.nearest_graphs:
                        if proj_prot is not None:
                            gnn_nets.model.prototype_vectors.data[proto_i] = proj_prot
                        break

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
                prot_correct = torch.t(
                    gnn_nets.model.prototype_class_identity[:, batch.y].bool()
                ).to(model_args.device)
                cluster_cost = torch.mean(
                    torch.min(min_distances[prot_correct]
                              .reshape(-1, model_args.num_prototypes_per_class), dim=1)[0]
                )

                # Separation loss
                prot_wrong = ~prot_correct
                sep_cost = -torch.mean(
                    torch.min(min_distances[prot_wrong]
                              .reshape(-1, (output_dim - 1) * model_args.num_prototypes_per_class), dim=1)[0]
                )

                # Sparsity (L1 on cross-class weights)
                l1_mask = 1 - torch.t(gnn_nets.model.prototype_class_identity).to(model_args.device)
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

        sel = eval_state.get("pr_auc", eval_state["acc"])
        sel = eval_state["acc"] if (isinstance(sel, float) and np.isnan(sel)) else sel

        # Early stopping
        if sel > best_metric:
            best_metric    = sel
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        # Save checkpoint
        is_best = (sel >= best_metric)
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
        }
        epoch_rows.append(row)

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:4d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
            f"Eval Loss: {eval_state['loss']:.4f}  Acc: {eval_state['acc']:.4f}"
            + (f"  PR-AUC: {eval_state['pr_auc']:.4f}" if "pr_auc" in eval_state else "")
            + f"  [{elapsed:.0f}s]"
        )

        if early_stop_cnt > train_args.early_stopping:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    # --- Load best model ---
    ckpt = torch.load(best, map_location=model_args.device)
    gnn_nets.update_state_dict(ckpt["net"])
    gnn_nets.eval()
    print(f"\n  Best checkpoint: epoch={ckpt['epoch']}, metric={ckpt['acc']:.4f}")

    # --- Test evaluation ---
    test_state = _evaluate(dataloader["test"], gnn_nets, criterion)
    print("\n  TEST RESULTS")
    for k, v in test_state.items():
        print(f"    {k:<18}: {v}")

    return gnn_nets, epoch_rows, test_state, output_dim, header


# ===========================================================================
# GraphXAI Explanation
# ===========================================================================

def explain_test_set(gnn_nets: GnnNets, output_dim: int, explain_n: int) -> list:
    """
    Run GradExplainer, IntegratedGradExplainer, and GNNExplainer on the test set.

    Returns
    -------
    List of dicts, one per graph, with node importance arrays for each explainer.
    """
    if not GRAPHXAI_OK:
        print("[WARNING] GraphXAI not available — skipping explanation step.")
        return []

    print("\n" + "=" * 60)
    print("GRAPHXAI EXPLANATIONS")
    print("=" * 60)

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

    records = []
    n_total = len(dataloader["test"])
    limit   = n_total if explain_n < 0 else min(explain_n, n_total)

    for idx, batch in enumerate(dataloader["test"]):
        if idx >= limit:
            break

        x          = batch.x.to(model_args.device)
        edge_index = batch.edge_index.to(model_args.device)
        label      = batch.y.to(model_args.device)
        null_batch = torch.zeros(x.size(0), dtype=torch.long, device=model_args.device)
        fwd_kwargs = {"batch": null_batch}

        with torch.no_grad():
            logits = wrapper(x, edge_index, null_batch)
            pred   = logits.argmax(dim=-1).item()

        record = {
            "graph_idx":   idx,
            "true_label":  int(label.item()),
            "pred_label":  int(pred),
            "correct":     bool(pred == label.item()),
            "num_nodes":   int(x.size(0)),
            "num_edges":   int(edge_index.size(1)),
            "explanations": {},
        }

        # ---- GradExplainer ----
        try:
            exp = grad_exp.get_explanation_graph(
                x=x, edge_index=edge_index, label=label, forward_kwargs=fwd_kwargs
            )
            imp = exp.node_imp.detach().cpu().numpy()
            record["explanations"]["GradExplainer"] = {
                "node_importance": imp.tolist(),
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
                x=x, edge_index=edge_index, label=label, forward_kwargs=fwd_kwargs
            )
            imp = exp.node_imp.detach().cpu().numpy()
            record["explanations"]["IntegratedGradExplainer"] = {
                "node_importance": imp.tolist(),
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
            record["explanations"]["GNNExplainer"] = {
                "node_importance": imp.tolist(),
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

    return records


# ===========================================================================
# Save results
# ===========================================================================

def save_results(
    epoch_rows: list,
    epoch_header: list,
    test_state: dict,
    explanation_records: list,
    clst: float,
    sep: float,
    output_dim: int,
    use_prot: bool = False,
):
    """Write all artefacts under results/."""
    _mkdir(RESULTS_DIR)
    _mkdir(os.path.join(RESULTS_DIR, "explanations"))

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
        "data_split_ratio":        data_args.data_split_ratio,
        "seed":                    data_args.seed,
    }
    with open(os.path.join(RESULTS_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- training_metrics.csv ----
    csv_path = os.path.join(RESULTS_DIR, "training_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_header)
        writer.writeheader()
        writer.writerows(epoch_rows)

    # ---- test_metrics.json ----
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(test_state, f, indent=2)

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
    best_epoch   = max(epoch_rows, key=lambda r: (r["eval_pr_auc"] or 0) if r["eval_pr_auc"] != "" else r["eval_acc"])
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

        f.write("GRAPHXAI EXPLANATION SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Graphs explained         : {n_explained}\n")
        f.write(f"  Model correct (on those) : {n_correct} / {n_explained}\n")
        for name, cnt in exp_success.items():
            f.write(f"  {name:<28}: {cnt} / {n_explained} successful\n")
        f.write("\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 40 + "\n")
        f.write(f"  results/model_config.json\n")
        f.write(f"  results/training_metrics.csv          ({total_epochs} rows)\n")
        f.write(f"  results/test_metrics.json\n")
        f.write(f"  results/explanations/graph_<i>.json   ({n_explained} files)\n")
        f.write(f"  results/explanations/explanations_summary.csv\n")
        f.write(f"  results/report.txt\n")

    print(f"\n  Saved report to {os.path.join(RESULTS_DIR, 'report.txt')}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Train ProtGNN and explain with GraphXAI")
    parser.add_argument("--clst",      type=float, default=0.1,  help="Cluster loss weight")
    parser.add_argument("--sep",       type=float, default=0.1,  help="Separation loss weight")
    parser.add_argument("--explain_n", type=int,   default=10,   help="Graphs to explain (-1 = all)")
    parser.add_argument("--no_prot",   action="store_true",      help="Disable prototype layers (standard GCN training)")
    args = parser.parse_args()

    use_prot = not args.no_prot

    # Train
    gnn_nets, epoch_rows, test_state, output_dim, epoch_header = train_model(args.clst, args.sep, use_prot)

    # Explain
    explanation_records = explain_test_set(gnn_nets, output_dim, args.explain_n)

    # Save
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    save_results(epoch_rows, epoch_header, test_state, explanation_records,
                 args.clst, args.sep, output_dim, use_prot)

    print(f"\n  All results written to: {RESULTS_DIR}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
