import csv
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.utils import k_hop_subgraph


PROJECT_ROOT = os.getcwd()
GRAPHXAI_ROOT = os.path.normpath(os.path.join(PROJECT_ROOT, os.pardir, os.pardir, "GraphXAI"))
if GRAPHXAI_ROOT not in sys.path:
    sys.path.insert(0, GRAPHXAI_ROOT)

from graphxai.explainers import GradExplainer, IntegratedGradExplainer, GNNExplainer


DATA_PATH = "datasets/ds1/patient_similarity/k10/processed/k10_ps_data.pt"
META_PATH = "datasets/ds1/patient_similarity/k10/metadata.json"
OUT_DIR = "datasets/ds1/patient_similarity/k10/train"
EXPLAIN_DIR = os.path.join(OUT_DIR, "explanations")

SEED = 7
HIDDEN_DIM = 64
DROPOUT = 0.4
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 300
EARLY_STOPPING = 50
EXPLAIN_N = 30
EXPLAIN_HOPS = 2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PatientGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim, normalize=True)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, normalize=True)
        self.conv3 = GCNConv(hidden_dim, output_dim, normalize=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv3(x, edge_index, edge_weight=edge_weight)


class UnweightedWrapper(nn.Module):
    """GraphXAI-compatible wrapper. Explanations use local topology and node features."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index):
        return self.model(x, edge_index, edge_weight=None)


def stratified_masks(y, train_ratio=0.8, val_ratio=0.1, seed=7):
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    y_np = y.cpu().numpy()
    for cls in np.unique(y_np):
        idx = np.where(y_np == cls)[0]
        rng.shuffle(idx)
        n_train = int(round(train_ratio * len(idx)))
        n_val = int(round(val_ratio * len(idx)))
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        torch.tensor(test_idx, dtype=torch.long),
    )


def binary_metrics(y_true, logits):
    probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    labels = y_true.detach().cpu().numpy()
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    acc = float(np.mean(preds == labels))
    try:
        pr_auc = float(average_precision_score(labels, probs))
    except ValueError:
        pr_auc = float("nan")
    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except ValueError:
        roc_auc = float("nan")
    return {
        "acc": round(acc, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
        "balanced_acc": round((recall + specificity) / 2, 6),
        "pr_auc": round(pr_auc, 6),
        "roc_auc": round(roc_auc, 6),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(model, data, idx, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr.view(-1))
        loss = criterion(logits[idx], data.y[idx])
    metrics = {"loss": round(float(loss.item()), 6)}
    metrics.update(binary_metrics(data.y[idx], logits[idx]))
    return metrics, logits


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def explanation_stats(exp):
    out = {}
    if getattr(exp, "feature_imp", None) is not None:
        vals = exp.feature_imp.detach().cpu().numpy().astype(float)
        out["feature_imp"] = vals.tolist()
        out["feature_min"] = float(vals.min())
        out["feature_max"] = float(vals.max())
        out["feature_mean"] = float(vals.mean())
        out["feature_std"] = float(vals.std())
    if getattr(exp, "node_imp", None) is not None:
        vals = exp.node_imp.detach().cpu().numpy().astype(float)
        out["node_imp"] = vals.tolist()
        out["node_min"] = float(vals.min())
        out["node_max"] = float(vals.max())
        out["node_mean"] = float(vals.mean())
        out["node_std"] = float(vals.std())
    if getattr(exp, "edge_imp", None) is not None:
        vals = exp.edge_imp.detach().cpu().numpy().astype(float)
        out["edge_imp"] = vals.tolist()
        out["edge_min"] = float(vals.min()) if vals.size else 0.0
        out["edge_max"] = float(vals.max()) if vals.size else 0.0
        out["edge_mean"] = float(vals.mean()) if vals.size else 0.0
        out["edge_std"] = float(vals.std()) if vals.size else 0.0
    return out


def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EXPLAIN_DIR, exist_ok=True)

    data = torch.load(DATA_PATH, map_location="cpu")
    with open(META_PATH) as f:
        graph_meta = json.load(f)

    train_idx, val_idx, test_idx = stratified_masks(data.y, seed=SEED)
    class_counts = torch.bincount(data.y[train_idx], minlength=2).float()
    class_weights = len(train_idx) / (2 * class_counts.clamp_min(1))

    model = PatientGCN(data.x.shape[1], HIDDEN_DIM, int(data.y.max().item()) + 1, DROPOUT)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    config = {
        "graph_source": DATA_PATH,
        "graph_type": graph_meta.get("graph_type"),
        "task": "node_classification",
        "note": "Patient similarity is one graph. Explanations are saved for 30 test patient nodes and their k-hop subgraphs.",
        "seed": SEED,
        "hidden_dim": HIDDEN_DIM,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "early_stopping": EARLY_STOPPING,
        "explain_n": EXPLAIN_N,
        "explain_hops": EXPLAIN_HOPS,
        "num_nodes": int(data.x.shape[0]),
        "num_edges": int(data.edge_index.shape[1]),
        "num_features": int(data.x.shape[1]),
        "split_sizes": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "class_weights": [float(v) for v in class_weights],
    }
    with open(os.path.join(OUT_DIR, "model_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(OUT_DIR, "split_indices.json"), "w") as f:
        json.dump({
            "train": train_idx.tolist(),
            "val": val_idx.tolist(),
            "test": test_idx.tolist(),
        }, f)

    best_metric = -1.0
    best_epoch = -1
    best_state = None
    patience = 0
    rows = []
    start = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr.view(-1))
        loss = criterion(logits[train_idx], data.y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        train_metrics = {"loss": round(float(loss.item()), 6)}
        train_metrics.update(binary_metrics(data.y[train_idx], logits[train_idx]))
        val_metrics, _ = evaluate(model, data, val_idx, criterion)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_f1": train_metrics["f1"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_f1": val_metrics["f1"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        rows.append(row)

        metric = val_metrics["pr_auc"]
        if np.isnan(metric):
            metric = val_metrics["balanced_acc"]
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1']:.4f} "
                f"val_pr_auc={val_metrics['pr_auc']:.4f}"
            )

        if patience > EARLY_STOPPING:
            break

    write_csv(
        os.path.join(OUT_DIR, "training_metrics.csv"),
        rows,
        ["epoch", "train_loss", "train_acc", "train_f1", "train_pr_auc", "val_loss", "val_acc", "val_f1", "val_pr_auc", "val_roc_auc"],
    )

    model.load_state_dict(best_state)
    test_metrics, logits = evaluate(model, data, test_idx, criterion)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "best_epoch": best_epoch,
            "best_val_metric": best_metric,
            "test_metrics": test_metrics,
        },
        os.path.join(OUT_DIR, "model_checkpoint.pt"),
    )
    with open(os.path.join(OUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    model.eval()
    wrapper = UnweightedWrapper(model)
    wrapper.eval()
    grad_exp = GradExplainer(wrapper, criterion=nn.CrossEntropyLoss())
    int_exp = IntegratedGradExplainer(wrapper, criterion=nn.CrossEntropyLoss())
    gnn_exp = GNNExplainer(wrapper)

    rng = np.random.default_rng(SEED)
    explain_nodes = rng.choice(test_idx.numpy(), size=min(EXPLAIN_N, len(test_idx)), replace=False).tolist()
    explain_nodes = [int(v) for v in explain_nodes]
    summary_rows = []

    for node_idx in explain_nodes:
        true_label = int(data.y[node_idx].item())
        pred_label = int(logits[node_idx].argmax().item())
        prob_pos = float(torch.softmax(logits[node_idx], dim=-1)[1].item())
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            node_idx, EXPLAIN_HOPS, data.edge_index, relabel_nodes=True, num_nodes=data.x.shape[0]
        )
        rec = {
            "node_idx": node_idx,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": bool(true_label == pred_label),
            "positive_probability": prob_pos,
            "num_subgraph_nodes": int(subset.numel()),
            "num_subgraph_edges": int(sub_edge_index.shape[1]),
            "explain_hops": EXPLAIN_HOPS,
            "explanations": {},
        }
        for name, explainer in [
            ("GradExplainer", grad_exp),
            ("IntegratedGradExplainer", int_exp),
            ("GNNExplainer", gnn_exp),
        ]:
            try:
                if name == "GradExplainer":
                    exp = explainer.get_explanation_node(
                        node_idx=node_idx,
                        x=data.x.clone(),
                        edge_index=data.edge_index,
                        label=data.y,
                        num_hops=EXPLAIN_HOPS,
                    )
                elif name == "IntegratedGradExplainer":
                    exp = explainer.get_explanation_node(
                        node_idx=node_idx,
                        x=data.x.clone(),
                        edge_index=data.edge_index,
                        y=data.y,
                        num_hops=EXPLAIN_HOPS,
                        steps=20,
                    )
                else:
                    exp = explainer.get_explanation_node(
                        node_idx=node_idx,
                        x=data.x.clone(),
                        edge_index=data.edge_index,
                        num_hops=EXPLAIN_HOPS,
                    )
                rec["explanations"][name] = explanation_stats(exp)
            except Exception as exc:
                rec["explanations"][name] = {"error": str(exc)}

        with open(os.path.join(EXPLAIN_DIR, f"node_{node_idx}.json"), "w") as f:
            json.dump(rec, f, indent=2)

        row = {
            "node_idx": node_idx,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": rec["correct"],
            "positive_probability": round(prob_pos, 6),
            "num_subgraph_nodes": rec["num_subgraph_nodes"],
            "num_subgraph_edges": rec["num_subgraph_edges"],
            "any_error": False,
        }
        for name in ["GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]:
            res = rec["explanations"].get(name, {})
            if "error" in res:
                row["any_error"] = True
                row[f"{name}_status"] = "error"
                row[f"{name}_summary"] = res["error"]
            else:
                row[f"{name}_status"] = "ok"
                row[f"{name}_summary"] = json.dumps({
                    k: round(v, 6) for k, v in res.items()
                    if k.endswith("_min") or k.endswith("_max") or k.endswith("_mean") or k.endswith("_std")
                })
        summary_rows.append(row)
        print(f"explained node={node_idx} true={true_label} pred={pred_label} correct={rec['correct']}")

    write_csv(
        os.path.join(EXPLAIN_DIR, "explanations_summary.csv"),
        summary_rows,
        [
            "node_idx", "true_label", "pred_label", "correct", "positive_probability",
            "num_subgraph_nodes", "num_subgraph_edges", "any_error",
            "GradExplainer_status", "GradExplainer_summary",
            "IntegratedGradExplainer_status", "IntegratedGradExplainer_summary",
            "GNNExplainer_status", "GNNExplainer_summary",
        ],
    )

    report = {
        "best_epoch": best_epoch,
        "best_val_metric": round(float(best_metric), 6),
        "epochs_run": len(rows),
        "test_metrics": test_metrics,
        "explanations_requested": EXPLAIN_N,
        "explanations_saved": len(summary_rows),
        "explanation_success": {
            name: sum(1 for row in summary_rows if row[f"{name}_status"] == "ok")
            for name in ["GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]
        },
        "elapsed_seconds": round(time.time() - start, 2),
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
        f.write("Patient Similarity GCN Training Report\n")
        f.write("=" * 44 + "\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Epochs run: {len(rows)}\n")
        f.write(f"Best validation metric: {best_metric:.6f}\n\n")
        f.write("Test metrics\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nGraphXAI explanations\n")
        f.write(f"  saved: {len(summary_rows)}\n")
        for name, cnt in report["explanation_success"].items():
            f.write(f"  {name}: {cnt}/{len(summary_rows)} ok\n")
    shutil.copy2(__file__, os.path.join(OUT_DIR, "train_patient_similarity.py"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
