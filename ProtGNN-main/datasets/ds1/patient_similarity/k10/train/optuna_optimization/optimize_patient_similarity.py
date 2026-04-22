import csv
import json
import os
import random
import shutil
import time

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import Adam
from torch_geometric.nn import GCNConv, SAGEConv


DATA_PATH = "datasets/ds1/patient_similarity/k10/processed/k10_ps_data.pt"
BASE_TRAIN_DIR = "datasets/ds1/patient_similarity/k10/train"
OUT_DIR = os.path.join(BASE_TRAIN_DIR, "optuna_optimization")
STUDY_DB = os.path.join(OUT_DIR, "optuna_study.db")
N_TRIALS = 25
MAX_EPOCHS = 140
EARLY_STOPPING = 25
SEED = 11


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_splits():
    with open(os.path.join(BASE_TRAIN_DIR, "split_indices.json")) as f:
        splits = json.load(f)
    return (
        torch.tensor(splits["train"], dtype=torch.long),
        torch.tensor(splits["val"], dtype=torch.long),
        torch.tensor(splits["test"], dtype=torch.long),
    )


class TunableGNN(nn.Module):
    def __init__(self, model_type, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.model_type = model_type
        self.dropout = dropout
        conv = GCNConv if model_type == "gcn" else SAGEConv
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.convs = nn.ModuleList()
        for i in range(len(dims) - 1):
            if model_type == "gcn":
                self.convs.append(conv(dims[i], dims[i + 1], normalize=True))
            else:
                self.convs.append(conv(dims[i], dims[i + 1]))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            if self.model_type == "gcn":
                x = conv(x, edge_index, edge_weight=edge_weight)
            else:
                x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def edge_weight_for_mode(data, mode):
    if mode == "none":
        return None
    w = data.edge_attr.view(-1)
    if mode == "similarity":
        return w
    if mode == "squared":
        return w.pow(2)
    if mode == "threshold_0_7":
        return torch.where(w >= 0.7, w, torch.zeros_like(w))
    raise ValueError(mode)


def metrics_from_scores(labels, scores, threshold=0.5):
    labels = labels.detach().cpu().numpy().astype(int)
    scores = np.asarray(scores, dtype=float)
    preds = (scores >= threshold).astype(int)
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
        pr_auc = float(average_precision_score(labels, scores))
    except ValueError:
        pr_auc = float("nan")
    try:
        roc_auc = float(roc_auc_score(labels, scores))
    except ValueError:
        roc_auc = float("nan")
    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_acc": (recall + specificity) / 2,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def tune_threshold(labels, scores, objective="f1"):
    best_t, best_val = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        m = metrics_from_scores(labels, scores, float(t))
        val = m[objective]
        if val > best_val:
            best_t, best_val = float(t), float(val)
    return best_t, best_val


def evaluate(model, data, idx, edge_weight, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, edge_weight=edge_weight)
        loss = criterion(logits[idx], data.y[idx])
        scores = torch.softmax(logits[idx], dim=-1)[:, 1].detach().cpu().numpy()
    m = metrics_from_scores(data.y[idx], scores, 0.5)
    m["loss"] = float(loss.item())
    return m, logits


def train_once(params, data, train_idx, val_idx, seed=SEED, return_state=False):
    set_seed(seed)
    model = TunableGNN(
        params["model_type"],
        data.x.shape[1],
        params["hidden_dim"],
        int(data.y.max().item()) + 1,
        params["num_layers"],
        params["dropout"],
    )
    edge_weight = edge_weight_for_mode(data, params["edge_weight_mode"])
    counts = torch.bincount(data.y[train_idx], minlength=2).float()
    weights = len(train_idx) / (2 * counts.clamp_min(1))
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    best_score = -1.0
    best_epoch = -1
    best_state = None
    patience = 0
    history = []

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index, edge_weight=edge_weight)
        loss = criterion(logits[train_idx], data.y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        val_m, _ = evaluate(model, data, val_idx, edge_weight, criterion)
        score = val_m["pr_auc"]
        history.append({"epoch": epoch, "train_loss": float(loss.item()), **{f"val_{k}": v for k, v in val_m.items() if isinstance(v, (int, float))}})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience > EARLY_STOPPING:
            break

    if return_state:
        model.load_state_dict(best_state)
        return best_score, best_epoch, model, edge_weight, criterion, history
    return best_score, best_epoch, history[-1]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    data = torch.load(DATA_PATH, map_location="cpu")
    train_idx, val_idx, test_idx = load_splits()

    def objective(trial):
        model_type = trial.suggest_categorical("model_type", ["gcn", "sage"])
        params = {
            "model_type": model_type,
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 2, 4),
            "dropout": trial.suggest_float("dropout", 0.1, 0.65),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True),
            "edge_weight_mode": "none",
        }
        if model_type == "gcn":
            params["edge_weight_mode"] = trial.suggest_categorical("edge_weight_mode", ["none", "similarity", "squared", "threshold_0_7"])
        score, best_epoch, last = train_once(params, data, train_idx, val_idx, seed=SEED + trial.number)
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("last_val_balanced_acc", last.get("val_balanced_acc"))
        trial.set_user_attr("last_val_f1", last.get("val_f1"))
        return score

    study = optuna.create_study(
        study_name="patient_similarity_gnn",
        direction="maximize",
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
    )
    study.optimize(objective, n_trials=N_TRIALS)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs"))
    trials_df.to_csv(os.path.join(OUT_DIR, "optuna_trials.csv"), index=False)

    best_params = dict(study.best_trial.params)
    if best_params["model_type"] != "gcn":
        best_params["edge_weight_mode"] = "none"
    best_score, best_epoch, model, edge_weight, criterion, history = train_once(
        best_params, data, train_idx, val_idx, seed=SEED, return_state=True
    )
    val_m, val_logits = evaluate(model, data, val_idx, edge_weight, criterion)
    test_m_default, test_logits = evaluate(model, data, test_idx, edge_weight, criterion)
    val_scores = torch.softmax(val_logits[val_idx], dim=-1)[:, 1].detach().cpu().numpy()
    test_scores = torch.softmax(test_logits[test_idx], dim=-1)[:, 1].detach().cpu().numpy()
    f1_t, _ = tune_threshold(data.y[val_idx], val_scores, "f1")
    bal_t, _ = tune_threshold(data.y[val_idx], val_scores, "balanced_acc")
    test_m_f1_threshold = metrics_from_scores(data.y[test_idx], test_scores, f1_t)
    test_m_bal_threshold = metrics_from_scores(data.y[test_idx], test_scores, bal_t)

    with open(os.path.join(OUT_DIR, "best_training_history.csv"), "w", newline="") as f:
        fieldnames = sorted(set().union(*[r.keys() for r in history]))
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_params": best_params,
            "best_epoch": best_epoch,
            "best_val_pr_auc": best_score,
            "val_metrics_default_threshold": val_m,
            "test_metrics_default_threshold": test_m_default,
            "test_metrics_val_f1_threshold": test_m_f1_threshold,
            "test_metrics_val_balanced_acc_threshold": test_m_bal_threshold,
        },
        os.path.join(OUT_DIR, "best_model_checkpoint.pt"),
    )

    report = {
        "tool": "Optuna",
        "optuna_version": optuna.__version__,
        "n_trials": len(study.trials),
        "objective": "maximize validation PR-AUC",
        "best_trial_number": study.best_trial.number,
        "best_value_val_pr_auc": study.best_value,
        "best_params": best_params,
        "best_retrain_epoch": best_epoch,
        "val_metrics_default_threshold": val_m,
        "test_metrics_default_threshold": test_m_default,
        "test_metrics_threshold_tuned_for_val_f1": test_m_f1_threshold,
        "test_metrics_threshold_tuned_for_val_balanced_acc": test_m_bal_threshold,
    }
    with open(os.path.join(OUT_DIR, "optimization_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(OUT_DIR, "optimization_report.txt"), "w") as f:
        f.write("Patient Similarity Hyperparameter Optimization\n")
        f.write("=" * 52 + "\n")
        f.write(f"Tool: Optuna {optuna.__version__}\n")
        f.write(f"Trials: {len(study.trials)}\n")
        f.write("Objective: maximize validation PR-AUC\n")
        f.write(f"Best trial: {study.best_trial.number}\n")
        f.write(f"Best validation PR-AUC: {study.best_value:.6f}\n")
        f.write(f"Best params: {json.dumps(best_params)}\n\n")
        f.write("Test metrics, threshold=0.5\n")
        for k, v in test_m_default.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTest metrics, threshold tuned on validation F1\n")
        for k, v in test_m_f1_threshold.items():
            f.write(f"  {k}: {v}\n")
    shutil.copy2(__file__, os.path.join(OUT_DIR, "optimize_patient_similarity.py"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
