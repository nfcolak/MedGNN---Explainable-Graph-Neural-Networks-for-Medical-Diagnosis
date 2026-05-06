"""
Patient Similarity GCN — Node Classification
=============================================
Each patient = one node in a large graph.
Edges connect the k most similar patients (cosine similarity).
GCN propagates information across similar patients → node classification.

Run:
    python3 train_patient_similarity.py [--k 15] [--epochs 200] [--hidden 128]
"""

import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    classification_report
)
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..","..","datasets/data/merged_ed_no_icd_sample_20k.csv"
)


# ── Model ─────────────────────────────────────────────────────────────────────

class PatientGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        return self.classifier(x)


# ── Data ──────────────────────────────────────────────────────────────────────

def build_graph(k):
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=["subject_id"], errors="ignore")

    y = torch.tensor(df["disposition"].values, dtype=torch.long)
    df = df.drop(columns=["disposition"])

    # z-score normalize continuous columns
    feat_df = df.copy()
    for col in feat_df.columns:
        if feat_df[col].nunique() > 2:
            std = feat_df[col].std()
            if std > 0:
                feat_df[col] = (feat_df[col] - feat_df[col].mean()) / std
    feat_df = feat_df.fillna(0)

    X = feat_df.values.astype(np.float32)
    print(f"  Patients: {X.shape[0]}  Features: {X.shape[1]}")

    # k-NN graph (cosine similarity)
    print(f"  Building k={k} nearest-neighbour graph (cosine)...")
    t0 = time.time()
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric="cosine", n_jobs=-1)
    nn_model.fit(X)
    distances, indices = nn_model.kneighbors(X)
    print(f"  k-NN done in {time.time()-t0:.1f}s")

    # build edge list (undirected, skip self-loops)
    rows, cols = [], []
    n = X.shape[0]
    for i in range(n):
        for j in indices[i][1:]:   # skip self (index 0)
            rows.append(i); cols.append(j)
            rows.append(j); cols.append(i)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    # deduplicate
    edge_index = torch.unique(edge_index, dim=1)

    print(f"  Edges (undirected): {edge_index.shape[1] // 2:,}")

    x = torch.tensor(X, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y), feat_df.columns.tolist()


def stratified_masks(y, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)

    for cls in [0, 1]:
        idx = np.where(y.numpy() == cls)[0]
        rng.shuffle(idx)
        t = int(len(idx) * train_ratio)
        v = int(len(idx) * val_ratio)
        train_mask[idx[:t]] = True
        val_mask[idx[t:t+v]] = True
        test_mask[idx[t+v:]] = True

    return train_mask, val_mask, test_mask


# ── Training ──────────────────────────────────────────────────────────────────

def evaluate(model, data, mask, criterion):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        loss   = criterion(logits[mask], data.y[mask]).item()
        probs  = torch.softmax(logits[mask], dim=1).cpu().numpy()
        preds  = probs.argmax(1)
        labels = data.y[mask].cpu().numpy()
    acc    = accuracy_score(labels, preds)
    f1     = f1_score(labels, preds, average="macro", zero_division=0)
    rocauc = roc_auc_score(labels, probs[:, 1])
    prauc  = average_precision_score(labels, probs[:, 1])
    return {"loss": loss, "acc": acc, "f1": f1, "roc_auc": rocauc, "pr_auc": prauc,
            "labels": labels, "preds": preds, "probs": probs[:, 1]}


def train(args):
    data, feat_cols = build_graph(args.k)

    train_mask, val_mask, test_mask = stratified_masks(data.y)
    print(f"\n  Train: {train_mask.sum()}  Val: {val_mask.sum()}  Test: {test_mask.sum()}")

    # class weights for imbalance
    train_labels = data.y[train_mask].numpy()
    counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    weights = torch.tensor(len(train_labels) / (2 * counts), dtype=torch.float)

    model    = PatientGCN(data.num_node_features, args.hidden, 2, dropout=args.dropout)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    print(f"\n  Input dim: {data.num_node_features}  Hidden: {args.hidden}")
    print(f"  Class weights: HOME={weights[0]:.3f}  ADMITTED={weights[1]:.3f}\n")

    best_prauc = -1
    patience   = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss   = criterion(logits[train_mask], data.y[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_state = evaluate(model, data, val_mask, criterion)
            print(
                f"  Epoch {epoch+1:4d} | Train Loss: {loss.item():.4f} | "
                f"Val Acc: {val_state['acc']:.4f}  PR-AUC: {val_state['pr_auc']:.4f}  "
                f"ROC-AUC: {val_state['roc_auc']:.4f}"
            )
            if val_state["pr_auc"] > best_prauc:
                best_prauc = val_state["pr_auc"]
                best_state = {k: v.clone() if torch.is_tensor(v) else v
                              for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= args.patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    # final test
    model.load_state_dict(best_state)
    test_state = evaluate(model, data, test_mask, criterion)

    print("\n" + "="*55)
    print("TEST RESULTS")
    print("="*55)
    print(f"Accuracy  : {test_state['acc']:.4f}")
    print(f"F1 (macro): {test_state['f1']:.4f}")
    print(f"ROC-AUC   : {test_state['roc_auc']:.4f}")
    print(f"PR-AUC    : {test_state['pr_auc']:.4f}")
    print()
    print(classification_report(
        test_state["labels"], test_state["preds"],
        target_names=["HOME", "ADMITTED"]
    ))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k",        type=int,   default=15,    help="k-NN neighbours")
    parser.add_argument("--hidden",   type=int,   default=128,   help="GCN hidden dim")
    parser.add_argument("--epochs",   type=int,   default=300,   help="Max epochs")
    parser.add_argument("--lr",       type=float, default=0.001, help="Learning rate")
    parser.add_argument("--wd",       type=float, default=1e-4,  help="Weight decay")
    parser.add_argument("--dropout",  type=float, default=0.5,   help="Dropout")
    parser.add_argument("--patience", type=int,   default=5,     help="Early stopping patience (×10 epochs)")
    args = parser.parse_args()
    train(args)
