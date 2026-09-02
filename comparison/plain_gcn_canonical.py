"""Plain-GCN ablation on the CANONICAL split.

Same encoder, same hyperparameters and the same class-weighted training loop as
the ProtGNN run, with the prototype layer switched off (model_args.enable_prot
= False). This isolates what the prototype layer costs or buys in predictive
terms.

protgnn_analysis/train.py cannot be used directly for this: its dataloader
splits ALL graphs in the processed dataset (77,697), while the canonical split
keeps only the 74,511 visits that have a disease label AND at least one
med/symptom/chief-complaint code. Training on the larger set would put the
ablation on a different test fold than ProtGNN and GraphCare. Here the folds
are read from comparison/canonical_split.json instead.

Run:  PYTHONPATH=. python3 -u comparison/plain_gcn_canonical.py
Out:  comparison/plain_gcn/metrics.json, comparison/plain_gcn/predictions.npz
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             top_k_accuracy_score)

from protgnn_analysis.config import data_args, model_args, train_args
from protgnn_analysis.models import GnnNets
from protgnn_analysis.load_dataset import get_dataset

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "comparison" / "plain_gcn"
MAX_EPOCHS = 300
PATIENCE = 10


def canonical_indices(n_graphs):
    """dataset index -> fold, via the CSV row order the graphs were built in."""
    split = json.load(open(REPO / "comparison" / "canonical_split.json"))
    fold, classes = split["fold"], split["classes"]
    df = pd.read_csv(REPO / "data" / "merged_ed.csv", low_memory=False,
                     usecols=["subject_id", "disease_1"])
    assert len(df) == n_graphs, (len(df), n_graphs)
    f = df["subject_id"].astype(str).map(fold)
    idx = {0: [], 1: [], 2: []}
    for i, v in enumerate(f.values):
        if not pd.isna(v):
            idx[int(v)].append(i)
    return idx, classes, df


@torch.no_grad()
def evaluate(model, loader, n_classes):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        logits, prob, _, _, _ = model(batch)
        probs.append(prob.detach().cpu().numpy())
        ys.append(batch.y.view(-1).detach().cpu().numpy())
    p = np.concatenate(probs)
    y = np.concatenate(ys)
    pred = p.argmax(1)
    labels = np.arange(n_classes)
    return y, pred, p, {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "top3_acc": float(top_k_accuracy_score(y, p, k=3, labels=labels)),
        "top5_acc": float(top_k_accuracy_score(y, p, k=5, labels=labels)),
    }


def main():
    data_args.dataset_name = "mimic_intra_patient_disease_cooccur"
    data_args.graph_structure = "cooccur"
    model_args.enable_prot = False          # the ablation

    print("  loading dataset ...", flush=True)
    t0 = time.time()
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name,
                          task=data_args.task, graph_structure="cooccur")
    print(f"  graphs: {len(dataset)}  ({time.time()-t0:.0f}s)", flush=True)

    idx, classes, df = canonical_indices(len(dataset))
    n_classes = int(dataset.num_classes)
    c2i = {c: i for i, c in enumerate(classes)}
    # the canonical class ids must be the dataset's class ids, else the folds
    # would carry different labels than ProtGNN saw
    for i in idx[2][:200]:
        assert int(dataset[i].y.view(-1)[0]) == c2i[str(df["disease_1"].iloc[i])], i
    print(f"  folds: train={len(idx[0])} val={len(idx[1])} test={len(idx[2])} "
          f"classes={n_classes} input_dim={dataset.num_node_features}", flush=True)

    loaders = {
        name: DataLoader([dataset[i] for i in idx[k]],
                         batch_size=train_args.batch_size, shuffle=(k == 0))
        for name, k in (("train", 0), ("val", 1), ("test", 2))
    }

    train_labels = np.array([int(dataset[i].y.view(-1)[0]) for i in idx[0]])
    counts = np.bincount(train_labels, minlength=n_classes).astype(float)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=model_args.device))

    model = GnnNets(dataset.num_node_features, n_classes, model_args)
    model.to_device()
    optimizer = Adam(model.parameters(), lr=train_args.learning_rate,
                     weight_decay=train_args.weight_decay)

    best_acc, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        tot, seen = 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad()
            logits, _, _, _, _ = model(batch)
            loss = criterion(logits, batch.y.view(-1).to(logits.device))
            loss.backward()
            optimizer.step()
            tot += float(loss) * batch.num_graphs
            seen += batch.num_graphs
        _, _, _, m = evaluate(model, loaders["val"], n_classes)
        print(f"  epoch {epoch:3d}  train_loss {tot/seen:.4f}  "
              f"val_acc {m['accuracy']:.4f}  val_macroF1 {m['macro_f1']:.4f}", flush=True)
        if m["accuracy"] > best_acc + 1e-4:
            best_acc, best_epoch, stale = m["accuracy"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"  early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break

    model.load_state_dict(best_state)
    y, pred, prob, test_m = evaluate(model, loaders["test"], n_classes)
    print("  test:", json.dumps(test_m), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump({"test": test_m, "best_epoch": best_epoch, "best_val_acc": best_acc,
               "n_train": len(idx[0]), "n_val": len(idx[1]), "n_test": len(idx[2]),
               "enable_prot": False, "split": "comparison/canonical_split.json"},
              open(OUT_DIR / "metrics.json", "w"), indent=2)
    np.savez_compressed(OUT_DIR / "predictions.npz", y_true=y, y_pred=pred, prob=prob,
                        classes=np.array(classes, dtype=object))
    print("  saved ->", OUT_DIR / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
