"""ProtGNN on the CANONICAL split, with test predictions saved.

Same purpose as comparison/plain_gcn_canonical.py, but with the prototype layer
ENABLED. The training loop mirrors protgnn_analysis/train.py: warm-up epochs
with a frozen last layer, cluster / margin-separation / L1 losses on top of the
class-weighted cross-entropy, gradient clipping, MCTS prototype projection from
epoch `proj_epochs`, early stopping on validation macro-F1, and the same
"prefer the latest post-projection checkpoint" selection rule.

It exists because train.py splits every graph in the processed dataset
(77,697) while the canonical split keeps 74,511, so a train.py run cannot be
compared against GraphCare or the tabular baselines. This script also stores
what the earlier runs did not: the checkpoint and the per-graph test
predictions needed for the confusion matrix and the per-class table.

Run:  PYTHONPATH=. python3 -u comparison/protgnn_canonical.py
Out:  comparison/protgnn_canonical/{metrics.json,predictions.npz,gcn_{best,latest}.pth}
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             top_k_accuracy_score)

from protgnn_analysis.config import data_args, model_args, train_args
from protgnn_analysis.models import GnnNets
from protgnn_analysis.load_dataset import get_dataset
from protgnn_analysis.my_mcts import mcts

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "comparison" / "protgnn_canonical"
CLST, SEP, MARGIN = 0.1, 0.1, 1.0        # train.py defaults


def canonical_indices(n_graphs):
    split = json.load(open(REPO / "comparison" / "canonical_split.json"))
    fold = split["fold"]
    df = pd.read_csv(REPO / "data" / "merged_ed.csv", low_memory=False,
                     usecols=["subject_id", "disease_1"])
    assert len(df) == n_graphs, (len(df), n_graphs)
    f = df["subject_id"].astype(str).map(fold)
    idx = {0: [], 1: [], 2: []}
    for i, v in enumerate(f.values):
        if not pd.isna(v):
            idx[int(v)].append(i)
    return idx, split["classes"], df


@torch.no_grad()
def evaluate(model, loader, n_classes):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        _, prob, _, _, _ = model(batch)
        probs.append(prob.detach().cpu().numpy())
        ys.append(batch.y.view(-1).detach().cpu().numpy())
    p, y = np.concatenate(probs), np.concatenate(ys)
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


def project_prototypes(model, dataset, train_idx, n_classes, epoch):
    """MCTS projection: replace each prototype with the closest real subgraph."""
    model.eval()
    candidates = list(train_idx)
    random.shuffle(candidates)
    n_prot = n_classes * model_args.num_prototypes_per_class
    t0 = time.time()
    print(f"\n  prototype projection (epoch {epoch}): {n_prot} prototypes ...", flush=True)
    for proto_i in range(n_prot):
        cls = proto_i // model_args.num_prototypes_per_class
        seen, best_sim, best_prot = 0, 0.0, None
        for j in candidates:
            d = dataset[j]
            if int(d.y.view(-1)[0].item()) != cls:
                continue
            seen += 1
            _, sim, prot = mcts(d, model, model.model.prototype_vectors[proto_i])
            if sim > best_sim:
                best_sim, best_prot = sim, prot
            if seen >= train_args.nearest_graphs:
                break
        if best_prot is not None:
            model.model.prototype_vectors.data[proto_i] = best_prot
        done = proto_i + 1
        el = time.time() - t0
        print(f"\r    {done:3d}/{n_prot} | class {cls:2d} | best_sim {best_sim:.3f} | "
              f"{el:5.0f}s | ETA {el/done*(n_prot-done):5.0f}s   ", end="", flush=True)
    print(f"\n  projection done in {time.time()-t0:.0f}s", flush=True)


def main():
    data_args.dataset_name = "mimic_intra_patient_disease_cooccur"
    data_args.graph_structure = "cooccur"
    model_args.enable_prot = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("  loading dataset ...", flush=True)
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name,
                          task=data_args.task, graph_structure="cooccur")
    idx, classes, df = canonical_indices(len(dataset))
    n_classes = int(dataset.num_classes)
    c2i = {c: i for i, c in enumerate(classes)}
    for i in idx[2][:200]:
        assert int(dataset[i].y.view(-1)[0]) == c2i[str(df["disease_1"].iloc[i])], i
    print(f"  folds: train={len(idx[0])} val={len(idx[1])} test={len(idx[2])} "
          f"classes={n_classes} input_dim={dataset.num_node_features}", flush=True)

    loaders = {name: DataLoader([dataset[i] for i in idx[k]],
                                batch_size=train_args.batch_size, shuffle=(k == 0))
               for name, k in (("train", 0), ("val", 1), ("test", 2))}

    train_labels = np.array([int(dataset[i].y.view(-1)[0]) for i in idx[0]])
    counts = np.bincount(train_labels, minlength=n_classes).astype(np.float32)
    w = np.ones(n_classes, dtype=np.float32)
    w[counts > 0] = len(train_labels) / (n_classes * counts[counts > 0])
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=model_args.device))

    model = GnnNets(dataset.num_node_features, n_classes, model_args)
    model.to_device()
    optimizer = Adam(model.parameters(), lr=train_args.learning_rate,
                     weight_decay=train_args.weight_decay)

    best_sel, best_patience, stale, best_epoch = -1.0, -1.0, 0, -1
    history = []
    for epoch in range(train_args.max_epochs):
        if (epoch >= train_args.proj_epochs
                and (epoch - train_args.proj_epochs) % train_args.proj_interval == 0):
            project_prototypes(model, dataset, idx[0], n_classes, epoch)
            torch.save({"net": model.state_dict(), "epoch": epoch},
                       OUT_DIR / "gcn_latest.pth")

        model.train()
        # warm-up: train encoder + prototypes, freeze the classification layer
        if epoch < train_args.warm_epochs:
            for p in model.model.gnn_layers.parameters():
                p.requires_grad = True
            model.model.prototype_vectors.requires_grad = True
            for p in model.model.last_layer.parameters():
                p.requires_grad = False
        else:
            for p in model.parameters():
                p.requires_grad = True

        losses = []
        for batch in loaders["train"]:
            logits, _, _, _, min_dist = model(batch)
            y = batch.y.view(-1).to(logits.device)
            loss = criterion(logits, y)

            if len(min_dist) > 0:
                ident = model.model.prototype_class_identity.to(min_dist.device)
                right = torch.t(ident[:, y].bool()).to(min_dist.device)
                cluster = torch.mean(torch.min(
                    min_dist[right].reshape(-1, model_args.num_prototypes_per_class), dim=1)[0])
                wrong_min = torch.min(
                    min_dist[~right].reshape(
                        -1, (n_classes - 1) * model_args.num_prototypes_per_class), dim=1)[0]
                sep = torch.mean(torch.clamp(MARGIN - wrong_min, min=0.0))
                l1 = (model.model.last_layer.weight * (1 - torch.t(ident).to(min_dist.device))).norm(p=1)
                total = loss + CLST * cluster + SEP * sep + 5e-4 * l1
                if not torch.isfinite(total):
                    total = loss
            else:
                total = loss

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
            optimizer.step()
            losses.append(float(total.detach()))

        _, _, _, m = evaluate(model, loaders["val"], n_classes)
        sel = m["macro_f1"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **m})
        print(f"  epoch {epoch:3d}  train_loss {np.mean(losses):.4f}  "
              f"val_acc {m['accuracy']:.4f}  val_macroF1 {sel:.4f}", flush=True)

        torch.save({"net": model.state_dict(), "epoch": epoch, "sel": sel},
                   OUT_DIR / "gcn_latest.pth")
        if sel > best_sel:
            best_sel, best_epoch = sel, epoch
            torch.save({"net": model.state_dict(), "epoch": epoch, "sel": sel},
                       OUT_DIR / "gcn_best.pth")
        if sel > best_patience + train_args.early_stop_min_delta:
            best_patience, stale = sel, 0
        else:
            stale += 1
            if stale > train_args.early_stopping:
                print(f"  early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break

    # train.py rule: once projection has run, the projected (latest) weights are
    # the model whose prototypes correspond to real training subgraphs
    projected = history and history[-1]["epoch"] >= train_args.proj_epochs
    ckpt_file = "gcn_latest.pth" if projected else "gcn_best.pth"
    ckpt = torch.load(OUT_DIR / ckpt_file, map_location=model_args.device)
    model.update_state_dict(ckpt["net"])
    print(f"  selected checkpoint: {ckpt_file} (epoch {ckpt['epoch']})", flush=True)

    y, pred, prob, test_m = evaluate(model, loaders["test"], n_classes)
    print("  test:", json.dumps(test_m), flush=True)

    json.dump({"test": test_m, "selected_checkpoint": ckpt_file,
               "selected_epoch": int(ckpt["epoch"]), "best_val_macro_f1": best_sel,
               "n_train": len(idx[0]), "n_val": len(idx[1]), "n_test": len(idx[2]),
               "enable_prot": True, "clst": CLST, "sep": SEP, "margin": MARGIN,
               "split": "comparison/canonical_split.json", "history": history},
              open(OUT_DIR / "metrics.json", "w"), indent=2)
    np.savez_compressed(OUT_DIR / "predictions.npz", y_true=y, y_pred=pred, prob=prob,
                        test_indices=np.array(idx[2]), classes=np.array(classes, dtype=object))
    print("  saved ->", OUT_DIR / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
