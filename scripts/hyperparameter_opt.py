"""
Hyperparameter Optimisation for Patient Similarity GCN
=======================================================
Uses Optuna with MedianPruner to efficiently search:
    k (neighbours), lr, batch_size, hidden layers,
    dropout, readout, emb_normalise

Optimises for PR-AUC on the validation set.
Each trial trains for at most --max_epochs_per_trial epochs
with early stopping.

Run:
    python3 scripts/hyperparameter_opt.py [--n_trials 50] [--max_epochs 80]

Results saved to:
    outputs/hpo_results/best_params.json
    outputs/hpo_results/study.pkl
"""

import os, sys, json, pickle, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import optuna
from optuna.pruners import MedianPruner

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _path in (_PROJECT_ROOT, _SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from prot_gnn.load_dataset import get_dataset, get_dataloader
from prot_gnn.models import GnnNets
from configs.config import DATA_DIR, OUTPUTS_DIR

DEVICE = 'mps' if torch.backends.mps.is_available() else \
         'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_DIR  = str(DATA_DIR)
RESULTS_DIR  = os.path.join(str(OUTPUTS_DIR), 'hpo_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _class_weights(labels, n):
    counts = np.bincount(labels, minlength=n).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    w[counts > 0] = len(labels) / (n * counts[counts > 0])
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)

def _average_precision(y_true, y_score):
    order  = np.argsort(-y_score)
    y_true = np.asarray(y_true)[order]
    if y_true.sum() == 0:
        return 0.0
    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    prec = tp / np.maximum(tp + fp, 1)
    rec  = tp / y_true.sum()
    prec = np.concatenate(([1.0], prec))
    rec  = np.concatenate(([0.0], rec))
    return float(np.sum((rec[1:] - rec[:-1]) * prec[1:]))

def evaluate(model, loader, criterion):
    model.eval()
    losses, labels_all, probs_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits, probs, _, _, _ = model(batch)
            losses.append(criterion(logits, batch.y.view(-1)).item())
            probs_all.append(probs.cpu().numpy())
            labels_all.append(batch.y.cpu().numpy())
    y_true = np.concatenate(labels_all)
    y_prob = np.concatenate(probs_all)
    pr_auc = _average_precision(y_true, y_prob[:, 1])
    return float(np.mean(losses)), pr_auc


# ── model args shim ───────────────────────────────────────────────────────────

class ModelArgs:
    def __init__(self, latent_dim, mlp_hidden, readout, dropout,
                 emb_normlize, adj_normlize=True, enable_prot=False,
                 num_prototypes_per_class=5):
        self.device               = DEVICE
        self.model_name           = 'gcn'
        self.latent_dim           = latent_dim
        self.mlp_hidden           = mlp_hidden
        self.readout              = readout
        self.dropout              = dropout
        self.emb_normlize         = emb_normlize
        self.adj_normlize         = adj_normlize
        self.enable_prot          = enable_prot
        self.num_prototypes_per_class = num_prototypes_per_class
        self.concate              = False
        self.gnn_dropout          = 0.0
        self.gat_dropout          = 0.6
        self.gat_heads            = 10
        self.gat_hidden           = 10
        self.gat_concate          = True
        self.num_gat_layer        = 3


# ── objective ─────────────────────────────────────────────────────────────────

def make_objective(max_epochs):
    def objective(trial):
        # ── search space ──────────────────────────────────────────────────────
        k           = trial.suggest_categorical('k',          [5, 10, 15, 20])
        lr          = trial.suggest_float('lr',               1e-4, 5e-3, log=True)
        batch_size  = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
        n_layers    = trial.suggest_int('n_layers',           2, 4)
        hidden_dim  = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        dropout     = trial.suggest_float('dropout',          0.1, 0.5)
        readout     = trial.suggest_categorical('readout',    ['mean', 'max', 'sum'])
        emb_norm    = trial.suggest_categorical('emb_norm',   [True, False])
        weight_decay= trial.suggest_float('weight_decay',     1e-5, 1e-3, log=True)

        latent_dim  = [hidden_dim] * n_layers
        mlp_hidden  = [hidden_dim // 2]

        dataset_name = f'mimic_patient_sim_k{k}'

        # ── data ──────────────────────────────────────────────────────────────
        dataset = get_dataset(DATASET_DIR, dataset_name)
        input_dim  = dataset.num_node_features
        output_dim = int(dataset.num_classes)

        loader = get_dataloader(dataset, batch_size,
                                random_split_flag=True,
                                data_split_ratio=[0.8, 0.1, 0.1],
                                seed=42)

        # ── model ─────────────────────────────────────────────────────────────
        margs = ModelArgs(latent_dim, mlp_hidden, readout, dropout, emb_norm)
        model = GnnNets(input_dim, output_dim, margs)
        model.to_device()

        train_labels = np.array([int(dataset[i].y.view(-1)[0].item())
                                  for i in loader['train'].dataset.indices])
        criterion  = nn.CrossEntropyLoss(weight=_class_weights(train_labels, output_dim))
        optimizer  = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_pr_auc    = -1.0
        no_improve     = 0
        PATIENCE       = 10

        for epoch in range(max_epochs):
            model.train()
            for batch in loader['train']:
                batch = batch.to(DEVICE)
                logits, _, _, _, _ = model(batch)
                loss = criterion(logits, batch.y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            val_loss, val_pr_auc = evaluate(model, loader['eval'], criterion)

            trial.report(val_pr_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                no_improve  = 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                break

        return best_pr_auc

    return objective


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials',   type=int, default=50)
    parser.add_argument('--max_epochs', type=int, default=80)
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Trials : {args.n_trials}  |  Max epochs/trial: {args.max_epochs}\n")

    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=15)
    study  = optuna.create_study(direction='maximize', pruner=pruner,
                                  study_name='mimic_patient_sim_hpo')

    study.optimize(make_objective(args.max_epochs),
                   n_trials=args.n_trials,
                   show_progress_bar=True)

    best = study.best_trial
    print("\n" + "="*55)
    print("BEST TRIAL")
    print("="*55)
    print(f"  PR-AUC : {best.value:.4f}")
    print(f"  Params :")
    for k, v in best.params.items():
        print(f"    {k:<20}: {v}")

    # save results
    with open(os.path.join(RESULTS_DIR, 'best_params.json'), 'w') as f:
        json.dump({'pr_auc': best.value, 'params': best.params}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'study.pkl'), 'wb') as f:
        pickle.dump(study, f)

    print(f"\nSaved to {os.path.join(RESULTS_DIR, 'best_params.json')}")

    # print top 5
    print("\nTop 5 trials:")
    trials = sorted(study.trials, key=lambda t: t.value or -1, reverse=True)
    for t in trials[:5]:
        if t.value is not None:
            print(f"  Trial {t.number:3d}  PR-AUC={t.value:.4f}  {t.params}")


if __name__ == '__main__':
    main()
