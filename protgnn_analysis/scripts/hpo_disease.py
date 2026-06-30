"""
Hyperparameter optimisation for the DISEASE intra-patient hetero-graph GCN.
=========================================================================
Tunes the GCN backbone (the ProtGNN feature extractor) on the 30-class
disease target, optimising **macro-F1** on the validation split — the same
metric train_and_explain.py uses for model selection in multi-class mode.

Why backbone-only (enable_prot=False):
  Prototype projection / MCTS adds large per-epoch cost and mostly affects
  explainability, not raw capacity. Tuning the plain GCN with class-weighted
  cross-entropy is a fast, faithful proxy; transfer the winning architecture
  to the full ProtGNN run.

The dataset is built ONCE and reused across trials (it does not depend on the
searched hyperparameters), so each trial only re-wraps the dataloaders.

Run:
    PYTHONPATH=src:. python3 scripts/hpo_disease.py --n_trials 40 --max_epochs 60
    PYTHONPATH=src:. python3 scripts/hpo_disease.py --dataset mimic_intra_patient_disease

Results:
    outputs/hpo_results/best_params_disease.json
    outputs/hpo_results/study_disease.pkl
"""

import os, sys, json, pickle, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import f1_score, top_k_accuracy_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
for _path in (_PROJECT_ROOT, _SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from protgnn_analysis.load_dataset import get_dataset, get_dataloader
from protgnn_analysis.models import GnnNets
from protgnn_analysis.config import DATA_DIR, OUTPUTS_DIR

DEVICE = 'mps' if torch.backends.mps.is_available() else \
         'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_DIR = str(DATA_DIR)
RESULTS_DIR = os.path.join(str(OUTPUTS_DIR), 'hpo_results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _class_weights(labels, n):
    counts = np.bincount(labels, minlength=n).astype(np.float32)
    w = np.ones(n, dtype=np.float32)
    w[counts > 0] = len(labels) / (n * counts[counts > 0])
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def evaluate(model, loader, criterion, n_classes):
    """Return (val_loss, macro_f1, top5_acc) on a loader."""
    model.eval()
    losses, labels_all, probs_all, preds_all = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            logits, probs, _, _, _ = model(batch)
            losses.append(criterion(logits, batch.y.view(-1)).item())
            probs_all.append(probs.cpu().numpy())
            preds_all.append(logits.argmax(-1).cpu().numpy())
            labels_all.append(batch.y.view(-1).cpu().numpy())
    y_true = np.concatenate(labels_all)
    y_pred = np.concatenate(preds_all)
    y_prob = np.concatenate(probs_all)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    top5 = (top_k_accuracy_score(y_true, y_prob, k=5, labels=np.arange(n_classes))
            if n_classes > 5 else float('nan'))
    return float(np.mean(losses)), float(macro_f1), float(top5)


class ModelArgs:
    """Shim matching the attributes GnnNets reads from model_args
    (verified: latent_dim, mlp_hidden, readout, dropout, emb_normlize,
     adj_normlize, enable_prot, concate, gat_*, model_name, device,
     num_prototypes_per_class)."""
    def __init__(self, latent_dim, mlp_hidden, readout, dropout, emb_normlize):
        self.device = DEVICE
        self.model_name = 'gcn'
        self.latent_dim = latent_dim
        self.mlp_hidden = mlp_hidden
        self.readout = readout
        self.dropout = dropout
        self.emb_normlize = emb_normlize
        self.adj_normlize = True
        self.enable_prot = False          # backbone-only search (fast)
        self.num_prototypes_per_class = 5
        self.concate = False
        self.gat_dropout = 0.6
        self.gat_heads = 10
        self.gat_hidden = 10
        self.gat_concate = True
        self.num_gat_layer = 3


def make_objective(dataset, input_dim, output_dim, max_epochs, seed):

    def objective(trial):
        lr           = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        batch_size   = trial.suggest_categorical('batch_size', [128, 256, 512])
        n_layers     = trial.suggest_int('n_layers', 2, 4)
        hidden_dim   = trial.suggest_categorical('hidden_dim', [64, 96, 128, 256])
        dropout      = trial.suggest_float('dropout', 0.1, 0.6)
        readout      = trial.suggest_categorical('readout', ['mean', 'max', 'sum'])
        emb_norm     = trial.suggest_categorical('emb_norm', [True, False])
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

        latent_dim = [hidden_dim] * n_layers
        mlp_hidden = [hidden_dim // 2]

        loader = get_dataloader(dataset, batch_size, random_split_flag=True,
                                data_split_ratio=[0.8, 0.1, 0.1], seed=seed)

        margs = ModelArgs(latent_dim, mlp_hidden, readout, dropout, emb_norm)
        model = GnnNets(input_dim, output_dim, margs)
        model.to_device()

        train_labels = np.array([int(dataset[i].y.view(-1)[0].item())
                                 for i in loader['train'].dataset.indices])
        criterion = nn.CrossEntropyLoss(weight=_class_weights(train_labels, output_dim))
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_f1, no_improve, PATIENCE = -1.0, 0, 8
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

            _, val_f1, _ = evaluate(model, loader['eval'], criterion, output_dim)
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if val_f1 > best_f1:
                best_f1, no_improve = val_f1, 0
            else:
                no_improve += 1
            if no_improve >= PATIENCE:
                break
        return best_f1

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mimic_intra_patient_disease')
    parser.add_argument('--n_trials', type=int, default=40)
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subsample', type=int, default=0,
                        help="train HPO on a random N-graph subset for speed "
                             "(0 = full dataset). Architecture search does not "
                             "need all 60k graphs.")
    args = parser.parse_args()

    print(f"Device  : {DEVICE}")
    print(f"Dataset : {args.dataset}")
    print(f"Trials  : {args.n_trials}  |  Max epochs/trial: {args.max_epochs}")

    print("\nBuilding dataset once (reused across trials)...")
    full = get_dataset(DATASET_DIR, args.dataset)
    input_dim = full.num_node_features
    output_dim = int(full.num_classes)
    print(f"  graphs={len(full)}  classes={output_dim}  feat_dim={input_dim}")

    dataset = full
    if args.subsample and args.subsample < len(full):
        from torch.utils.data import Subset
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(full), size=args.subsample, replace=False).tolist()
        dataset = Subset(full, idx)
        print(f"  HPO subsample: {args.subsample} graphs (faster search)")
    print()

    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=12)
    study = optuna.create_study(direction='maximize', pruner=pruner,
                                study_name='mimic_disease_hpo')
    study.optimize(make_objective(dataset, input_dim, output_dim,
                                  args.max_epochs, args.seed),
                   n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print("\n" + "=" * 55)
    print("BEST TRIAL (validation macro-F1)")
    print("=" * 55)
    print(f"  macro_F1 : {best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k:<20}: {v}")

    out = os.path.join(RESULTS_DIR, 'best_params_disease.json')
    with open(out, 'w') as f:
        json.dump({'macro_f1': best.value, 'params': best.params}, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'study_disease.pkl'), 'wb') as f:
        pickle.dump(study, f)
    print(f"\nSaved to {out}")

    print("\nTop 5 trials:")
    trials = sorted([t for t in study.trials if t.value is not None],
                    key=lambda t: t.value, reverse=True)
    for t in trials[:5]:
        print(f"  Trial {t.number:3d}  macroF1={t.value:.4f}  {t.params}")


if __name__ == '__main__':
    main()
