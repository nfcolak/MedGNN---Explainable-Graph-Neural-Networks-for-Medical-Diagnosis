"""
Hyperparameter optimisation for the DISEASE intra-patient hetero-graph GCN.
=========================================================================
Tunes the GCN backbone (the ProtGNN feature extractor) on the disease
target, optimising **macro-F1** on the validation split — the same metric
train.py uses for model selection in multi-class mode.

Brought up to the current cross-method comparison standard (matches
gsat_analysis/scripts/hpo_gsat.py):
  - canonical split (comparison/canonical_split.json), REQUIRED — not the old
    ad-hoc random 80/10/10 split this script used to build itself. Train/val
    stay the exact same patients GSAT/GraphCare's HPO and the 3-seed matrix
    already use.
  - sqrt_inverse class weighting (the policy locked in for this comparison),
    not the old hardcoded full-inverse weighting.
  - explicit --graph_structure (default star), passed straight to
    get_dataset/get_dataloader instead of being inferred from --dataset name
    tricks.
Everything else that already worked is kept as-is: Optuna + MedianPruner,
the backbone-only speed trick, the --subsample option.

Why backbone-only (enable_prot=False):
  Prototype projection / MCTS adds large per-epoch cost and mostly affects
  explainability, not raw capacity. Tuning the plain GCN with class-weighted
  cross-entropy is a fast, faithful proxy; transfer the winning architecture
  to the full ProtGNN run.

The dataset is built ONCE and reused across trials (it does not depend on the
searched hyperparameters), so each trial only re-wraps the dataloaders.
output_dim is read from the dataset at runtime (not hardcoded), so this keeps
working unchanged if/when the class count changes.

Run:
    PYTHONPATH=. .venv-protgnn/bin/python3 -m protgnn_analysis.scripts.hpo_disease \
        --graph_structure star --n_trials 40 --max_epochs 60 --subsample 20000

Results:
    protgnn_analysis/outputs/hpo_results/best_params_disease_<structure>.json
    protgnn_analysis/outputs/hpo_results/study_disease_<structure>.pkl
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
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
for _path in (_PROJECT_ROOT,):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from shared.lib.benchmark_contract import load_canonical_split
from shared.lib.graph_structures import resolve as resolve_structure
# Reused verbatim (not reimplemented) so ProtGNN's HPO weights classes with the
# exact same formula GSAT/GraphCare's HPO and the 3-seed matrix use.
from comparison.standardized.performance_review import class_weights as _class_weight_policy
from protgnn_analysis.load_dataset import get_dataset, get_dataloader
from protgnn_analysis.models import GnnNets
from protgnn_analysis.config import DATA_DIR, OUTPUTS_DIR

DEVICE = 'mps' if torch.backends.mps.is_available() else \
         'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_DIR = str(DATA_DIR)
RESULTS_DIR = os.path.join(str(OUTPUTS_DIR), 'hpo_results')
LOSS_WEIGHTING = "sqrt_inverse"  # this comparison's locked-in policy; not a CLI choice


def _class_weights(labels, n):
    weights = _class_weight_policy(labels, n, LOSS_WEIGHTING)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


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


def make_objective(train_dataset, val_loader, input_dim, output_dim, max_epochs, batch_size_cap=None):
    """train_dataset/val_loader come from ONE canonical split, built once.
    Each trial only re-wraps train_dataset in a fresh-batch-size DataLoader."""
    from torch_geometric.loader import DataLoader

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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        margs = ModelArgs(latent_dim, mlp_hidden, readout, dropout, emb_norm)
        model = GnnNets(input_dim, output_dim, margs)
        model.to_device()

        train_labels = np.array([int(g.y.view(-1)[0].item()) for g in train_dataset])
        criterion = nn.CrossEntropyLoss(weight=_class_weights(train_labels, output_dim))
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_f1, no_improve, PATIENCE = -1.0, 0, 8
        for epoch in range(max_epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(DEVICE)
                logits, _, _, _, _ = model(batch)
                loss = criterion(logits, batch.y.view(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            _, val_f1, _ = evaluate(model, val_loader, criterion, output_dim)
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
    parser.add_argument('--graph_structure', '--graph', default='star',
                        help='star|cooccur|ontology|full')
    parser.add_argument('--canonical_split', default='comparison/canonical_split.json')
    parser.add_argument('--n_trials', type=int, default=40)
    parser.add_argument('--max_epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--subsample', type=int, default=0,
                        help="cap the TRAIN fold to this many graphs for search speed "
                             "(0 = full train fold; val is always full-size). "
                             "Architecture search does not need all 60k graphs.")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    load_canonical_split(args.canonical_split)
    os.environ["CANONICAL_SPLIT_JSON"] = args.canonical_split
    structure = resolve_structure(args.graph_structure, "protgnn")

    print(f"Device           : {DEVICE}")
    print(f"Dataset          : {args.dataset}")
    print(f"Graph structure  : {structure}")
    print(f"Canonical split  : {args.canonical_split}")
    print(f"Loss weighting   : {LOSS_WEIGHTING}")
    print(f"Trials           : {args.n_trials}  |  Max epochs/trial: {args.max_epochs}")

    print("\nBuilding dataset once (reused across trials)...")
    full = get_dataset(DATASET_DIR, args.dataset, graph_structure=structure,
                       canonical_split=args.canonical_split)
    input_dim = full.num_node_features
    output_dim = int(full.num_classes)
    print(f"  graphs={len(full)}  classes={output_dim}  feat_dim={input_dim}")

    loaders = get_dataloader(full, batch_size=128, canonical_split=args.canonical_split)
    train_dataset, val_loader = loaders['train'].dataset, loaders['eval']
    print(f"  train (full)     : {len(train_dataset)}  |  val: {len(val_loader.dataset)}")

    if args.subsample and args.subsample < len(train_dataset):
        from torch.utils.data import Subset
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(train_dataset), size=args.subsample, replace=False).tolist()
        train_dataset = Subset(train_dataset, idx)
        print(f"  train (subsample): {args.subsample} graphs (search speed; val untouched)")
    print()

    pruner = MedianPruner(n_startup_trials=8, n_warmup_steps=12)
    study = optuna.create_study(direction='maximize', pruner=pruner,
                                study_name=f'mimic_disease_hpo_{structure}')
    study.optimize(make_objective(train_dataset, val_loader, input_dim, output_dim, args.max_epochs),
                   n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print("\n" + "=" * 55)
    print("BEST TRIAL (validation macro-F1)")
    print("=" * 55)
    print(f"  macro_F1 : {best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k:<20}: {v}")

    out = os.path.join(RESULTS_DIR, f'best_params_disease_{structure}.json')
    with open(out, 'w') as f:
        json.dump({
            'macro_f1': best.value, 'params': best.params,
            'structure': structure, 'loss_weighting': LOSS_WEIGHTING,
            'canonical_split': args.canonical_split,
            'n_trials': args.n_trials, 'max_epochs': args.max_epochs,
            'subsample': args.subsample, 'seed': args.seed,
        }, f, indent=2)
    with open(os.path.join(RESULTS_DIR, f'study_disease_{structure}.pkl'), 'wb') as f:
        pickle.dump(study, f)
    print(f"\nSaved to {out}")

    print("\nTop 5 trials:")
    trials = sorted([t for t in study.trials if t.value is not None],
                    key=lambda t: t.value, reverse=True)
    for t in trials[:5]:
        print(f"  Trial {t.number:3d}  macroF1={t.value:.4f}  {t.params}")


if __name__ == '__main__':
    main()
