"""
Hyperparameter optimisation for GraphCare (Method B).
======================================================
No prior GraphCare HPO existed anywhere in this repo — this is new, not a
port of an old script. Built to the same standard as the other two:
gsat_analysis/scripts/hpo_gsat.py and protgnn_analysis/scripts/hpo_disease.py.

  - canonical split (comparison/canonical_split.json), same train/val/test
    patients as the 3-seed matrix and the other two methods' HPO.
  - sqrt_inverse class weighting (this comparison's locked-in policy), via
    the exact same shared formula (see _train_fold_class_weights, reused
    verbatim from graphcare_analysis/run.py — not reimplemented here).
  - Optuna + MedianPruner, optimises VALIDATION macro-F1 only. Test is never
    touched; a single final test evaluation happens later, once, after a
    winning config is retrained at full scale.
  - train-fold SUBSAMPLE for search speed (val is untouched and full-size).

Must run inside .venv-graphcare (GraphCare's own BAT-GNN + pinned older
torch). Optuna was not previously installed there — added as a dependency;
it's a pure orchestration library with no torch/PyG version coupling, so it
does not disturb the pinned GraphCare environment.

Run (inside the GraphCare venv):
    PYTHONPATH=.:external/GraphCare .venv-graphcare/bin/python3 \
        -m graphcare_analysis.scripts.hpo_graphcare \
        --graph_structure star --n_trials 30 --max_epochs 25 --subsample 20000

Results:
    graphcare_analysis/outputs/hpo_results/best_params_<structure>.json
    graphcare_analysis/outputs/hpo_results/study_<structure>.pkl
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from optuna.pruners import MedianPruner
from torch.utils.data import Subset

from shared.lib.benchmark_contract import load_canonical_split
from shared.lib.config_base import set_seed
from shared.lib.graph_structures import resolve as resolve_structure, structure_dir
from graphcare_analysis.config import cfg
from graphcare_analysis.adapter import build_loaders, _collate
from graphcare_analysis.run import (
    _load_kg, _move, _forward, _evaluate, _train_fold_class_weights, build_graphcare_model,
)
from torch.utils.data import DataLoader

RESULTS_DIR = Path(cfg.OUTPUTS_DIR) / "hpo_results"


def make_objective(train_dataset, val_loader, kg, num_classes, device, max_epochs):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        emb_dim = trial.suggest_categorical("emb_dim", [64, 96, 128, 192])
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=_collate)

        # build_graphcare_model reads dims from cfg — mutate for this trial only.
        previous = (cfg.emb_dim, cfg.num_layers, cfg.dropout)
        cfg.emb_dim, cfg.num_layers, cfg.dropout = emb_dim, num_layers, dropout
        try:
            model = build_graphcare_model(kg, num_classes, device)
        finally:
            cfg.emb_dim, cfg.num_layers, cfg.dropout = previous
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        class_weights = _train_fold_class_weights(train_loader, num_classes, "sqrt_inverse")
        weight_t = None if class_weights is None else torch.tensor(
            class_weights, dtype=torch.float32, device=device)

        best_f1, no_improve, PATIENCE = -1.0, 0, 5
        for epoch in range(max_epochs):
            model.train()
            for b in train_loader:
                b = _move(b, device)
                loss = F.cross_entropy(_forward(model, b), b["y"], weight=weight_t)
                opt.zero_grad()
                loss.backward()
                opt.step()

            val = _evaluate(model, val_loader, device)
            trial.report(val["macro_f1"], epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            if val["macro_f1"] > best_f1:
                best_f1, no_improve = val["macro_f1"], 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    break
        return best_f1

    return objective


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph_structure", "--graph", default="star")
    ap.add_argument("--canonical_split", default="comparison/canonical_split.json")
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--max_epochs", type=int, default=25, help="max epochs per trial")
    ap.add_argument("--subsample", type=int, default=20000,
                    help="cap the TRAIN fold to this many graphs for search speed "
                         "(0 = full train fold; val is always full-size)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    # torch 1.12 MPS is unreliable for GraphCare — CPU, same as run.py.
    device = torch.device("cpu" if cfg.device == "mps" else cfg.device)
    load_canonical_split(args.canonical_split)
    structure = resolve_structure(args.graph_structure, "graphcare")
    print(f"  graph structure  : {structure}")
    print(f"  device           : {device}")

    kg_path = structure_dir(cfg.data_dir, structure, "graphcare") / "kg.pt"
    kg = _load_kg(kg_path, split_json=args.canonical_split)
    train_loader, val_loader, _test_loader, num_classes, _ = build_loaders(
        kg, split_json=args.canonical_split, structure=structure)
    train_dataset = train_loader.dataset
    print(f"  train (full)     : {len(train_dataset)}  |  val: {len(val_loader.dataset)}")

    if args.subsample and args.subsample < len(train_dataset):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(train_dataset), size=args.subsample, replace=False).tolist()
        train_dataset = Subset(train_dataset, idx)
        print(f"  train (subsample): {len(train_dataset)}  (search speed; val untouched)")

    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=3))
    objective = make_objective(train_dataset, val_loader, kg, num_classes, device, args.max_epochs)

    t0 = time.time()
    study.optimize(objective, n_trials=args.n_trials)
    elapsed = time.time() - t0
    print(f"\n  search done in {elapsed:.0f}s  ({elapsed / max(args.n_trials, 1):.0f}s/trial avg)")

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  trials: {len(completed)} completed, {len(pruned)} pruned")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best = {
        "macro_f1": study.best_value,
        "params": study.best_trial.params,
        "structure": structure,
        "loss_weighting": "sqrt_inverse",
        "canonical_split": str(args.canonical_split),
        "n_trials": args.n_trials,
        "max_epochs": args.max_epochs,
        "subsample": args.subsample,
        "seed": args.seed,
        "elapsed_s": round(elapsed, 1),
    }
    (RESULTS_DIR / f"best_params_{structure}.json").write_text(json.dumps(best, indent=2))
    with open(RESULTS_DIR / f"study_{structure}.pkl", "wb") as f:
        pickle.dump(study, f)
    print("\n  BEST:")
    print(json.dumps(best, indent=2))
    print(f"\n  wrote {RESULTS_DIR / f'best_params_{structure}.json'}")


if __name__ == "__main__":
    main()
