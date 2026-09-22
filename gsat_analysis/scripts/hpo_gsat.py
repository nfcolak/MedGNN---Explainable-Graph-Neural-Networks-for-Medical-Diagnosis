"""
Hyperparameter optimisation for GSAT (Method C).
=================================================
Ports the pattern from protgnn_analysis/scripts/hpo_disease.py (Optuna +
MedianPruner) to GSAT, fixed to this project's CURRENT comparison protocol —
two things that script predates and does NOT use, which this one does:

  - the canonical split (comparison/canonical_split.json), not an ad-hoc
    random split, so the search is consistent with the 3-seed matrix / XGBoost
    baseline already run;
  - sqrt_inverse class weighting, the policy locked in for this comparison
    (see gsat_analysis/train.py --loss_weighting), not unweighted or
    ProtGNN's old default 'inverse'.

Optimises VALIDATION macro-F1 only. Test is never touched here — same
discipline as every other run in this comparison; a single final test
evaluation happens later, once, after a winning config is chosen and retrained
at full scale (see gsat_analysis/train.py).

Speed: each trial trains on a random TRAIN-fold SUBSAMPLE (val is untouched
and always full-size, so the reported macro-F1 is a fair signal). Architecture
/ LR search doesn't need the full 59.6k train graphs — same precedent already
accepted in this repo (protgnn_analysis/scripts/hpo_disease.py's --subsample).
MedianPruner kills clearly-losing trials early using per-epoch reports.

Fixed, NOT searched (kept identical to the rest of the comparison so the
result stays comparable): --graph_structure/topology, node-level attention
(the project's explanation contract depends on it), the r-schedule
(init_r/final_r/decay_interval — the GSAT paper reports r in [0.5, 0.9] is
robust; not worth the extra search dimensions on this budget).

Run:
    PYTHONPATH=. .venv-protgnn/bin/python3 -m gsat_analysis.scripts.hpo_gsat \
        --graph_structure star --n_trials 30 --max_epochs 25 --subsample 20000

Results:
    gsat_analysis/outputs/hpo_results/best_params_<structure>.json
    gsat_analysis/outputs/hpo_results/study_<structure>.pkl
"""
import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from shared.lib.benchmark_contract import load_canonical_split
from shared.lib.config_base import set_seed
from shared.lib.graph_structures import resolve as resolve_structure
from protgnn_analysis.load_dataset import get_dataset, get_dataloader
from gsat_analysis.config import cfg
from gsat_analysis.models import GIN, GSAT, ExtractorMLP
from gsat_analysis.train import evaluate, _train_fold_class_weights

RESULTS_DIR = Path(cfg.OUTPUTS_DIR) / "hpo_results"


def make_objective(train_dataset, val_loader, x_dim, num_classes, device, max_epochs):
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [128, 256])
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128, 192])
        num_layers = trial.suggest_int("num_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        readout = trial.suggest_categorical("readout", ["mean", "sum"])  # gin.py only implements these two
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        info_loss_coef = trial.suggest_float("info_loss_coef", 0.5, 2.0)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        clf = GIN(x_dim, num_classes, hidden_dim=hidden_dim, num_layers=num_layers,
                  dropout=dropout, readout=readout)
        extractor = ExtractorMLP(hidden_dim, attention_level=cfg.attention_level)
        class_weights = _train_fold_class_weights(train_loader, num_classes, "sqrt_inverse")
        gsat = GSAT(clf, extractor, attention_level=cfg.attention_level,
                    temperature=cfg.temperature, info_loss_coef=info_loss_coef,
                    init_r=cfg.init_r, final_r=cfg.final_r,
                    decay_interval=cfg.decay_interval, decay_r=cfg.decay_r,
                    class_weights=class_weights).to(device)
        opt = torch.optim.Adam(gsat.parameters(), lr=lr, weight_decay=weight_decay)

        best_f1, no_improve, PATIENCE = -1.0, 0, 5
        for epoch in range(max_epochs):
            gsat.train()
            for data in train_loader:
                data = data.to(device)
                out = gsat(data, epoch=epoch, training=True)
                opt.zero_grad()
                out["loss"].backward()
                opt.step()

            val = evaluate(gsat, val_loader, device, epoch)
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
                         "(0 = full 59.6k train fold; val is always full-size)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(cfg.device)
    load_canonical_split(args.canonical_split)
    os.environ["CANONICAL_SPLIT_JSON"] = args.canonical_split
    structure = resolve_structure(args.graph_structure, "gsat")
    print(f"  graph structure  : {structure}")
    print(f"  device           : {device}")

    dataset = get_dataset(str(cfg.data_dir), cfg.dataset_name, graph_structure=structure,
                          canonical_split=args.canonical_split)
    loaders = get_dataloader(dataset, batch_size=cfg.batch_size,
                             data_split_ratio=list(cfg.split_ratio), seed=args.seed,
                             canonical_split=args.canonical_split)
    train_dataset, val_loader = loaders["train"].dataset, loaders["eval"]
    print(f"  train (full)     : {len(train_dataset)}  |  val: {len(val_loader.dataset)}")

    if args.subsample and args.subsample < len(train_dataset):
        rng = random.Random(args.seed)
        idx = list(range(len(train_dataset)))
        rng.shuffle(idx)
        train_dataset = Subset(train_dataset, idx[: args.subsample])
        print(f"  train (subsample): {len(train_dataset)}  (search speed; val untouched)")

    x_dim = dataset[0].x.size(1)

    study = optuna.create_study(direction="maximize", pruner=MedianPruner(n_warmup_steps=3))
    objective = make_objective(train_dataset, val_loader, x_dim, cfg.num_classes, device, args.max_epochs)

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
