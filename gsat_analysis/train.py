"""Train + evaluate GSAT on mimic_intra_patient_disease  —  Method C.

Parallel to protgnn_analysis/train.py and graphcare_analysis/run.py: same csv,
same subject-aware canonical split, same seed, and same shared metrics. GSAT
uses ProtGNN's split-aware per-patient graph representation; standardized cache
identity and metadata bind the source dataset, split, recipe, and fit subjects.

Run (inside the ProtGNN venv — GSAT needs torch 2.x / PyG 2.6):
    PYTHONPATH=. python3 gsat_analysis/train.py \
        --graph_structure star --max_epochs 100 \
        --out_dir comparison/standardized/results/gsat/star/1234 \
        --canonical_split comparison/canonical_split.json --seed 1234

Explanations are generated separately by
gsat_analysis/explainability/explain_gsat.py (mirrors explain_graphcare.py).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from shared.lib.benchmark_contract import BenchmarkSpec, load_canonical_split
from shared.lib.config_base import isolated_callable, set_seed
from shared.lib.graph_structures import resolve as resolve_structure
from shared.lib.metrics import multiclass_metrics
from protgnn_analysis.load_dataset import get_dataset, get_dataloader
from gsat_analysis.config import cfg
from gsat_analysis.models import GIN, GSAT, ExtractorMLP


def build_model(x_dim, num_classes, device):
    clf = GIN(x_dim, num_classes, hidden_dim=cfg.hidden_dim,
              num_layers=cfg.num_layers, dropout=cfg.dropout, readout=cfg.readout)
    extractor = ExtractorMLP(cfg.hidden_dim, attention_level=cfg.attention_level)
    gsat = GSAT(clf, extractor, attention_level=cfg.attention_level,
                temperature=cfg.temperature, info_loss_coef=cfg.info_loss_coef,
                init_r=cfg.init_r, final_r=cfg.final_r,
                decay_interval=cfg.decay_interval, decay_r=cfg.decay_r)
    return gsat.to(device)


@torch.no_grad()
def evaluate(gsat, loader, device, epoch):
    gsat.eval()
    ys, preds, probs, tot, n = [], [], [], 0.0, 0
    for data in loader:
        data = data.to(device)
        out = gsat(data, epoch=epoch, training=False)
        logits = out["logits"]
        y = data.y.view(-1).long()
        tot += F.cross_entropy(logits, y).item() * y.size(0)
        n += y.size(0)
        ys.append(y.cpu())
        preds.append(logits.argmax(1).cpu())
        probs.append(torch.softmax(logits, 1).cpu())
    y = torch.cat(ys).numpy()
    pred = torch.cat(preds).numpy()
    prob = torch.cat(probs).numpy()
    m = multiclass_metrics(y, pred, prob)
    m["loss"] = round(tot / max(n, 1), 6)
    return m


def _write_report(test, structure, n_train, n_val, n_test, epochs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "GSAT (Method C) -- stochastic-attention GIN on mimic_intra_patient_disease",
        f"  dataset          : {cfg.dataset_name}  (graphs reused from protgnn cache)",
        f"  graph structure  : {structure}",
        f"  split            : train={n_train} val={n_val} test={n_test}",
        f"  backbone         : GIN dim={cfg.hidden_dim} x {cfg.num_layers} layers, "
        f"readout={cfg.readout}, dropout={cfg.dropout}",
        f"  attention        : {cfg.attention_level}-level, temp={cfg.temperature}, "
        f"info_coef={cfg.info_loss_coef}, r {cfg.init_r}->{cfg.final_r}",
        f"  classes          : {cfg.num_classes}",
        f"  epochs run       : {epochs}",
        "",
        "TEST SET PERFORMANCE",
    ]
    for k in ("accuracy", "balanced_acc", "macro_f1", "micro_f1", "top3_acc", "top5_acc"):
        if k in test:
            lines.append(f"  {k:16s}: {test[k]:.6f}")
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps({
        "parameter_count": test["parameter_count"],
        "test": test,
    }, indent=2))
    print("  wrote", out_dir / "report.txt")


def _validate_optional_positive_int(name, value):
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{name} must be a positive exact integer when provided.")


@isolated_callable(cfg)
def main(graph_structure, max_epochs, limit, out_dir, canonical_split, seed):
    """Run GSAT under the standardized benchmark contract.

    RNGs are seeded here. Cross-backend bitwise determinism is intentionally
    recorded by the run-manifest policy rather than forced: CUDA/MPS supported
    deterministic operations differ across installed torch/PyG versions.
    """
    if canonical_split is None:
        raise ValueError("Standardized GSAT runs require a canonical split.")
    if out_dir is None:
        raise ValueError("Standardized GSAT runs require an output directory.")
    _validate_optional_positive_int("max_epochs", max_epochs)
    _validate_optional_positive_int("limit", limit)
    spec = BenchmarkSpec("gsat", graph_structure, seed)
    load_canonical_split(canonical_split)
    set_seed(spec.seed)

    env_key = "CANONICAL_SPLIT_JSON"
    previous_split = os.environ.get(env_key)
    os.environ[env_key] = str(canonical_split)
    try:
        return _run_training(
            spec, max_epochs, limit, Path(out_dir), Path(canonical_split)
        )
    finally:
        if previous_split is None:
            os.environ.pop(env_key, None)
        else:
            os.environ[env_key] = previous_split


def _limit_fold_loaders(loaders, limit):
    """Cap each already-constructed canonical fold without reassigning graphs."""
    if limit is None:
        return loaders
    return {
        name: DataLoader(
            Subset(loader.dataset, range(min(limit, len(loader.dataset)))),
            batch_size=loader.batch_size,
            shuffle=name == "train",
        )
        for name, loader in loaders.items()
    }


def _run_training(spec, max_epochs, limit, out_dir, canonical_split):
    device = torch.device(cfg.device)
    max_epochs = cfg.max_epochs if max_epochs is None else max_epochs
    structure = resolve_structure(spec.structure, "gsat")
    print(f"  graph structure  : {structure}")
    print(f"  device           : {device}")

    dataset = get_dataset(
        str(cfg.data_dir), cfg.dataset_name, graph_structure=structure,
        canonical_split=canonical_split,
    )
    loaders = get_dataloader(dataset, batch_size=cfg.batch_size,
                             data_split_ratio=list(cfg.split_ratio), seed=spec.seed,
                             canonical_split=canonical_split)
    loaders = _limit_fold_loaders(loaders, limit)
    train_loader, val_loader, test_loader = loaders["train"], loaders["eval"], loaders["test"]
    n_train = len(train_loader.dataset)
    n_val, n_test = len(val_loader.dataset), len(test_loader.dataset)

    x_dim = dataset[0].x.size(1)
    gsat = build_model(x_dim, cfg.num_classes, device)
    opt = torch.optim.Adam(gsat.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    parameter_count = sum(p.numel() for p in gsat.parameters())
    print(f"  params           : {parameter_count:,}")

    best_f1 = -1.0
    best_for_patience = -1.0
    best_state, bad, t0, last_epoch = None, 0, time.time(), 0
    for epoch in range(max_epochs):
        last_epoch = epoch + 1
        gsat.train()
        ep_loss = {"pred": 0.0, "info": 0.0}
        for data in train_loader:
            data = data.to(device)
            out = gsat(data, epoch=epoch, training=True)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            ep_loss["pred"] += out["pred_loss"].item()
            ep_loss["info"] += out["info_loss"].item()
        val = evaluate(gsat, val_loader, device, epoch)
        r = gsat.get_r(epoch)
        print(f"  epoch {epoch:3d} | r {r:.2f} | pred {ep_loss['pred']/len(train_loader):.3f} "
              f"info {ep_loss['info']/len(train_loader):.3f} | val macro_f1 {val['macro_f1']:.4f} "
              f"acc {val['accuracy']:.4f} | {time.time()-t0:.0f}s", flush=True)

        # The benchmark selects the exact highest validation macro-F1 checkpoint.
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in gsat.state_dict().items()
            }

        # ``min_delta`` controls patience only; it must not change checkpoint choice.
        if val["macro_f1"] > best_for_patience + cfg.min_delta:
            best_for_patience = val["macro_f1"]
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"  early stop @ epoch {epoch}")
                break

    if best_state is not None:
        gsat.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "gsat_model.pt"
    torch.save({"state_dict": gsat.state_dict(), "x_dim": x_dim,
                "num_classes": cfg.num_classes, "structure": structure,
                "config": {k: getattr(cfg, k) for k in (
                    "hidden_dim", "num_layers", "dropout", "readout",
                    "attention_level", "temperature", "info_loss_coef",
                    "init_r", "final_r", "decay_interval", "decay_r")}},
               ckpt)
    print("  saved", ckpt)

    test = evaluate(gsat, test_loader, device, last_epoch)
    test["parameter_count"] = parameter_count
    print(f"  TEST macro_f1 {test['macro_f1']:.4f} | micro_f1 {test['micro_f1']:.4f} "
          f"| acc {test['accuracy']:.4f}")
    _write_report(test, structure, n_train, n_val, n_test, last_epoch, out_dir)
    return test


def cli(argv=None):
    """Parse the non-interactive standardized benchmark command line."""
    parser = argparse.ArgumentParser(description="Train GSAT (Method C)")
    parser.add_argument(
        "--graph_structure",
        "--graph",
        required=True,
        help="star|cooccur|ontology|full",
    )
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap #graphs (quick runs)")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--canonical_split", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    return main(
        args.graph_structure,
        args.max_epochs,
        args.limit,
        args.out_dir,
        args.canonical_split,
        args.seed,
    )


if __name__ == "__main__":
    cli()
