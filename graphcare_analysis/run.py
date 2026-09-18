"""Train + evaluate GraphCare on merged_ed.csv  —  Phase 3 (model-direct).

Path B (validated in test_model_smoke.py + adapter.py): we drive the upstream
`GraphCare` BAT-GNN with tensors we build from our KG, bypassing GraphCare's
hardwired data_prepare.py. Evaluation uses the SHARED metrics so the result is
directly comparable with protgnn_analysis (same csv / same subject-aware split /
same seed / same metrics), written to outputs/report.txt in the same style.

Run (inside the GraphCare venv):
    PYTHONPATH=.:external/GraphCare .venv-graphcare/bin/python3 \
        graphcare_analysis/run.py --limit 2000 --max_epochs 15
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from shared.lib.benchmark_contract import BenchmarkSpec, load_canonical_split
from shared.lib.config_base import isolated_callable, set_seed
from shared.lib.graph_structures import resolve as resolve_structure
from shared.lib.graph_structures import prompt_for_structure, structure_dir
from shared.lib.metrics import multiclass_metrics
from graphcare_analysis.config import cfg
from graphcare_analysis.build_kg import (
    build_global_kg,
    validate_kg_schema,
    validate_training_provenance,
)
from graphcare_analysis.adapter import build_loaders
# Reused verbatim (pure numpy, no torch/PyG dependency — safe in this venv too)
# so GraphCare, GSAT and ProtGNN compute class weights with the exact same
# formula for this comparison — see --loss_weighting below.
from comparison.standardized.performance_review import class_weights as _class_weight_policy

LOSS_WEIGHTING_CHOICES = ("none", "sqrt_inverse")


def _train_fold_class_weights(train_loader, num_classes, loss_weighting):
    if loss_weighting == "none":
        return None
    labels = np.array([train_loader.dataset[i]["y"] for i in range(len(train_loader.dataset))])
    return _class_weight_policy(labels, num_classes, loss_weighting)


def _load_kg(kg_path, split_json=None):
    if kg_path.exists():
        kg = torch.load(kg_path)
        validate_kg_schema(kg)
        if split_json:
            validate_training_provenance(kg, split_json)
        return kg
    return build_global_kg(
        save=True, save_path=kg_path, split_json=split_json
    )


def _move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model, b):
    return model(b["node_ids"], b["rel_ids"], b["edge_index"], b["batch"],
                 b["visit_node"], b["ehr_nodes"])


def build_graphcare_model(kg, num_classes, device):
    """Construct the exact GraphCare architecture used by train and explain."""
    sys.path.insert(0, str(cfg.UPSTREAM_DIR))
    if not (cfg.UPSTREAM_DIR / "graphcare_" / "model.py").exists():
        raise FileNotFoundError("Clone upstream GraphCare first (external/GraphCare/README.md).")
    from graphcare_analysis.model import GraphCare

    return GraphCare(
        num_nodes=kg["num_nodes"],
        num_rels=kg["num_rels"],
        max_visit=1,
        embedding_dim=cfg.emb_dim,
        hidden_dim=cfg.emb_dim,
        out_channels=num_classes,
        layers=cfg.num_layers,
        dropout=cfg.dropout,
        patient_mode="joint",
        use_alpha=True,
        use_beta=True,
        gnn="BAT",
    ).to(device)


@torch.no_grad()
def _evaluate(model, loader, device):
    model.eval()
    ys, preds, probs, tot, n = [], [], [], 0.0, 0
    for b in loader:
        b = _move(b, device)
        logits = _forward(model, b)
        tot += F.cross_entropy(logits, b["y"]).item() * b["y"].size(0)
        n += b["y"].size(0)
        ys.append(b["y"].cpu())
        preds.append(logits.argmax(1).cpu())
        probs.append(torch.softmax(logits, 1).cpu())
    y = torch.cat(ys).numpy()
    pred = torch.cat(preds).numpy()
    prob = torch.cat(probs).numpy()
    m = multiclass_metrics(y, pred, prob)
    m["loss"] = round(tot / max(n, 1), 6)
    return m


def _write_report(test, C, kg, limit, epochs, out_dir=None, loss_weighting="none"):
    out_dir = out_dir or cfg.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "GRAPHCARE (Method B) -- model-direct on merged_ed.csv",
        f"  dataset          : merged_ed.csv (limit={limit})",
        f"  loss_weighting   : {loss_weighting}",
        f"  KG               : {kg['num_nodes']} nodes, {kg['num_rels']} rels (ontology+PMI)",
        f"  model            : GraphCare BAT-GNN, dim={cfg.emb_dim}, layers={cfg.num_layers}",
        f"  classes          : {C}",
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


def _validate_standardized_output(out_dir, spec):
    expected_suffix = (
        "comparison", "standardized", "results", "graphcare",
        spec.structure, f"seed_{spec.seed}",
    )
    path = Path(out_dir)
    if tuple(path.parts[-len(expected_suffix):]) != expected_suffix:
        raise ValueError(
            "Standardized GraphCare output must be under standardized results at "
            f"comparison/standardized/results/graphcare/{spec.structure}/"
            f"seed_{spec.seed}."
        )
    return path


@isolated_callable(cfg)
def main(limit=None, max_epochs=None, patience=10, min_delta=0.005,
         split_json=None, out_dir=None, graph_structure=None, seed=None,
         loss_weighting="none"):
    """Run GraphCare in fail-closed standardized mode or explicit legacy mode.

    loss_weighting: "none" (default, original unweighted cross-entropy) or
    "sqrt_inverse" (train-fold-only class weighting, shared formula with
    GSAT/ProtGNN). Additive: omitting the flag reproduces every prior run.
    """
    _validate_optional_positive_int("limit", limit)
    _validate_optional_positive_int("max_epochs", max_epochs)
    _validate_optional_positive_int("patience", patience)
    if loss_weighting not in LOSS_WEIGHTING_CHOICES:
        raise ValueError(f"loss_weighting must be one of {LOSS_WEIGHTING_CHOICES}.")
    standardized = split_json is not None or seed is not None
    if standardized:
        required = {
            "canonical split": split_json,
            "output directory": out_dir,
            "graph structure": graph_structure,
            "seed": seed,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Standardized GraphCare mode requires all of canonical split, output "
                f"directory, graph structure, and seed; missing: {', '.join(missing)}."
            )
        spec = BenchmarkSpec("graphcare", graph_structure, seed)
        target_out = _validate_standardized_output(out_dir, spec)
        load_canonical_split(split_json)
        structure = spec.structure
        resolved_seed = spec.seed
        split_path = Path(split_json)
    else:
        structure = resolve_structure(graph_structure or cfg.graph_structure, "graphcare")
        resolved_seed = cfg.seed if seed is None else seed
        target_out = Path(out_dir) if out_dir is not None else (cfg.OUTPUTS_DIR / structure)
        split_path = Path(split_json) if split_json is not None else None

    previous = (cfg.seed, cfg.graph_structure)
    try:
        cfg.seed = resolved_seed
        cfg.graph_structure = structure
        set_seed(resolved_seed)
        return _run_training(
            limit=limit,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            split_json=split_path,
            out_dir=target_out,
            graph_structure=structure,
            seed=resolved_seed,
            loss_weighting=loss_weighting,
        )
    finally:
        cfg.seed, cfg.graph_structure = previous


def _run_training(*, limit, max_epochs, patience, min_delta, split_json,
                  out_dir, graph_structure, seed, loss_weighting="none"):
    # torch 1.12 MPS is unreliable -> CPU for GraphCare
    device = torch.device("cpu" if cfg.device == "mps" else cfg.device)
    max_epochs = cfg.max_epochs if max_epochs is None else max_epochs
    structure = resolve_structure(graph_structure, "graphcare")
    kg_path = structure_dir(cfg.data_dir, structure, "graphcare") / "kg.pt"
    print(f"  graph structure  : {structure}  (kg cache: {kg_path})")

    kg = _load_kg(kg_path, split_json=split_json)
    train_loader, val_loader, test_loader, C, _ = build_loaders(
        kg, limit=limit, split_json=split_json, structure=structure)

    model = build_graphcare_model(kg, C, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    class_weights = _train_fold_class_weights(train_loader, C, loss_weighting)
    weight_t = None if class_weights is None else torch.tensor(
        class_weights, dtype=torch.float32, device=device)
    if weight_t is not None:
        print(f"  loss_weighting   : {loss_weighting}  "
              f"(min={weight_t.min():.3f} max={weight_t.max():.3f})")

    best_f1 = -1.0
    best_for_patience = -1.0
    best_state, bad, t0 = None, 0, time.time()
    last_epoch = 0
    for epoch in range(max_epochs):
        last_epoch = epoch + 1
        model.train()
        for b in train_loader:
            b = _move(b, device)
            loss = F.cross_entropy(_forward(model, b), b["y"], weight=weight_t)
            opt.zero_grad(); loss.backward(); opt.step()
        val = _evaluate(model, val_loader, device)
        print(f"  epoch {epoch:3d} | val macro_f1 {val['macro_f1']:.4f} | "
              f"val acc {val['accuracy']:.4f} | {time.time()-t0:.0f}s", flush=True)
        # Checkpoint selection uses the exact highest validation macro-F1.
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # min_delta controls patience only and cannot suppress a better checkpoint.
        if val["macro_f1"] > best_for_patience + min_delta:
            best_for_patience = val["macro_f1"]
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop @ epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("GraphCare training produced no validation checkpoint.")
    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "graphcare_model.pt"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.save(
        {
            "state_dict": model.state_dict(),
            "structure": structure,
            "seed": seed,
            "num_classes": C,
            "parameter_count": parameter_count,
            "loss_weighting": loss_weighting,
        },
        checkpoint,
    )
    test = _evaluate(model, test_loader, device)
    test["parameter_count"] = parameter_count
    print(f"  TEST macro_f1 {test['macro_f1']:.4f} | micro_f1 {test['micro_f1']:.4f} | acc {test['accuracy']:.4f}")
    _write_report(test, C, kg, limit, last_epoch, out_dir=out_dir, loss_weighting=loss_weighting)
    return test


def cli(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap #patients (quick runs)")
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--canonical_split", type=Path, default=None,
                    help="comparison/canonical_split.json")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="isolated standardized run directory")
    ap.add_argument("--graph_structure", "--graph", default=None,
                    help="Graph topology: star|cooccur|ontology|full|"
                         "full_kg_expanded (prompted interactively if omitted)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--loss_weighting", choices=LOSS_WEIGHTING_CHOICES, default="none",
                    help="none (default, unweighted CE) | sqrt_inverse "
                         "(train-fold class weighting for imbalance)")
    args = ap.parse_args(argv)
    structure = args.graph_structure
    if structure is None and not any(
        value is not None for value in (args.canonical_split, args.out_dir, args.seed)
    ):
        structure = prompt_for_structure("graphcare", default=cfg.graph_structure)
    return main(
        limit=args.limit,
        max_epochs=args.max_epochs,
        patience=args.patience,
        split_json=args.canonical_split,
        out_dir=args.out_dir,
        graph_structure=structure,
        seed=args.seed,
        loss_weighting=args.loss_weighting,
    )


if __name__ == "__main__":
    cli()
