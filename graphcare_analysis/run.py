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
import sys
import time

import torch
import torch.nn.functional as F

from shared.lib.config_base import set_seed
from shared.lib.metrics import multiclass_metrics
from graphcare_analysis.config import cfg
from graphcare_analysis.build_kg import build_global_kg
from graphcare_analysis.adapter import build_loaders


def _load_kg():
    return torch.load(cfg.kg_path) if cfg.kg_path.exists() else build_global_kg(save=True)


def _move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model, b):
    return model(b["node_ids"], b["rel_ids"], b["edge_index"], b["batch"],
                 b["visit_node"], b["ehr_nodes"])


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
    m["accuracy"] = round(float((y == pred).mean()), 6)
    m["loss"] = round(tot / max(n, 1), 6)
    return m


def _write_report(test, C, kg, limit, epochs, out_dir=None):
    out_dir = out_dir or cfg.OUTPUTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "GRAPHCARE (Method B) -- model-direct on merged_ed.csv",
        f"  dataset          : merged_ed.csv (limit={limit})",
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
    (out_dir / "metrics.json").write_text(json.dumps({"test": test}, indent=2))
    print("  wrote", out_dir / "report.txt")


def main(limit=None, max_epochs=None, patience=10, min_delta=0.005,
         split_json=None, out_dir=None):
    set_seed(cfg.seed)
    # torch 1.12 MPS is unreliable -> CPU for GraphCare
    device = torch.device("cpu" if cfg.device == "mps" else cfg.device)
    max_epochs = max_epochs or cfg.max_epochs

    sys.path.insert(0, str(cfg.UPSTREAM_DIR))
    if not (cfg.UPSTREAM_DIR / "graphcare_" / "model.py").exists():
        raise FileNotFoundError("Clone upstream GraphCare first (external/GraphCare/README.md).")
    from graphcare_.model import GraphCare

    kg = _load_kg()
    train_loader, val_loader, test_loader, C, _ = build_loaders(kg, limit=limit, split_json=split_json)

    model = GraphCare(num_nodes=kg["num_nodes"], num_rels=kg["num_rels"], max_visit=1,
                      embedding_dim=cfg.emb_dim, hidden_dim=cfg.emb_dim, out_channels=C,
                      layers=cfg.num_layers, dropout=cfg.dropout,
                      patient_mode="joint", use_alpha=True, use_beta=True, gnn="BAT").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1, best_state, bad, t0 = -1.0, None, 0, time.time()
    last_epoch = 0
    for epoch in range(max_epochs):
        last_epoch = epoch + 1
        model.train()
        for b in train_loader:
            b = _move(b, device)
            loss = F.cross_entropy(_forward(model, b), b["y"])
            opt.zero_grad(); loss.backward(); opt.step()
        val = _evaluate(model, val_loader, device)
        print(f"  epoch {epoch:3d} | val macro_f1 {val['macro_f1']:.4f} | "
              f"val acc {val['accuracy']:.4f} | {time.time()-t0:.0f}s", flush=True)
        if val["macro_f1"] > best_f1 + min_delta:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop @ epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = _evaluate(model, test_loader, device)
    print(f"  TEST macro_f1 {test['macro_f1']:.4f} | micro_f1 {test['micro_f1']:.4f} | acc {test['accuracy']:.4f}")
    _write_report(test, C, kg, limit, last_epoch, out_dir=out_dir)
    return test


if __name__ == "__main__":
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap #patients (quick runs)")
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--canonical_split", default=None, help="comparison/canonical_split.json")
    ap.add_argument("--out_dir", default=None, help="where to write report.txt")
    args = ap.parse_args()
    main(limit=args.limit, max_epochs=args.max_epochs, patience=args.patience,
         split_json=args.canonical_split, out_dir=Path(args.out_dir) if args.out_dir else None)
