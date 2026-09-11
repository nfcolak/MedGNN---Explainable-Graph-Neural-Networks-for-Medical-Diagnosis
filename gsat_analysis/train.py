"""Train + evaluate GSAT on mimic_intra_patient_disease  —  Method C.

Parallel to protgnn_analysis/train.py and graphcare_analysis/run.py: same csv,
same subject-aware canonical split, same seed, same shared metrics, written to
outputs/<structure>/report.txt in the same style. GSAT reuses ProtGNN's
per-patient graph cache verbatim (data/graphs/<structure>/protgnn/).

Run (inside the ProtGNN venv — GSAT needs torch 2.x / PyG 2.6):
    PYTHONPATH=. .venv-protgnn/bin/python3 gsat_analysis/train.py \
        --graph_structure star --max_epochs 100
    # identical-test-patient comparison with the other two methods:
    CANONICAL_SPLIT_JSON=comparison/canonical_split.json \
        PYTHONPATH=. .venv-protgnn/bin/python3 gsat_analysis/train.py --graph_structure star

Explanations are generated separately by
gsat_analysis/explainability/explain_gsat.py (mirrors explain_graphcare.py).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from shared.lib.config_base import set_seed
from shared.lib.graph_structures import prompt_for_structure
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
    m["accuracy"] = round(float((y == pred).mean()), 6)
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
    (out_dir / "metrics.json").write_text(json.dumps({"test": test}, indent=2))
    print("  wrote", out_dir / "report.txt")


def main(graph_structure=None, max_epochs=None, limit=None, out_dir=None):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    max_epochs = max_epochs or cfg.max_epochs
    structure = resolve_structure(graph_structure or cfg.graph_structure, "gsat")
    out_dir = out_dir or (cfg.OUTPUTS_DIR / structure)
    print(f"  graph structure  : {structure}")
    print(f"  device           : {device}")

    dataset = get_dataset(str(cfg.data_dir), cfg.dataset_name, graph_structure=structure)
    if limit is not None:
        dataset = dataset[:limit]
    loaders = get_dataloader(dataset, batch_size=cfg.batch_size,
                             data_split_ratio=list(cfg.split_ratio), seed=cfg.seed)
    train_loader, val_loader, test_loader = loaders["train"], loaders["eval"], loaders["test"]
    n_train = len(train_loader.dataset)
    n_val, n_test = len(val_loader.dataset), len(test_loader.dataset)

    x_dim = dataset[0].x.size(1)
    gsat = build_model(x_dim, cfg.num_classes, device)
    opt = torch.optim.Adam(gsat.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(f"  params           : {sum(p.numel() for p in gsat.parameters()):,}")

    best_f1, best_state, bad, t0, last_epoch = -1.0, None, 0, time.time(), 0
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

        # Only start tracking "best" once the r curriculum has settled (paper's
        # update_best_epoch_res gates on current_r == final_r).
        if r <= cfg.final_r and val["macro_f1"] > best_f1 + cfg.min_delta:
            best_f1 = val["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in gsat.state_dict().items()}
            bad = 0
        elif r <= cfg.final_r:
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
    print(f"  TEST macro_f1 {test['macro_f1']:.4f} | micro_f1 {test['micro_f1']:.4f} "
          f"| acc {test['accuracy']:.4f}")
    _write_report(test, structure, n_train, n_val, n_test, last_epoch, out_dir)
    return test


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train GSAT (Method C)")
    ap.add_argument("--graph_structure", "--graph", default=None,
                    help="star|cooccur|ontology|full (prompted if omitted)")
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap #graphs (quick runs)")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    structure = args.graph_structure or prompt_for_structure("gsat",
                                                             default=cfg.graph_structure)
    main(graph_structure=structure, max_epochs=args.max_epochs, limit=args.limit,
         out_dir=Path(args.out_dir) if args.out_dir else None)
