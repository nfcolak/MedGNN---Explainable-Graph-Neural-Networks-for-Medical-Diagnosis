"""
Evaluate a trained checkpoint on the test set WITHOUT retraining.

Loads outputs/checkpoints/<dataset>/<model>_<which>.pth, runs a forward pass
over the test split, and prints + saves full metrics (PR-AUC, ROC-AUC,
precision/recall/specificity/F1 at both argmax and a validation-calibrated
threshold) plus a threshold-sensitivity table.

Usage (from project root):
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/eval_checkpoint.py
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/eval_checkpoint.py --which best
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/eval_checkpoint.py --dataset mimic_intra_patient

Writes: outputs/results/eval_checkpoint_metrics.json
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn as nn

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, os.pardir))
for p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "external", "GraphXAI-main")):
    if p not in sys.path:
        sys.path.insert(0, p)

from protgnn_analysis.config import data_args, model_args, train_args, OUTPUTS_DIR
from protgnn_analysis.models import GnnNets
from protgnn_analysis.load_dataset import get_dataset, get_dataloader


def average_precision(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    p = tp / np.maximum(tp + fp, 1); r = tp / y.sum()
    p = np.concatenate(([1.0], p)); r = np.concatenate(([0.0], r))
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def roc_auc(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float)
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / max(y.sum(), 1); fpr = fp / max((1 - y).sum(), 1)
    tpr = np.concatenate(([0.], tpr, [1.])); fpr = np.concatenate(([0.], fpr, [1.]))
    return float(np.trapz(tpr, fpr))


def binmet(y, p, t):
    y = np.asarray(y, int); pr = (np.asarray(p, float) >= t).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum())
    fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return dict(threshold=round(float(t), 4), precision=round(prec, 4), recall=round(rec, 4),
                specificity=round(spec, 4), f1=round(f1, 4),
                balanced_acc=round((rec + spec) / 2, 4),
                acc=round((tp + tn) / max(len(y), 1), 4),
                tp=tp, tn=tn, fp=fp, fn=fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="override dataset name")
    ap.add_argument("--which", default="latest", choices=["latest", "best"],
                    help="which checkpoint (prototype mode uses 'latest' for fresh prototypes)")
    args = ap.parse_args()
    if args.dataset:
        data_args.dataset_name = args.dataset

    ds = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    dl = get_dataloader(ds, train_args.batch_size,
                        random_split_flag=data_args.random_split,
                        data_split_ratio=data_args.data_split_ratio, seed=data_args.seed)
    gnn = GnnNets(ds.num_node_features, ds.num_classes, model_args); gnn.to_device()

    ckpt_path = os.path.join(model_args.checkpoint, data_args.dataset_name,
                             f"{model_args.model_name}_{args.which}.pth")
    if not os.path.isfile(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=model_args.device)
    gnn.update_state_dict(ckpt["net"]); gnn.eval()
    print(f"Loaded {args.which} checkpoint: epoch={ckpt.get('epoch')} "
          f"val_metric={ckpt.get('acc'):.4f}  ({ckpt_path})")

    def predict(loader):
        ys, ps = [], []
        with torch.no_grad():
            for b in loader:
                _, probs, _, _, _ = gnn(b)
                ys.append(b.y.cpu()); ps.append(probs.cpu())
        return torch.cat(ys).numpy(), torch.cat(ps).numpy()

    # calibrate threshold on eval (max balanced accuracy)
    ev_y, ev_p = predict(dl["eval"])
    best_t, best_b = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        m = binmet(ev_y, ev_p[:, 1], t)
        if m["balanced_acc"] > best_b:
            best_b, best_t = m["balanced_acc"], float(t)

    te_y, te_p = predict(dl["test"])
    pr = average_precision(te_y, te_p[:, 1]); rc = roc_auc(te_y, te_p[:, 1])

    print(f"\n=== TEST SET (n={len(te_y)}, ADMITTED rate={te_y.mean():.4f}) ===")
    print(f"  PR-AUC  : {pr:.4f}")
    print(f"  ROC-AUC : {rc:.4f}")
    m05 = binmet(te_y, te_p[:, 1], 0.5)
    mc = binmet(te_y, te_p[:, 1], best_t)
    print(f"\n  argmax (t=0.50):")
    for k in ("precision", "recall", "specificity", "f1", "balanced_acc", "acc"):
        print(f"    {k:<13}: {m05[k]:.4f}")
    print(f"  calibrated (t={best_t:.3f}, eval bal-acc={best_b:.4f}):")
    for k in ("precision", "recall", "specificity", "f1", "balanced_acc", "acc"):
        print(f"    {k:<13}: {mc[k]:.4f}")

    print("\n  THRESHOLD SENSITIVITY (test):")
    table = []
    for t in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, round(best_t, 3)]:
        m = binmet(te_y, te_p[:, 1], t)
        table.append(m)
        print(f"    t={m['threshold']:.3f} | P={m['precision']:.3f} R={m['recall']:.3f} "
              f"Spec={m['specificity']:.3f} F1={m['f1']:.3f} BalAcc={m['balanced_acc']:.3f} | "
              f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")

    out = {
        "dataset": data_args.dataset_name,
        "checkpoint": ckpt_path,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_metric": ckpt.get("acc"),
        "test_n": int(len(te_y)),
        "test_admitted_rate": round(float(te_y.mean()), 4),
        "pr_auc": round(pr, 4),
        "roc_auc": round(rc, 4),
        "calibrated_threshold": round(best_t, 4),
        "metrics_argmax": m05,
        "metrics_calibrated": mc,
        "threshold_sensitivity": table,
    }
    os.makedirs(os.path.join(str(OUTPUTS_DIR), "results"), exist_ok=True)
    out_path = os.path.join(str(OUTPUTS_DIR), "results", "eval_checkpoint_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
