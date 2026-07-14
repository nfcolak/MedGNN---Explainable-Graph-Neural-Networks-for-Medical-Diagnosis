"""
Generate explanations + clinical summary for an ALREADY-TRAINED checkpoint,
WITHOUT retraining.

Reuses train_and_explain.py's explain_test_set() + save_results() machinery,
but loads weights from a saved checkpoint instead of training from scratch.
save_results() calls _prepare_results_dir(), which deletes the old
outputs/results/explanations/ and clinical_explanations/ folders first — so
the previous model's summary is removed and replaced with this model's.

Then it runs generate_clinical_explanations.py to (re)build
outputs/results/clinical_explanations/summary.{csv,xlsx,md}.

Usage (from project root):
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/explain_checkpoint.py --explain_n 20
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/explain_checkpoint.py --explain_n -1   # all test graphs
"""

import os
import re
import sys
import glob
import argparse
import subprocess

import numpy as np
import torch
import torch.nn as nn

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, os.pardir))
for p in (_THIS, _ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "external", "GraphXAI-main")):
    if p not in sys.path:
        sys.path.insert(0, p)

from protgnn_analysis.config import data_args, model_args, train_args
from protgnn_analysis.models import GnnNets
from protgnn_analysis.load_dataset import get_dataset, get_dataloader
from protgnn_analysis import train as TE

_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*Train Loss:\s*([\d.]+)\s*Acc:\s*([\d.]+)\s*\|\s*"
    r"Eval Loss:\s*([\d.]+)\s*Acc:\s*([\d.]+)(?:\s*PR-AUC:\s*([\d.]+))?"
)


def parse_epoch_rows(log_path):
    """Reconstruct epoch_rows from a training log so save_results/report.txt work."""
    rows = []
    if not log_path or not os.path.isfile(log_path):
        return rows
    with open(log_path) as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if not m:
                continue
            ep, tl, ta, el, ea, pr = m.groups()
            rows.append({
                "epoch": int(ep),
                "train_loss": float(tl), "train_acc": float(ta),
                "eval_loss": float(el), "eval_acc": float(ea),
                "eval_f1": "", "eval_pr_auc": float(pr) if pr else "",
                "eval_recall": "", "eval_precision": "",
            })
    return rows


def find_latest_log():
    logs = glob.glob(os.path.join(_ROOT, "outputs", "logs", "*.log"))
    return max(logs, key=os.path.getmtime) if logs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="override dataset name")
    ap.add_argument("--which", default="latest", choices=["latest", "best"],
                    help="checkpoint to load (prototype mode → 'latest' for fresh prototypes)")
    ap.add_argument("--explain_n", type=int, default=20, help="# test graphs to explain (-1 = all)")
    ap.add_argument("--clst", type=float, default=0.02, help="recorded in model_config only")
    ap.add_argument("--sep", type=float, default=0.1, help="recorded in model_config only")
    ap.add_argument("--clinical_limit", type=int, default=None,
                    help="limit for generate_clinical_explanations (default: same as explain_n)")
    ap.add_argument("--skip_clinical", action="store_true",
                    help="only produce explanation JSONs, skip the summary.xlsx step")
    ap.add_argument("--log", default=None,
                    help="training log to rebuild training_metrics.csv (default: newest in outputs/logs/)")
    args = ap.parse_args()
    if args.dataset:
        data_args.dataset_name = args.dataset

    # --- Data ---
    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name, task=data_args.task)
    output_dim = int(dataset.num_classes)
    dataloader = get_dataloader(
        dataset, train_args.batch_size,
        random_split_flag=data_args.random_split,
        data_split_ratio=data_args.data_split_ratio, seed=data_args.seed,
    )

    # --- Model from checkpoint (NO training) ---
    gnn_nets = GnnNets(dataset.num_node_features, output_dim, model_args)
    gnn_nets.to_device()
    ckpt_path = os.path.join(model_args.checkpoint, data_args.dataset_name,
                             f"{model_args.model_name}_{args.which}.pth")
    if not os.path.isfile(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=model_args.device)
    gnn_nets.update_state_dict(ckpt["net"])
    gnn_nets.eval()
    print(f"Loaded {args.which} checkpoint: epoch={ckpt.get('epoch')} "
          f"val_metric={ckpt.get('acc'):.4f}\n  {ckpt_path}")

    # --- Class-weighted criterion (mirrors training) ---
    train_idx = dataloader["train"].dataset.indices
    eval_idx = dataloader["eval"].dataset.indices
    test_idx = dataloader["test"].dataset.indices
    train_labels = np.array([int(dataset[i].y.view(-1)[0].item()) for i in train_idx])
    eval_labels_s = np.array([int(dataset[i].y.view(-1)[0].item()) for i in eval_idx])
    test_labels_s = np.array([int(dataset[i].y.view(-1)[0].item()) for i in test_idx])
    class_weights, _ = TE._compute_class_weights(train_labels, output_dim)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- Calibrate threshold on eval ---
    calibrated_threshold = None
    diagnostics = {
        "label_distribution": {
            "train": TE._label_distribution(train_labels, output_dim),
            "eval": TE._label_distribution(eval_labels_s, output_dim),
            "test": TE._label_distribution(test_labels_s, output_dim),
        }
    }
    if output_dim == 2:
        _l, ev_y, _p, ev_probs = TE._collect_predictions(dataloader["eval"], gnn_nets, criterion)
        calibrated_threshold, tscore = TE._find_best_threshold(ev_y, ev_probs[:, 1], "balanced_acc")
        print(f"Calibrated threshold: {calibrated_threshold:.3f} (eval balanced_acc={tscore:.4f})")

    # --- Test metrics + threshold sensitivity ---
    test_loss, test_y, test_argmax, test_probs = TE._collect_predictions(
        dataloader["test"], gnn_nets, criterion
    )
    if output_dim == 2:
        tvals = [0.3, 0.4, 0.45, 0.5, 0.6, 0.7]
        if calibrated_threshold is not None:
            tvals.append(calibrated_threshold)
        diagnostics["threshold_sensitivity"] = TE._threshold_sensitivity(test_y, test_probs[:, 1], tvals)
        diagnostics["confusion_matrices"] = {"argmax": TE._confusion_counts(test_y, test_argmax)}
    test_state = TE._state_from_predictions(test_loss, test_y, test_argmax, test_probs,
                                            threshold=calibrated_threshold)
    print("\nTEST RESULTS")
    for k, v in test_state.items():
        print(f"  {k:<16}: {v}")

    # --- Explanation pass (reuses existing GraphXAI machinery) ---
    records = TE.explain_test_set(gnn_nets, output_dim, args.explain_n, threshold=calibrated_threshold)

    # --- Save everything (this DELETES old explanations + clinical_explanations) ---
    epoch_header = ["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc",
                    "eval_f1", "eval_pr_auc", "eval_recall", "eval_precision"]
    log_path = args.log or find_latest_log()
    epoch_rows = parse_epoch_rows(log_path)
    if epoch_rows:
        print(f"Parsed {len(epoch_rows)} epoch rows from log: {log_path}")
    else:
        # Fallback: synthesize a single row from the checkpoint so report.txt works.
        print("No training log parsed — writing a single synthetic epoch row.")
        epoch_rows = [{
            "epoch": int(ckpt.get("epoch", 0)),
            "train_loss": "", "train_acc": "",
            "eval_loss": "", "eval_acc": float(ckpt.get("acc", 0.0)),
            "eval_f1": "", "eval_pr_auc": float(ckpt.get("acc", 0.0)),
            "eval_recall": "", "eval_precision": "",
        }]
    TE.save_results(
        epoch_rows=epoch_rows, epoch_header=epoch_header,
        test_state=test_state, diagnostics=diagnostics,
        explanation_records=records,
        clst=args.clst, sep=args.sep, output_dim=output_dim,
        use_prot=model_args.enable_prot, threshold=calibrated_threshold,
    )
    print(f"\nWrote results to {TE.RESULTS_DIR}/ (timestamped run, prior runs kept)")

    # --- Clinical summary (summary.csv / .xlsx / .md) ---
    if not args.skip_clinical:
        limit = args.clinical_limit if args.clinical_limit is not None else max(args.explain_n, 1)
        cmd = [sys.executable, os.path.join(_ROOT, "scripts", "generate_clinical_explanations.py"),
               "--results_dir", TE.RESULTS_DIR]
        if args.explain_n != -1:
            cmd += ["--limit", str(limit)]
        env = dict(os.environ)
        env["PYTHONPATH"] = f"src:external/GraphXAI-main:.{os.pathsep}{env.get('PYTHONPATH','')}"
        print(f"\nRunning clinical-explanation summary: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=_ROOT, env=env)
        print(f"\nsummary.xlsx → {os.path.join(TE.RESULTS_DIR, 'clinical_explanations', 'summary.xlsx')}")


if __name__ == "__main__":
    main()
