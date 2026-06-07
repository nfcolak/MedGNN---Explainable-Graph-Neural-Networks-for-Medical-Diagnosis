"""
Run train_and_explain.py over multiple seeds and aggregate metrics.

Usage:
    PYTHONPATH=src:external/GraphXAI-main:. \\
        python3 scripts/run_multi_seed.py --seeds 1 2 3 4 5 \\
        -- --clst 0.02 --sep 0.1 --explain_n 0 --archive_tag multiseed

Anything after `--` is forwarded verbatim to train_and_explain.py for each
seed (with `--seed <s>` appended and `--archive_tag <tag>_seed<s>`).

The script reads `outputs/results/test_metrics.json` after each run and writes
`outputs/results/multi_seed_summary.json` with per-seed metrics and mean/std.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
TEST_METRICS = RESULTS_DIR / "test_metrics.json"
SUMMARY_PATH = RESULTS_DIR / "multi_seed_summary.json"

METRIC_KEYS = ["acc", "pr_auc", "precision", "recall", "specificity", "f1", "balanced_acc"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, required=True,
                   help="Seeds to iterate, e.g. --seeds 1 2 3 4 5")
    p.add_argument("--tag", default="multiseed",
                   help="Tag prefix for archived runs (default: multiseed)")
    p.add_argument("forwarded", nargs=argparse.REMAINDER,
                   help="Args after `--` are forwarded to train_and_explain.py")
    return p.parse_args()


def run_seed(seed, tag, forwarded):
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "train_and_explain.py"),
           "--seed", str(seed), "--archive_tag", f"{tag}_seed{seed}"]
    extras = [a for a in forwarded if a != "--"]
    cmd.extend(extras)
    print("\n" + "=" * 70)
    print(f"[multi-seed] seed={seed}")
    print(" ".join(cmd))
    print("=" * 70)
    subprocess.run(cmd, check=True)
    with open(TEST_METRICS) as f:
        return json.load(f)


def aggregate(per_seed):
    summary = {"per_seed": per_seed, "aggregate": {}}
    for key in METRIC_KEYS:
        vals = [m[key] for m in per_seed.values() if isinstance(m.get(key), (int, float))]
        if not vals:
            continue
        summary["aggregate"][key] = {
            "mean": round(statistics.mean(vals), 6),
            "std": round(statistics.pstdev(vals), 6) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 6),
            "max": round(max(vals), 6),
            "n": len(vals),
        }
    return summary


def main():
    args = parse_args()
    forwarded = args.forwarded[1:] if args.forwarded and args.forwarded[0] == "--" else args.forwarded
    per_seed = {}
    for s in args.seeds:
        per_seed[str(s)] = run_seed(s, args.tag, forwarded or [])
    summary = aggregate(per_seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 70)
    print("[multi-seed] DONE")
    for key, stats in summary["aggregate"].items():
        print(f"  {key:<14}: mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"(min={stats['min']:.4f}, max={stats['max']:.4f}, n={stats['n']})")
    print(f"\nWrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
