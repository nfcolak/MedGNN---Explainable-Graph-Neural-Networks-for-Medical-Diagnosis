"""Per-class metrics, confusion matrix and most-confused pairs from a saved
predictions.npz (written by the *_canonical.py runs).

Run:  PYTHONPATH=. python3 comparison/error_analysis.py --run protgnn_canonical
      PYTHONPATH=. python3 comparison/error_analysis.py --run plain_gcn --no_figure

Out (next to the predictions):  per_class_metrics.csv, confusion_matrix.csv,
most_confused_pairs.csv, and optionally the heat-map used in the thesis.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="protgnn_canonical",
                    help="subfolder of comparison/ holding predictions.npz")
    ap.add_argument("--figure", default=str(REPO / "thesis" / "assets" / "confusion_matrix.pdf"))
    ap.add_argument("--no_figure", action="store_true")
    ap.add_argument("--top_pairs", type=int, default=15)
    args = ap.parse_args()

    d = REPO / "comparison" / args.run
    z = np.load(d / "predictions.npz", allow_pickle=True)
    y, pred = z["y_true"], z["y_pred"]
    classes = [str(c) for c in z["classes"]]
    n = len(classes)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y, pred, labels=np.arange(n), zero_division=0)
    per_class = pd.DataFrame({"class": classes, "precision": prec, "recall": rec,
                              "f1": f1, "support": sup}).sort_values("support", ascending=False)
    per_class.to_csv(d / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(y, pred, labels=np.arange(n))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(d / "confusion_matrix.csv")

    pairs = [(classes[i], classes[j], int(cm[i, j]), float(cm[i, j] / max(cm[i].sum(), 1)))
             for i in range(n) for j in range(n) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: -t[2])
    pd.DataFrame(pairs, columns=["true", "predicted", "count", "share_of_true_class"]) \
        .to_csv(d / "most_confused_pairs.csv", index=False)

    print(f"  accuracy {(y == pred).mean():.4f} over {len(y)} graphs")
    print("  worst recall:")
    for _, r in per_class.sort_values("recall").head(5).iterrows():
        print(f"    {r['class'][:46]:<46} recall {r['recall']:.3f}  support {int(r['support'])}")
    print("  most confused:")
    for t, p, c, s in pairs[:args.top_pairs][:5]:
        print(f"    {t[:34]:<34} -> {p[:34]:<34} {c:4d} ({s:.1%})")

    if not args.no_figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        row_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(11, 9.5))
        im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        short = [c if len(c) <= 34 else c[:33] + "…" for c in classes]
        ax.set_xticklabels(short, rotation=90, fontsize=6.5)
        ax.set_yticklabels(short, fontsize=6.5)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Share of the true class")
        fig.tight_layout()
        out = Path(args.figure); out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        print("  figure ->", out)
    print("  tables ->", d)


if __name__ == "__main__":
    main()
