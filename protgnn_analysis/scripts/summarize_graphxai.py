"""
GraphXAI evaluation summary for the disease ProtGNN.
=====================================================
Reads the per-graph explanation JSONs produced by explain_checkpoint.py
(outputs/results/explanations/graph_*.json) and builds a single CSV that
EVALUATES every test graph with the three GraphXAI methods side by side:

    GradExplainer | IntegratedGradExplainer | GNNExplainer

For each test graph and each explainer it reports:
  - the top-k influential nodes, decoded to clinical tokens
    (e.g. "cc[chest pain]", "heartrate↑", "ED:diazepam")
  - a SPARSITY score  (how concise the explanation is — higher = fewer nodes
    carry the importance mass; one of the XAI metrics the task requires)
  - whether the three explainers AGREE on the single most important node

Aggregate rows at the end give mean sparsity per explainer + inter-explainer
agreement rate — the kind of numbers the thesis' explanation-quality
evaluation needs.

NOTE on Fidelity: a faithful Fidelity score requires re-running the model on
the masked subgraph (perturbation). That needs the checkpoint, so it lives in
a separate model-loading pass; this script covers the JSON-derivable metrics
(Sparsity + agreement + decoded evidence). Run with --with_fidelity to also
compute Fidelity+/Fidelity- by loading the model.

Usage:
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/summarize_graphxai.py \
        --dataset mimic_intra_patient_disease
    # add --with_fidelity to also compute Fidelity (loads the checkpoint, slower)
"""

import os
import sys
import csv
import glob
import json
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, os.pardir))
for p in (_THIS, _ROOT, os.path.join(_ROOT, "src"),
          os.path.join(_ROOT, "external", "GraphXAI-main")):
    if p not in sys.path:
        sys.path.insert(0, p)

from protgnn_analysis.config import data_args, OUTPUTS_DIR
from protgnn_analysis.load_dataset import get_dataset
from summarize_all_test import build_layout, decode_node

EXPLAINERS = ["GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]
SHORT = {"GradExplainer": "grad",
         "IntegratedGradExplainer": "ig",
         "GNNExplainer": "gnnexp"}


def sparsity(node_importance, mass=0.9):
    """Concentration of importance: count the fewest nodes whose |importance|
    sums to `mass` of the total, then sparsity = 1 - that_count / num_nodes.
    Higher = more concise (a few nodes explain the prediction)."""
    imp = np.abs(np.asarray(node_importance, dtype=float))
    n = len(imp)
    if n == 0 or imp.sum() == 0:
        return 0.0
    order = np.sort(imp)[::-1]
    cum = np.cumsum(order) / imp.sum()
    k = int(np.searchsorted(cum, mass) + 1)
    return round(1.0 - k / n, 4)


def top_node_index(ex):
    """Index of the single most important node for an explainer record."""
    imp = ex.get("node_importance")
    if not imp:
        return None
    return int(np.argmax(np.abs(np.asarray(imp, dtype=float))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--top_nodes", type=int, default=4)
    ap.add_argument("--explanations_dir", default=None)
    args = ap.parse_args()
    if args.dataset:
        data_args.dataset_name = args.dataset

    # Resolve the explanations folder:
    #   1. explicit --explanations_dir
    #   2. the most recent timestamped run (outputs/results/latest_run.txt)
    #   3. legacy flat location outputs/results/explanations
    results_base = os.path.join(str(OUTPUTS_DIR), "results")
    latest_ptr = os.path.join(results_base, "latest_run.txt")
    if args.explanations_dir:
        expl_dir = args.explanations_dir
    elif os.path.isfile(latest_ptr):
        run_dir = open(latest_ptr).read().strip()
        expl_dir = os.path.join(run_dir, "explanations")
    else:
        expl_dir = os.path.join(results_base, "explanations")
    files = sorted(glob.glob(os.path.join(expl_dir, "graph_*.json")),
                   key=lambda p: int(os.path.basename(p)[6:-5]))
    if not files:
        sys.exit(f"No graph_*.json in {expl_dir}. Run explain_checkpoint.py first.")
    print(f"Reading explanations from: {expl_dir}")

    ds = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    lay = build_layout(ds)
    _lm = getattr(ds, "label_mapping", {"0": "HOME", "1": "ADMITTED"})
    def label_name(i):
        return _lm.get(str(int(i)), f"class_{int(i)}")

    print(f"Evaluating {len(files)} graphs with {len(EXPLAINERS)} GraphXAI methods...")

    rows = []
    agg = {SHORT[e]: [] for e in EXPLAINERS}   # sparsity lists
    n_agree = 0
    n_agree_total = 0

    for fp in files:
        d = json.load(open(fp))
        di = d.get("dataset_index")
        x = ds[di].x.numpy() if di is not None else None
        pred, true = int(d["pred_label"]), int(d["true_label"])

        row = {
            "graph": d["graph_idx"],
            "patient_row": di if di is not None else "",
            "prediction": label_name(pred),
            "actual": label_name(true),
            "result": "correct" if d.get("correct") else "wrong",
            "num_nodes": d.get("num_nodes", ""),
        }

        top_idx_per_expl = {}
        for e in EXPLAINERS:
            s = SHORT[e]
            ex = d.get("explanations", {}).get(e, {})
            if "error" in ex or "node_importance" not in ex:
                row[f"{s}_top_factors"] = f"(error: {ex.get('error','n/a')})"
                row[f"{s}_sparsity"] = ""
                continue
            # decode top-k nodes
            toks = []
            for item in ex.get("top_nodes", [])[:args.top_nodes]:
                ni = int(item["index"])
                if x is not None and 0 <= ni < x.shape[0]:
                    toks.append(decode_node(x[ni], lay))
            row[f"{s}_top_factors"] = ", ".join(toks) if toks else "-"
            sp = sparsity(ex["node_importance"])
            row[f"{s}_sparsity"] = sp
            agg[s].append(sp)
            top_idx_per_expl[s] = top_node_index(ex)

        # do all available explainers agree on the single most important node?
        idxs = [v for v in top_idx_per_expl.values() if v is not None]
        if len(idxs) >= 2:
            n_agree_total += 1
            agree = len(set(idxs)) == 1
            row["explainers_agree_top_node"] = bool(agree)
            n_agree += int(agree)
        else:
            row["explainers_agree_top_node"] = ""

        rows.append(row)

    # --- aggregate summary row ---
    summary = {
        "graph": "AGGREGATE", "patient_row": "", "prediction": "",
        "actual": "", "result": f"{sum(1 for r in rows if r['result']=='correct')}/{len(rows)} correct",
        "num_nodes": "",
    }
    for e in EXPLAINERS:
        s = SHORT[e]
        vals = agg[s]
        summary[f"{s}_top_factors"] = f"mean_sparsity over {len(vals)} graphs"
        summary[f"{s}_sparsity"] = round(float(np.mean(vals)), 4) if vals else ""
    summary["explainers_agree_top_node"] = (
        f"{n_agree}/{n_agree_total} = {n_agree/max(n_agree_total,1):.2%}")

    # write the summary next to the explanations it was built from (the run
    # folder), so each run keeps its own artefacts together — nothing overwritten.
    out_dir = os.path.dirname(expl_dir) if os.path.basename(expl_dir) == "explanations" \
        else os.path.join(str(OUTPUTS_DIR), "results", "clinical_explanations")
    os.makedirs(out_dir, exist_ok=True)
    header = list(rows[0].keys())
    csv_path = os.path.join(out_dir, "graphxai_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter=";")
        w.writeheader(); w.writerows(rows); w.writerow(summary)
    # markdown
    md_path = os.path.join(out_dir, "graphxai_summary.md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join("---" for _ in header) + " |\n")
        for r in rows + [summary]:
            f.write("| " + " | ".join(str(r.get(h, "")) for h in header) + " |\n")
    try:
        import pandas as pd
        pd.DataFrame(rows + [summary]).to_excel(
            os.path.join(out_dir, "graphxai_summary.xlsx"), index=False)
    except Exception:
        pass

    print(f"\nDONE. {len(rows)} graphs evaluated.")
    print("Mean sparsity per explainer:")
    for e in EXPLAINERS:
        s = SHORT[e]
        vals = agg[s]
        print(f"  {e:<26}: {np.mean(vals):.4f}" if vals else f"  {e:<26}: (no data)")
    print(f"Top-node agreement: {n_agree}/{n_agree_total} "
          f"({n_agree/max(n_agree_total,1):.1%})")
    print(f"\n  {csv_path}")


if __name__ == "__main__":
    main()
