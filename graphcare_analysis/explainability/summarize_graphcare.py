"""
GraphXAI evaluation summary for GraphCare — the GraphCare counterpart to
protgnn_analysis/scripts/summarize_graphxai.py, emitting the IDENTICAL column
layout so both methods drop into one comparison table.

Reads the per-graph explanation JSONs written by explain_graphcare.py
(outputs/results/explanations/graph_*.json) and, for each test graph, reports
per explainer (GradExplainer | IntegratedGradExplainer | GNNExplainer):
  - top-k influential nodes, decoded to clinical tokens via the KG vocabulary
    (node global id -> "med[...]", "symptom[...]", "cc[...]")
  - a SPARSITY score (same definition as the ProtGNN summarizer)
  - whether the explainers agree on the single most important node
plus an AGGREGATE row (mean sparsity per explainer + agreement rate).

The one GraphCare-specific difference from the ProtGNN summarizer is node
decoding: ProtGNN decodes feature-layout rows; GraphCare decodes GLOBAL KG node
ids through kg["ent2id"] (inverted). Everything else — columns, delimiter,
aggregate wording — is matched so the two summary.csv files are comparable.

Usage:
    PYTHONPATH=.:external/GraphCare .venv-graphcare/bin/python3 \
        graphcare_analysis/explainability/summarize_graphcare.py
    # optional: --explanations_dir <path>  --top_nodes 4
"""
import os
import sys
import csv
import glob
import json
import argparse

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graphcare_analysis.config import cfg

EXPLAINERS = ["GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]
SHORT = {"GradExplainer": "grad", "IntegratedGradExplainer": "ig", "GNNExplainer": "gnnexp"}


def sparsity(node_importance, mass=0.9):
    """Identical definition to summarize_graphxai.py."""
    imp = np.abs(np.asarray(node_importance, dtype=float))
    n = len(imp)
    if n == 0 or imp.sum() == 0:
        return 0.0
    order = np.sort(imp)[::-1]
    cum = np.cumsum(order) / imp.sum()
    k = int(np.searchsorted(cum, mass) + 1)
    return round(1.0 - k / n, 4)


def load_id2ent(kg_path=None):
    kg = torch.load(kg_path or cfg.kg_path)
    return {int(i): name for name, i in kg["ent2id"].items()}


def decode_token(global_id, id2ent):
    """Global KG node id -> clinical token. Mirrors ProtGNN's decode style."""
    name = id2ent.get(int(global_id), f"node{int(global_id)}")
    if name.startswith("med:"):
        return f"med[{name[4:]}]"
    if name.startswith("sym:"):
        return f"symptom[{name[4:]}]"
    if name.startswith("cc:"):
        return f"cc[{name[3:]}]"
    return name


def _top_node_index(ex):
    imp = ex.get("node_importance")
    if not imp:
        return None
    return int(np.argmax(np.abs(np.asarray(imp, dtype=float))))


def build_rows(files, id2ent, top_nodes=4):
    rows = []
    agg = {SHORT[e]: [] for e in EXPLAINERS}
    n_agree = n_agree_total = 0
    for fp in files:
        d = json.load(open(fp))
        gids = d.get("node_global_ids") or []
        row = {
            "graph": d.get("graph_idx"),
            "patient_row": d.get("dataset_index") if d.get("dataset_index") is not None else "",
            "prediction": d.get("pred_class", d.get("pred_label")),
            "actual": d.get("true_class", d.get("true_label")),
            "result": "correct" if d.get("correct") else "wrong",
            "num_nodes": d.get("num_nodes", ""),
        }
        top_idx_per_expl = {}
        for e in EXPLAINERS:
            s = SHORT[e]
            ex = d.get("explanations", {}).get(e, {})
            if "error" in ex or "node_importance" not in ex:
                row[f"{s}_top_factors"] = f"(error: {ex.get('error', 'n/a')})"
                row[f"{s}_sparsity"] = ""
                continue
            toks = []
            for item in ex.get("top_nodes", [])[:top_nodes]:
                li = int(item["index"])                 # LOCAL subgraph index
                if 0 <= li < len(gids):
                    toks.append(decode_token(gids[li], id2ent))
            row[f"{s}_top_factors"] = ", ".join(toks) if toks else "-"
            sp = sparsity(ex["node_importance"])
            row[f"{s}_sparsity"] = sp
            agg[s].append(sp)
            top_idx_per_expl[s] = _top_node_index(ex)
        idxs = [v for v in top_idx_per_expl.values() if v is not None]
        if len(idxs) >= 2:
            n_agree_total += 1
            agree = len(set(idxs)) == 1
            row["explainers_agree_top_node"] = bool(agree)
            n_agree += int(agree)
        else:
            row["explainers_agree_top_node"] = ""
        rows.append(row)
    return rows, agg, n_agree, n_agree_total


def aggregate_row(rows, agg, n_agree, n_agree_total):
    summary = {
        "graph": "AGGREGATE", "patient_row": "", "prediction": "", "actual": "",
        "result": f"{sum(1 for r in rows if r['result'] == 'correct')}/{len(rows)} correct",
        "num_nodes": "",
    }
    for e in EXPLAINERS:
        s = SHORT[e]
        vals = agg[s]
        summary[f"{s}_top_factors"] = f"mean_sparsity over {len(vals)} graphs"
        summary[f"{s}_sparsity"] = round(float(np.mean(vals)), 4) if vals else ""
    summary["explainers_agree_top_node"] = (
        f"{n_agree}/{n_agree_total} = {n_agree / max(n_agree_total, 1):.2%}")
    return summary


def header_order():
    h = ["graph", "patient_row", "prediction", "actual", "result", "num_nodes"]
    for e in EXPLAINERS:
        s = SHORT[e]
        h += [f"{s}_top_factors", f"{s}_sparsity"]
    h += ["explainers_agree_top_node"]
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explanations_dir", default=None)
    ap.add_argument("--top_nodes", type=int, default=4)
    args = ap.parse_args()

    expl_dir = args.explanations_dir or os.path.join(str(cfg.OUTPUTS_DIR), "results", "explanations")
    files = sorted(glob.glob(os.path.join(expl_dir, "graph_*.json")),
                   key=lambda p: int(os.path.basename(p)[6:-5]))
    if not files:
        sys.exit(f"No graph_*.json in {expl_dir}. Run explain_graphcare.py first.")
    print(f"Reading {len(files)} explanations from: {expl_dir}")

    id2ent = load_id2ent()
    rows, agg, n_agree, n_agree_total = build_rows(files, id2ent, top_nodes=args.top_nodes)
    summary = aggregate_row(rows, agg, n_agree, n_agree_total)

    out_dir = os.path.dirname(expl_dir)
    os.makedirs(out_dir, exist_ok=True)
    header = header_order()

    csv_path = os.path.join(out_dir, "graphxai_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter=";")
        w.writeheader(); w.writerows(rows); w.writerow(summary)
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

    print("Mean sparsity per explainer:")
    for e in EXPLAINERS:
        s = SHORT[e]; vals = agg[s]
        print(f"  {e:<26}: {np.mean(vals):.4f}" if vals else f"  {e:<26}: (no data)")
    print(f"Top-node agreement: {n_agree}/{n_agree_total} "
          f"({n_agree / max(n_agree_total, 1):.1%})")
    print(f"\n  {csv_path}")


if __name__ == "__main__":
    main()