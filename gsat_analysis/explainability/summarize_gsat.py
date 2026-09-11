"""
GraphXAI evaluation summary for GSAT — the GSAT counterpart to
protgnn_analysis/scripts/summarize_graphxai.py and
graphcare_analysis/explainability/summarize_graphcare.py, emitting the same
column layout so all three methods drop into one comparison table.

Reads the per-graph JSONs written by explain_gsat.py
(outputs/<structure>/results/explanations/graph_*.json) and reports, per graph
per explainer (BuiltinAttention | GradExplainer | IntegratedGradExplainer |
GNNExplainer): top-k influential clinical tokens, SPARSITY, Fidelity+,
Fidelity- (shared/lib/fidelity.py — computed at generation time, read back
here), and whether the explainers agree on the single most important node.
Plus an AGGREGATE row (means + agreement rate).

Node decoding is identical to ProtGNN's (GSAT reuses ProtGNN's graphs) — the
tokens are already decoded into ``node_tokens`` at generation time, so this
script does no model/vocab loading.

Usage:
    PYTHONPATH=. .venv-protgnn/bin/python3 \
        gsat_analysis/explainability/summarize_gsat.py --graph_structure star
    # or point straight at a dir:  --explanations_dir <path>
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, os.pardir, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gsat_analysis.config import cfg
from shared.lib.fidelity import sparsity

EXPLAINERS = ["BuiltinAttention", "GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]
SHORT = {"BuiltinAttention": "attn", "GradExplainer": "grad",
         "IntegratedGradExplainer": "ig", "GNNExplainer": "gnnexp"}


def _top_node_index(ex):
    imp = ex.get("node_importance")
    if not imp:
        return None
    return int(np.argmax(np.abs(np.asarray(imp, dtype=float))))


def build_rows(files, top_nodes=4):
    rows = []
    agg = {SHORT[e]: [] for e in EXPLAINERS}
    agg_fp = {SHORT[e]: [] for e in EXPLAINERS}
    agg_fm = {SHORT[e]: [] for e in EXPLAINERS}
    n_agree = n_agree_total = 0
    for fp_path in files:
        d = json.load(open(fp_path))
        tokens = d.get("node_tokens") or []
        row = {
            "graph": d.get("graph_idx"),
            "subject_id": d.get("subject_id") if d.get("subject_id") is not None else "",
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
                row[f"{s}_sparsity"] = row[f"{s}_fidelity_plus"] = row[f"{s}_fidelity_minus"] = ""
                continue
            toks = [tokens[int(it["index"])] for it in ex.get("top_nodes", [])[:top_nodes]
                    if 0 <= int(it["index"]) < len(tokens)]
            row[f"{s}_top_factors"] = ", ".join(toks) if toks else "-"
            sp = ex.get("sparsity", sparsity(ex["node_importance"]))
            row[f"{s}_sparsity"] = sp
            agg[s].append(sp)
            vfp = ex.get("fidelity_plus", {}).get("prob")
            vfm = ex.get("fidelity_minus", {}).get("prob")
            row[f"{s}_fidelity_plus"] = vfp if vfp is not None else ""
            row[f"{s}_fidelity_minus"] = vfm if vfm is not None else ""
            if vfp is not None:
                agg_fp[s].append(vfp)
            if vfm is not None:
                agg_fm[s].append(vfm)
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
    return rows, agg, agg_fp, agg_fm, n_agree, n_agree_total


def aggregate_row(rows, agg, agg_fp, agg_fm, n_agree, n_agree_total):
    summary = {
        "graph": "AGGREGATE", "subject_id": "", "prediction": "", "actual": "",
        "result": f"{sum(1 for r in rows if r['result'] == 'correct')}/{len(rows)} correct",
        "num_nodes": "",
    }
    for e in EXPLAINERS:
        s = SHORT[e]
        vals = agg[s]
        summary[f"{s}_top_factors"] = f"mean over {len(vals)} graphs"
        summary[f"{s}_sparsity"] = round(float(np.mean(vals)), 4) if vals else ""
        summary[f"{s}_fidelity_plus"] = round(float(np.mean(agg_fp[s])), 4) if agg_fp[s] else ""
        summary[f"{s}_fidelity_minus"] = round(float(np.mean(agg_fm[s])), 4) if agg_fm[s] else ""
    summary["explainers_agree_top_node"] = (
        f"{n_agree}/{n_agree_total} = {n_agree / max(n_agree_total, 1):.2%}")
    return summary


def header_order():
    h = ["graph", "subject_id", "prediction", "actual", "result", "num_nodes"]
    for e in EXPLAINERS:
        s = SHORT[e]
        h += [f"{s}_top_factors", f"{s}_sparsity", f"{s}_fidelity_plus", f"{s}_fidelity_minus"]
    return h + ["explainers_agree_top_node"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_structure", "--graph", default=None)
    ap.add_argument("--explanations_dir", default=None)
    ap.add_argument("--top_nodes", type=int, default=4)
    args = ap.parse_args()

    if args.explanations_dir:
        expl_dir = args.explanations_dir
    else:
        structure = args.graph_structure or cfg.graph_structure
        expl_dir = os.path.join(str(cfg.OUTPUTS_DIR), structure, "results", "explanations")
    files = sorted(glob.glob(os.path.join(expl_dir, "graph_*.json")),
                   key=lambda p: int(os.path.basename(p)[6:-5]))
    if not files:
        sys.exit(f"No graph_*.json in {expl_dir}. Run explain_gsat.py first.")
    print(f"Reading {len(files)} explanations from: {expl_dir}")

    rows, agg, agg_fp, agg_fm, n_agree, n_agree_total = build_rows(files, top_nodes=args.top_nodes)
    summary = aggregate_row(rows, agg, agg_fp, agg_fm, n_agree, n_agree_total)

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

    report_path = os.path.join(out_dir, "graphxai_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 64 + "\n")
        f.write("GSAT + GraphXAI — Fidelity / Sparsity Report\n")
        f.write("=" * 64 + "\n\n")
        f.write(f"Explanations dir : {expl_dir}\n")
        f.write(f"Graphs evaluated : {len(rows)}\n")
        f.write(f"Accuracy         : {summary['result']}\n\n")
        f.write("MEAN SPARSITY / FIDELITY+ / FIDELITY- PER EXPLAINER\n")
        f.write("-" * 40 + "\n")
        for e in EXPLAINERS:
            s = SHORT[e]
            vals, fps, fms = agg[s], agg_fp[s], agg_fm[s]
            sp_str = f"{np.mean(vals):.4f}" if vals else "n/a"
            fp_str = f"{np.mean(fps):+.4f}" if fps else "n/a"
            fm_str = f"{np.mean(fms):+.4f}" if fms else "n/a"
            f.write(f"  {e:<28}: sparsity={sp_str}  fidelity+={fp_str}  fidelity-={fm_str}\n")
        f.write(f"\nTop-node agreement: {n_agree}/{n_agree_total} "
                f"({n_agree / max(n_agree_total, 1):.1%})\n")

    print("Mean sparsity / fidelity+ / fidelity- per explainer:")
    for e in EXPLAINERS:
        s = SHORT[e]
        vals, fps, fms = agg[s], agg_fp[s], agg_fm[s]
        if vals:
            print(f"  {e:<26}: sparsity={np.mean(vals):.4f}  "
                  f"fidelity+={np.mean(fps):+.4f}  fidelity-={np.mean(fms):+.4f}")
        else:
            print(f"  {e:<26}: (no data)")
    print(f"\n  {csv_path}")


if __name__ == "__main__":
    main()
