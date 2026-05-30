"""
Build a clinical-style summary (csv + xlsx + md) for the intra-patient
heterogeneous graph model, from the explanation JSONs already written under
outputs/results/explanations/.

Unlike generate_clinical_explanations.py (which assumes the patient-similarity
STAR graph with "similar patient" neighbours), this reads the intra-patient
hetero graph where nodes are PATIENT / VITAL / MED. It decodes each graph's
top influential nodes into clinical names and reports the nearest / supporting
prototype.

No retraining, no re-running explanations — pure post-processing.

Usage (from project root):
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/summarize_hetero_explanations.py
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/summarize_hetero_explanations.py --explainer GradExplainer
"""

import os
import sys
import json
import glob
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, os.pardir))
for p in (_THIS, _ROOT, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.config import data_args, OUTPUTS_DIR
from prot_gnn.load_dataset import get_dataset

RESULTS_DIR = os.path.join(str(OUTPUTS_DIR), "results")
EXPL_DIR = os.path.join(RESULTS_DIR, "explanations")
OUT_DIR = os.path.join(RESULTS_DIR, "clinical_explanations")
LABELS = {0: "HOME", 1: "ADMITTED"}


def decode_node(row, Vv, Vm, vital_vocab, med_vocab, demo_vocab):
    """Map a node feature row to a human-readable clinical token."""
    row = np.asarray(row, dtype=float)
    IDX_VAL = 3 + Vv + Vm
    IDX_ABN = IDX_VAL + 1
    if row[0] == 1:  # PATIENT
        demos = [demo_vocab[i] for i in range(len(demo_vocab))
                 if row[IDX_ABN + 1 + i] != 0]
        return "patient(" + ", ".join(d.replace("_", " ") for d in demos) + ")"
    if row[1] == 1:  # VITAL
        vid = int(np.argmax(row[3:3 + Vv]))
        z = row[IDX_VAL]
        abn = " ABNORMAL" if row[IDX_ABN] == 1 else ""
        arrow = "↑" if z > 0 else "↓"
        return f"{vital_vocab[vid]}{arrow}(z={z:+.1f}{abn})"
    if row[2] == 1:  # MED
        mid = int(np.argmax(row[3 + Vv:3 + Vv + Vm]))
        return med_vocab[mid].replace("med_", "")
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--explainer", default="IntegratedGradExplainer",
                    choices=["IntegratedGradExplainer", "GradExplainer", "GNNExplainer"])
    ap.add_argument("--top_nodes", type=int, default=4, help="# influential nodes to list")
    args = ap.parse_args()
    if args.dataset:
        data_args.dataset_name = args.dataset

    files = sorted(glob.glob(os.path.join(EXPL_DIR, "graph_*.json")),
                   key=lambda p: int(os.path.basename(p)[6:-5]))
    if not files:
        sys.exit(f"No explanation JSONs in {EXPL_DIR}. Run explain_checkpoint.py first.")

    ds = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    vital_vocab = list(getattr(ds, "vital_vocab", []))
    med_vocab = list(getattr(ds, "med_vocab", []))
    demo_vocab = list(getattr(ds, "demo_vocab", []))
    Vv, Vm = len(vital_vocab), len(med_vocab)
    print(f"Decoding with {Vv} vitals, {Vm} meds, {len(demo_vocab)} demo flags")

    rows = []
    for fp in files:
        d = json.load(open(fp))
        gi = d["graph_idx"]
        di = d.get("dataset_index")
        true_l = int(d["true_label"]); pred_l = int(d["pred_label"])
        result = "correct" if d.get("correct") else "wrong"

        # --- decode top influential nodes ---
        node_tokens = []
        ex = d.get("explanations", {}).get(args.explainer, {})
        top = ex.get("top_nodes", [])
        if di is not None and top:
            graph = ds[di]
            x = graph.x.numpy()
            for item in top[:args.top_nodes]:
                ni = int(item["index"])
                if 0 <= ni < x.shape[0]:
                    node_tokens.append(decode_node(x[ni], Vv, Vm,
                                                   vital_vocab, med_vocab, demo_vocab))

        # --- prototype evidence ---
        pe = d.get("prototype_evidence", {}) or {}
        nearest = (pe.get("top_closest") or [None])[0]
        support = (pe.get("top_contributors") or [None])[0]
        np_proto = f"P{nearest['prototype_index']}" if nearest else ""
        np_class = LABELS.get(nearest["prototype_class"], "") if nearest else ""
        np_dist = round(nearest["distance"], 4) if nearest else ""
        sp_proto = f"P{support['prototype_index']}" if support else ""
        sp_class = LABELS.get(support["prototype_class"], "") if support else ""
        sp_contrib = round(support["contribution_to_pred"], 4) if support else ""

        signals = ", ".join(node_tokens) if node_tokens else "(no node attributions)"
        reason = (f"{LABELS[pred_l]} (true {LABELS[true_l]}, {result}): "
                  f"key factors = {signals}; "
                  f"nearest prototype {np_proto} [{np_class}] dist={np_dist}; "
                  f"strongest prototype {sp_proto} [{sp_class}] contrib={sp_contrib}")

        rows.append({
            "graph": gi,
            "patient_row": di if di is not None else "",
            "prediction": LABELS[pred_l],
            "actual": LABELS[true_l],
            "result": result,
            "num_nodes": d.get("num_nodes", ""),
            "key_factor_1": node_tokens[0] if len(node_tokens) > 0 else "",
            "key_factor_2": node_tokens[1] if len(node_tokens) > 1 else "",
            "key_factor_3": node_tokens[2] if len(node_tokens) > 2 else "",
            "key_factor_4": node_tokens[3] if len(node_tokens) > 3 else "",
            "nearest_prototype": np_proto,
            "nearest_prototype_class": np_class,
            "nearest_prototype_distance": np_dist,
            "supporting_prototype": sp_proto,
            "supporting_prototype_class": sp_class,
            "supporting_prototype_contribution": sp_contrib,
            "one_line_reason": reason,
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    header = list(rows[0].keys())

    # --- csv (semicolon, matching the old summary.csv style) ---
    import csv
    csv_path = os.path.join(OUT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter=";")
        w.writeheader(); w.writerows(rows)

    # --- comma csv (Excel-friendly) ---
    csv_comma = os.path.join(OUT_DIR, "summary_comma.csv")
    with open(csv_comma, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader(); w.writerows(rows)

    # --- xlsx ---
    xlsx_path = os.path.join(OUT_DIR, "summary.xlsx")
    try:
        import pandas as pd
        pd.DataFrame(rows).to_excel(xlsx_path, index=False)
        xlsx_ok = True
    except Exception as e:
        xlsx_ok = False
        print(f"[WARN] xlsx not written ({e}); csv is available.")

    # --- md ---
    md_path = os.path.join(OUT_DIR, "summary.md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join("---" for _ in header) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in header) + " |\n")

    n = len(rows)
    n_correct = sum(1 for r in rows if r["result"] == "correct")
    print(f"\nSummarized {n} explained graphs ({n_correct} correct, {n - n_correct} wrong)")
    print(f"  csv : {csv_path}")
    print(f"  csv : {csv_comma}")
    if xlsx_ok:
        print(f"  xlsx: {xlsx_path}")
    print(f"  md  : {md_path}")
    print("\nFirst 3 reasons:")
    for r in rows[:3]:
        print("  -", r["one_line_reason"])


if __name__ == "__main__":
    main()
