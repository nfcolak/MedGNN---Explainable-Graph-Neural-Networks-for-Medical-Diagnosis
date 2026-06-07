"""
Fast clinical summary for the ENTIRE test set (all ~2000 patients) of the
intra-patient hetero-graph ProtGNN — without the slow GraphXAI pass.

Why this is fast: the clinical summary only needs
  (a) prototype evidence  -> one forward pass per graph
  (b) per-node saliency    -> one backward pass per graph (gradient of the
                              predicted-class logit w.r.t. node features),
                              equivalent to GradExplainer but inline.
Both run on the configured device (MPS / CUDA / CPU). No IntegratedGrad
(40 forward passes) and no GNNExplainer (optimisation loop), so all 2000
test graphs finish in minutes instead of hours.

Outputs (overwrites the 20-graph clinical summary):
  outputs/results/clinical_explanations/summary.csv
  outputs/results/clinical_explanations/summary_comma.csv
  outputs/results/clinical_explanations/summary.xlsx   (if openpyxl available)
  outputs/results/clinical_explanations/summary.md

Usage (from project root):
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/summarize_all_test.py
    PYTHONPATH=src:external/GraphXAI-main:. python3 scripts/summarize_all_test.py --limit 500
"""

import os
import sys
import csv
import argparse

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, os.pardir))
for p in (_THIS, _ROOT, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.config import data_args, model_args, train_args, OUTPUTS_DIR
from prot_gnn.models import GnnNets
from prot_gnn.load_dataset import get_dataset, get_dataloader

OUT_DIR = os.path.join(str(OUTPUTS_DIR), "results", "clinical_explanations")


def build_layout(ds):
    """Compute feature-vector offsets for the 6-type intra-patient graph,
    matching IntraPatientHeteroDataset.process() exactly:
      [type(6) | vital_id | med_id | icd_id | sym_id | cc_id |
       value | abnormal | missing | demo | pnum]
    """
    vital = list(getattr(ds, "vital_vocab", []))
    med   = list(getattr(ds, "med_vocab", []))
    icd   = list(getattr(ds, "icd_vocab", []))
    sym   = list(getattr(ds, "symptom_vocab", []))
    cc    = list(getattr(ds, "cc_vocab", []))
    demo  = list(getattr(ds, "demo_vocab", []))
    Vv, Vm, Vi, Vs, Vc, Dd = len(vital), len(med), len(icd), len(sym), len(cc), len(demo)
    N_TYPES = 6
    IV = N_TYPES
    IM = IV + Vv
    II = IM + Vm
    IS = II + Vi
    IC = IS + Vs
    IDX_VAL = IC + Vc
    return {
        "Vv": Vv, "Vm": Vm, "Vi": Vi, "Vs": Vs, "Vc": Vc, "Dd": Dd,
        "vital": IV, "med": IM, "icd": II, "sym": IS, "cc": IC,
        "value": IDX_VAL, "abn": IDX_VAL + 1, "miss": IDX_VAL + 2,
        "demo": IDX_VAL + 3,
        "vital_vocab": vital, "med_vocab": med, "icd_vocab": icd,
        "sym_vocab": sym, "cc_vocab": cc, "demo_vocab": demo,
    }


def decode_node(row, lay):
    """Human-readable token for one node row (6-type layout)."""
    row = np.asarray(row, dtype=float)
    t = int(np.argmax(row[:6]))     # 0 pat,1 vital,2 med,3 icd,4 sym,5 cc
    IDX_VAL, IDX_ABN = lay["value"], lay["abn"]
    if t == 0:
        dv, di = lay["demo_vocab"], lay["demo"]
        demos = [dv[i] for i in range(len(dv)) if row[di + i] != 0]
        return "patient(" + ", ".join(d.replace("_", " ") for d in demos) + ")"
    if t == 1 and lay["Vv"]:
        vid = int(np.argmax(row[lay["vital"]:lay["vital"] + lay["Vv"]]))
        z = row[IDX_VAL]
        abn = " ABNORMAL" if row[IDX_ABN] == 1 else ""
        return f"{lay['vital_vocab'][vid]}{'↑' if z > 0 else '↓'}(z={z:+.1f}{abn})"
    if t == 2 and lay["Vm"]:
        mid = int(np.argmax(row[lay["med"]:lay["med"] + lay["Vm"]]))
        nm = lay["med_vocab"][mid]
        return "ED:" + nm[4:] if nm.startswith("pyx_") else nm.replace("med_", "")
    if t == 3 and lay["Vi"]:
        iid = int(np.argmax(row[lay["icd"]:lay["icd"] + lay["Vi"]]))
        return f"icd[{lay['icd_vocab'][iid]}]"
    if t == 4 and lay["Vs"]:
        sid = int(np.argmax(row[lay["sym"]:lay["sym"] + lay["Vs"]]))
        return f"sym[{lay['sym_vocab'][sid]}]"
    if t == 5 and lay["Vc"]:
        cid = int(np.argmax(row[lay["cc"]:lay["cc"] + lay["Vc"]]))
        return f"cc[{lay['cc_vocab'][cid]}]"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--which", default="latest", choices=["latest", "best"])
    ap.add_argument("--limit", type=int, default=-1, help="max test graphs (-1 = all)")
    ap.add_argument("--top_nodes", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=None,
                    help="decision threshold on P(ADMITTED); default = argmax")
    args = ap.parse_args()
    if args.dataset:
        data_args.dataset_name = args.dataset

    device = torch.device(model_args.device)
    ds = get_dataset(data_args.dataset_dir, data_args.dataset_name)
    dl = get_dataloader(ds, 1, random_split_flag=data_args.random_split,
                        data_split_ratio=data_args.data_split_ratio, seed=data_args.seed)

    lay = build_layout(ds)
    # class id -> human name (disease names for 30-class, HOME/ADMITTED for binary)
    _lm = getattr(ds, "label_mapping", {"0": "HOME", "1": "ADMITTED"})
    def label_name(i):
        return _lm.get(str(int(i)), f"class_{int(i)}")

    gnn = GnnNets(ds.num_node_features, ds.num_classes, model_args)
    gnn.to_device()
    ckpt_path = os.path.join(model_args.checkpoint, data_args.dataset_name,
                             f"{model_args.model_name}_{args.which}.pth")
    if not os.path.isfile(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    gnn.update_state_dict(ckpt["net"])
    gnn.eval()
    model = gnn.model
    num_per_class = max(1, model.num_prototypes // ds.num_classes)
    epsilon = float(getattr(model, "epsilon", 1e-4))
    last_w = model.last_layer.weight.detach().cpu().numpy()
    print(f"Loaded {args.which} ckpt epoch={ckpt.get('epoch')} | device={device} | "
          f"prototypes={model.num_prototypes} ({num_per_class}/class)")

    test_loader = dl["test"]
    n_total = len(test_loader)
    limit = n_total if args.limit < 0 else min(args.limit, n_total)
    print(f"Summarizing {limit}/{n_total} test graphs...\n")

    rows = []
    for i, batch in enumerate(test_loader):
        if i >= limit:
            break
        batch = batch.to(device)
        x_in = batch.x.detach().clone().requires_grad_(True)
        data_like = batch.clone()
        data_like.x = x_in

        logits, probs, _ne, _ge, min_distances = gnn(data_like)
        probs_np = probs.detach().cpu().numpy().reshape(-1)
        if args.threshold is not None and probs_np.shape[0] == 2:
            pred = int(probs_np[1] >= args.threshold)
        else:
            pred = int(np.argmax(probs_np))
        true_l = int(batch.y.view(-1)[0].item())

        # --- node saliency: d logit[pred] / d x, summed |grad| per node ---
        node_tokens, node_importance = [], []
        try:
            gnn.zero_grad(set_to_none=True)
            if x_in.grad is not None:
                x_in.grad = None
            logits[0, pred].backward(retain_graph=False)
            sal = x_in.grad.detach().abs().sum(dim=1).cpu().numpy()
            total_sal = float(sal.sum()) or 1.0
            order = np.argsort(-sal)[:args.top_nodes]
            xrows = batch.x.detach().cpu().numpy()
            for ni in order:
                node_tokens.append(decode_node(xrows[int(ni)], lay))
                # importance = this node's share of the graph's total saliency (%)
                node_importance.append(round(100.0 * float(sal[int(ni)]) / total_sal, 1))
        except Exception as e:
            node_tokens = [f"(saliency error: {e})"]
            node_importance = [0.0]

        # --- prototype evidence ---
        dist = min_distances.detach().cpu().numpy().reshape(-1)
        act = np.log((dist + 1.0) / (dist + epsilon))
        contrib = act * last_w[pred]
        nearest_i = int(np.argmin(dist))
        support_i = int(np.argmax(np.abs(contrib)))

        di = int(batch.dataset_index.view(-1)[0].item()) if hasattr(batch, "dataset_index") else ""
        row = {
            "graph": i,
            "patient_row": di,
            "prediction": label_name(pred),
            "actual": label_name(true_l),
            "result": "correct" if pred == true_l else "wrong",
            "p_pred": round(float(probs_np[pred]), 4),
            "num_nodes": int(batch.x.shape[0]),
        }
        # dynamic key-factor columns + their importance (% of graph saliency)
        for j in range(args.top_nodes):
            row[f"key_factor_{j+1}"] = node_tokens[j] if j < len(node_tokens) else ""
            row[f"key_factor_{j+1}_importance_%"] = node_importance[j] if j < len(node_importance) else ""
        row.update({
            "nearest_prototype": f"P{nearest_i}",
            "nearest_prototype_class": label_name(nearest_i // num_per_class),
            "nearest_prototype_distance": round(float(dist[nearest_i]), 4),
            "supporting_prototype": f"P{support_i}",
            "supporting_prototype_class": label_name(support_i // num_per_class),
            "supporting_prototype_contribution": round(float(contrib[support_i]), 4),
        })
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{limit} done")

    # --- reason column (factors with their importance %) ---
    for r in rows:
        kf = []
        for j in range(1, args.top_nodes + 1):
            tok = r.get(f"key_factor_{j}", "")
            imp = r.get(f"key_factor_{j}_importance_%", "")
            if tok:
                kf.append(f"{tok} ({imp}%)")
        r["one_line_reason"] = (
            f"{r['prediction']} (true {r['actual']}, {r['result']}): "
            f"key factors = {', '.join(kf) if kf else 'n/a'}; "
            f"nearest prototype {r['nearest_prototype']} [{r['nearest_prototype_class']}] "
            f"dist={r['nearest_prototype_distance']}; strongest {r['supporting_prototype']} "
            f"[{r['supporting_prototype_class']}] contrib={r['supporting_prototype_contribution']}"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    header = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, delimiter=";"); w.writeheader(); w.writerows(rows)
    with open(os.path.join(OUT_DIR, "summary_comma.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
    xlsx_ok = False
    try:
        import pandas as pd
        pd.DataFrame(rows).to_excel(os.path.join(OUT_DIR, "summary.xlsx"), index=False)
        xlsx_ok = True
    except Exception as e:
        print(f"[WARN] xlsx skipped ({e}); install openpyxl for Excel output.")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("| " + " | ".join("---" for _ in header) + " |\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[h]) for h in header) + " |\n")

    n = len(rows)
    nc = sum(1 for r in rows if r["result"] == "correct")
    print(f"\nDONE. {n} graphs | correct={nc} ({nc/n:.1%}) wrong={n-nc}")
    print(f"  {os.path.join(OUT_DIR, 'summary.csv')}")
    if xlsx_ok:
        print(f"  {os.path.join(OUT_DIR, 'summary.xlsx')}")


if __name__ == "__main__":
    main()
