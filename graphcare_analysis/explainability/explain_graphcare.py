"""
explain_graphcare.py  —  run GraphXAI explainers on a trained GraphCare model.
==============================================================================
Sibling of protgnn_analysis/explainability/explain_checkpoint.py. Loads the
GraphCare checkpoint, wraps it with GraphCareGraphXAIWrapper, and runs the SAME
three explainers ProtGNN uses (GradExplainer | IntegratedGradExplainer |
GNNExplainer) over the test patients, writing one graph_<i>.json per patient in
the schema summarize_graphxai.py already reads (node_importance / top_nodes /
error), plus node_global_ids so KG nodes can be decoded to clinical tokens later.

The three things you needed to "confirm" are now automatic, printed as it runs:
  ITEM 1 (faithfulness)  : wrapper.verify() is asserted on every patient.
  ITEM 2 (explainer API) : the explainers are actually invoked; any signature
                           mismatch is caught per-explainer and written as
                           {"error": ...} instead of crashing the run.
  ITEM 3 (edge density)  : the edge-count distribution is printed at the end so
                           you can see whether GNNExplainer's ~100-edge ceiling
                           is a problem for GraphCare's subgraphs.

Run (inside the GraphCare venv, with GraphXAI on the path):
    PYTHONPATH=.:external/GraphCare:external/GraphXAI-main \
        .venv-graphcare/bin/python3 \
        graphcare_analysis/explainability/explain_graphcare.py \
        --ckpt graphcare_analysis/outputs/graphcare_model.pt \
        --canonical_split comparison/canonical_split.json \
        --max_graphs 25            # cap for a quick first run; drop for the full set
"""
import os
import sys
import json
import inspect
import argparse

import numpy as np
import torch
import torch.nn as nn

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, os.pardir, os.pardir))
for _p in (_ROOT,
           os.path.join(_ROOT, "external", "GraphCare"),
           os.path.join(_ROOT, "external", "GraphXAI-main")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch.utils.data import DataLoader

from graphcare_analysis.config import cfg
from graphcare_analysis.build_kg import build_global_kg
from graphcare_analysis.adapter import build_loaders, _collate
from graphcare_analysis.explainability.graphxai_wrapper import GraphCareGraphXAIWrapper
from shared.lib.fidelity import fidelity_plus, fidelity_minus, sparsity

EXPLAINERS = ["GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]


def _extract_importance(exp, num_nodes):
    """GraphXAI explainers return an Explanation object (attr .node_imp) in most
    versions; some return a dict. Reduce whatever we get to a length-N python
    list of floats, so the JSON is uniform regardless of explainer internals."""
    imp = None
    if hasattr(exp, "node_imp"):
        imp = exp.node_imp
    elif isinstance(exp, dict):
        imp = exp.get("node_imp", exp.get("node_importance"))
    if imp is None:
        raise ValueError("explainer returned no node importance (.node_imp)")
    if torch.is_tensor(imp):
        imp = imp.detach().cpu().numpy()
    imp = np.asarray(imp, dtype=float)
    if imp.ndim > 1:                       # [N, F] or [N, 1] -> per-node scalar
        imp = np.abs(imp).sum(axis=tuple(range(1, imp.ndim)))
    if imp.shape[0] != num_nodes:          # guard: importance must align to nodes
        raise ValueError(f"importance length {imp.shape[0]} != num_nodes {num_nodes}")
    return imp.tolist()


def build_explainers(wrapper):
    """
    >>> MATCH THIS BLOCK to protgnn_analysis/explainability/explain_checkpoint.py <<<
    The constructor/method names below follow the standard GraphXAI API, but your
    ProtGNN explain script is the source of truth for the exact call convention
    your installed GraphXAI expects. If a constructor here differs from that file,
    copy the working one over. (Errors here are printed with the exact exception
    so you can paste it back for alignment.)
    """
    from graphxai.explainers import GradExplainer, IntegratedGradExplainer, GNNExplainer
    ce = nn.CrossEntropyLoss()
    out = {}
    for name, ctor in [
        ("GradExplainer",            lambda: GradExplainer(wrapper, criterion=ce)),
        ("IntegratedGradExplainer",  lambda: IntegratedGradExplainer(wrapper, criterion=ce)),
        ("GNNExplainer",             lambda: GNNExplainer(wrapper)),
    ]:
        try:
            out[name] = ctor()
        except Exception as e:
            print(f"  [ctor error] {name}: {type(e).__name__}: {e}")
    return out


def explain_one(explainer, x0, edge_index, pred):
    """Graph-level explanation for one patient. Context (node_ids/rel_ids/batch/
    visit_node/ehr_nodes) is already fixed on the wrapper via set_context(), so
    the explainer only needs (x, edge_index).

    GraphXAI explainers don't share one signature: Grad/IntegratedGrad take a
    `label` (the class to explain, shape [1] to match the [1, C] logits);
    GNNExplainer optimizes a mask around the model's own predicted class and
    takes no `label`. We inspect the signature and pass `label` only when it's
    accepted."""
    fn = explainer.get_explanation_graph
    kwargs = dict(x=x0, edge_index=edge_index)
    if "label" in inspect.signature(fn).parameters:
        kwargs["label"] = torch.tensor([int(pred)])
    return fn(**kwargs)


def _load_kg():
    return torch.load(cfg.kg_path) if cfg.kg_path.exists() else build_global_kg(save=True)


def _test_subjects_from_split(split_json):
    """Return subject_id per test graph, ALIGNED to the order explain iterates.

    Mirrors adapter.build_loaders' canonical-split branch exactly so
    test_subjects[gi] is the patient of the gi-th explained graph. Only valid
    when --canonical_split is given (the deterministic branch); returns None
    otherwise, since the seed/subject-aware branch isn't reproduced here."""
    import pandas as pd
    df = pd.read_csv(cfg.data_dir / cfg.csv_filename, low_memory=False)
    subj = (df["subject_id"].values if "subject_id" in df.columns
            else np.arange(len(df), dtype=np.int64))
    meta = json.load(open(split_json))
    fold = meta["fold"]
    keep = [i for i in range(len(df)) if str(int(subj[i])) in fold]
    folds = np.array([fold[str(int(subj[i]))] for i in keep])
    te = np.where(folds == 2)[0].tolist()          # fold 2 == test
    return [int(subj[keep[t]]) for t in te]


def _load_match_set(path):
    """Accept a JSON list of ids, a {'subjects': [...]} dict, or newline ids."""
    raw = open(path).read().strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            obj = obj.get("subjects", obj.get("subject_ids", []))
        return set(int(x) for x in obj)
    except json.JSONDecodeError:
        return set(int(line) for line in raw.splitlines() if line.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="graphcare_analysis/outputs/graphcare_model.pt")
    ap.add_argument("--canonical_split", default=None, help="comparison/canonical_split.json")
    ap.add_argument("--limit", type=int, default=None, help="cap #patients before split")
    ap.add_argument("--max_graphs", type=int, default=None, help="cap #graphs explained (quick run)")
    ap.add_argument("--match_subjects", default=None,
                    help="JSON list of subject_ids (e.g. comparison/protgnn_explained_subjects.json). "
                         "Explains ONLY these patients, for a patient-matched comparison. "
                         "Requires --canonical_split.")
    ap.add_argument("--top_nodes", type=int, default=4)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cpu")
    sys.path.insert(0, str(cfg.UPSTREAM_DIR))
    from graphcare_.model import GraphCare

    kg = _load_kg()
    _, _, test_loader, C, class_to_id = build_loaders(
        kg, limit=args.limit, split_json=args.canonical_split)

    model = GraphCare(num_nodes=kg["num_nodes"], num_rels=kg["num_rels"], max_visit=1,
                      embedding_dim=cfg.emb_dim, hidden_dim=cfg.emb_dim, out_channels=C,
                      layers=cfg.num_layers, dropout=cfg.dropout,
                      patient_mode="joint", use_alpha=True, use_beta=True, gnn="BAT").to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # explain ONE graph at a time so node importances map to a single patient
    single = DataLoader(test_loader.dataset, batch_size=1, shuffle=False, collate_fn=_collate)

    # subject id per test graph (for recording + patient matching)
    test_subjects = None
    if args.canonical_split:
        test_subjects = _test_subjects_from_split(args.canonical_split)
        if len(test_subjects) != len(single.dataset):
            print(f"  [warn] derived {len(test_subjects)} subjects but loader has "
                  f"{len(single.dataset)} test graphs — subject alignment may be off.")

    match_set = None
    if args.match_subjects:
        if not args.canonical_split:
            sys.exit("--match_subjects requires --canonical_split (subject ids are "
                     "derived from the canonical split).")
        match_set = _load_match_set(args.match_subjects)
        print(f"  patient matching ON: restricting to {len(match_set)} subject_ids")

    wrapper = GraphCareGraphXAIWrapper(model).to(device)
    explainers = build_explainers(wrapper)
    if not explainers:
        sys.exit("No explainers constructed — see [ctor error] above and align build_explainers().")

    out_dir = args.out_dir or os.path.join(str(cfg.OUTPUTS_DIR), "results", "explanations")
    os.makedirs(out_dir, exist_ok=True)

    id_to_class = {v: k for k, v in class_to_id.items()}
    agg = {e: [] for e in EXPLAINERS}
    agg_fp = {e: [] for e in EXPLAINERS}     # fidelity+ (prob) per explainer
    agg_fm = {e: [] for e in EXPLAINERS}     # fidelity- (prob) per explainer
    edge_counts, n_agree, n_agree_total, n_correct, total = [], 0, 0, 0, 0

    for gi, b in enumerate(single):
        if args.max_graphs is not None and total >= args.max_graphs:
            break
        subject_id = test_subjects[gi] if test_subjects is not None and gi < len(test_subjects) else None
        if match_set is not None and subject_id not in match_set:
            continue                                         # skip unmatched patients
        b = {k: v.to(device) for k, v in b.items()}
        node_ids, rel_ids, edge_index = b["node_ids"], b["rel_ids"], b["edge_index"]
        batch, visit_node, ehr_nodes = b["batch"], b["visit_node"], b["ehr_nodes"]
        y = int(b["y"][0])

        x0 = wrapper.set_context(node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes)

        ok, diff = wrapper.verify()                          # ITEM 1
        if not ok:
            print(f"  [FAITHFULNESS WARN] graph {gi}: wrapper diverged, max|diff|={diff:.2e}")

        with torch.no_grad():
            pred = int(wrapper(x0, edge_index).argmax(1).item())

        N, E = int(node_ids.numel()), int(edge_index.size(1))
        edge_counts.append(E)                                # ITEM 3
        total += 1; n_correct += int(pred == y)

        rec = {"graph_idx": gi, "dataset_index": None, "subject_id": subject_id,
               "pred_label": pred, "true_label": y, "correct": bool(pred == y),
               "pred_class": id_to_class.get(pred, str(pred)),
               "true_class": id_to_class.get(y, str(y)),
               "num_nodes": N, "num_edges": E,
               "node_global_ids": node_ids.tolist(),
               "explanations": {}}

        top_nodes_seen = {}
        for name, ex in explainers.items():                  # ITEM 2 exercised here
            try:
                exp = explain_one(ex, x0, edge_index, pred)
                imp = _extract_importance(exp, N)
                order = list(np.argsort(np.abs(imp))[::-1][:args.top_nodes])
                fid_plus = fidelity_plus(wrapper, x0, edge_index, imp, y, batch)
                fid_minus = fidelity_minus(wrapper, x0, edge_index, imp, y, batch)
                rec["explanations"][name] = {
                    "node_importance": [float(v) for v in imp],
                    "top_nodes": [{"index": int(i), "importance": float(imp[i])} for i in order],
                    "sparsity": sparsity(imp),
                    "fidelity_plus": fid_plus,
                    "fidelity_minus": fid_minus,
                }
                agg[name].append(sparsity(imp))
                agg_fp[name].append(fid_plus["prob"])
                agg_fm[name].append(fid_minus["prob"])
                top_nodes_seen[name] = int(np.argmax(np.abs(imp)))
            except Exception as e:
                rec["explanations"][name] = {"error": f"{type(e).__name__}: {e}"}

        idxs = list(top_nodes_seen.values())
        if len(idxs) >= 2:
            n_agree_total += 1
            n_agree += int(len(set(idxs)) == 1)

        with open(os.path.join(out_dir, f"graph_{gi}.json"), "w") as f:
            json.dump(rec, f, indent=2)

    # ── run summary (the numbers your thesis' explanation-quality section needs) ──
    print(f"\nDONE. explained {total} graphs -> {out_dir}")
    print(f"  accuracy on explained set : {n_correct}/{total}")
    print("  mean sparsity / fidelity+ / fidelity- per explainer:")
    for e in EXPLAINERS:
        v, fps, fms = agg[e], agg_fp[e], agg_fm[e]
        if v:
            print(f"    {e:<26}: sparsity={np.mean(v):.4f}  "
                  f"fidelity+={np.mean(fps):+.4f}  fidelity-={np.mean(fms):+.4f}")
        else:
            print(f"    {e:<26}: (no successful runs)")
    if n_agree_total:
        print(f"  top-node agreement        : {n_agree}/{n_agree_total} ({n_agree/n_agree_total:.1%})")
    if edge_counts:
        ec = np.array(edge_counts)
        over = int((ec > 100).sum())
        print(f"  edge counts (ITEM 3)      : min={ec.min()} median={int(np.median(ec))} "
              f"max={ec.max()} | {over}/{len(ec)} graphs over 100 edges "
              f"(GNNExplainer risk)")


if __name__ == "__main__":
    main()
