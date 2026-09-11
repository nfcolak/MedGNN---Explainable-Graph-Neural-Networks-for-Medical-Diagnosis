"""
explain_gsat.py  —  explanations for a trained GSAT model.
=========================================================
Sibling of protgnn_analysis/train.py::explain_test_set and
graphcare_analysis/explainability/explain_graphcare.py. For each test graph it
records, in the schema summarize_graphxai.py / summarize_gsat.py read:

  - "BuiltinAttention" : GSAT's OWN inherent explanation — the deterministic
    per-node stochastic-attention probability p_v (faithful by construction).
  - "GradExplainer" / "IntegratedGradExplainer" / "GNNExplainer" : the SAME
    three GraphXAI post-hoc explainers ProtGNN and GraphCare use, run through
    GSATGraphXAIWrapper — for an apples-to-apples post-hoc comparison.

Each explainer's node importance also gets sparsity + Fidelity+/Fidelity-
(shared/lib/fidelity.py — the node-level metric already used by both other
methods; the edge-level variant faithful to GSAT's native granularity is a
later task, see docs/PROJECT_CONTEXT.md).

Run (inside the ProtGNN venv, with GraphXAI on the path):
    PYTHONPATH=.:external/GraphXAI-main .venv-protgnn/bin/python3 \
        gsat_analysis/explainability/explain_gsat.py \
        --ckpt gsat_analysis/outputs/star/gsat_model.pt \
        --graph_structure star \
        --canonical_split comparison/canonical_split.json \
        --max_graphs 50
"""
import argparse
import inspect
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_THIS, os.pardir, os.pardir))
for _p in (_ROOT, os.path.join(_ROOT, "external", "GraphXAI-main")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from torch_geometric.loader import DataLoader

from shared.lib.config_base import set_seed
from shared.lib.graph_structures import resolve as resolve_structure
from shared.lib.fidelity import fidelity_plus, fidelity_minus, sparsity
from protgnn_analysis.load_dataset import get_dataset, get_dataloader
from protgnn_analysis.scripts.summarize_all_test import build_layout, decode_node
from gsat_analysis.config import cfg
from gsat_analysis.models import GIN, GSAT, ExtractorMLP
from gsat_analysis.explainability.graphxai_wrapper import GSATGraphXAIWrapper

POSTHOC = ["GradExplainer", "IntegratedGradExplainer", "GNNExplainer"]
EXPLAINERS = ["BuiltinAttention"] + POSTHOC


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    c = ck["config"]
    clf = GIN(ck["x_dim"], ck["num_classes"], hidden_dim=c["hidden_dim"],
              num_layers=c["num_layers"], dropout=c["dropout"], readout=c["readout"])
    extractor = ExtractorMLP(c["hidden_dim"], attention_level=c["attention_level"])
    gsat = GSAT(clf, extractor, attention_level=c["attention_level"],
                temperature=c["temperature"], info_loss_coef=c["info_loss_coef"],
                init_r=c["init_r"], final_r=c["final_r"],
                decay_interval=c["decay_interval"], decay_r=c["decay_r"])
    gsat.load_state_dict(ck["state_dict"])
    gsat.to(device).eval()
    return gsat, ck["structure"]


def build_explainers(wrapper):
    """Same constructors as explain_graphcare.build_explainers / ProtGNN."""
    from graphxai.explainers import GradExplainer, IntegratedGradExplainer, GNNExplainer
    ce = nn.CrossEntropyLoss()
    out = {}
    for name, ctor in [
        ("GradExplainer", lambda: GradExplainer(wrapper, criterion=ce)),
        ("IntegratedGradExplainer", lambda: IntegratedGradExplainer(wrapper, criterion=ce)),
        ("GNNExplainer", lambda: GNNExplainer(wrapper)),
    ]:
        try:
            out[name] = ctor()
        except Exception as e:
            print(f"  [ctor error] {name}: {type(e).__name__}: {e}")
    return out


def explain_one(explainer, x, edge_index, pred, fwd_kwargs):
    fn = explainer.get_explanation_graph
    kwargs = dict(x=x, edge_index=edge_index, forward_kwargs=fwd_kwargs)
    if "label" in inspect.signature(fn).parameters:
        kwargs["label"] = torch.tensor([int(pred)], device=x.device)
    return fn(**kwargs)


def _importance(exp, n_nodes):
    imp = exp.node_imp if hasattr(exp, "node_imp") else exp
    if torch.is_tensor(imp):
        imp = imp.detach().cpu().numpy()
    imp = np.asarray(imp, dtype=float)
    if imp.ndim > 1:
        imp = np.abs(imp).sum(axis=tuple(range(1, imp.ndim)))
    if imp.shape[0] != n_nodes:
        raise ValueError(f"importance length {imp.shape[0]} != num_nodes {n_nodes}")
    return imp


def _record_importance(rec_expl, name, imp, wrapper, x, edge_index, y, batch, top_k, agg):
    order = list(np.argsort(np.abs(imp))[::-1][:top_k])
    fp = fidelity_plus(wrapper, x, edge_index, imp, y, batch)
    fm = fidelity_minus(wrapper, x, edge_index, imp, y, batch)
    rec_expl[name] = {
        "node_importance": [float(v) for v in imp],
        "top_nodes": [{"index": int(i), "importance": float(imp[i])} for i in order],
        "sparsity": sparsity(imp),
        "fidelity_plus": fp,
        "fidelity_minus": fm,
    }
    agg["sp"][name].append(sparsity(imp))
    agg["fp"][name].append(fp["prob"])
    agg["fm"][name].append(fm["prob"])


def _load_match_set(path):
    """Accept a JSON list of ids, a {'subjects': [...]} dict, or newline ids.
    Mirrors graphcare_analysis/explainability/explain_graphcare.py::_load_match_set
    so the patient-matched comparison is driven identically across methods."""
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--graph_structure", "--graph", default=None)
    ap.add_argument("--canonical_split", default=None, help="comparison/canonical_split.json")
    ap.add_argument("--max_graphs", type=int, default=50)
    ap.add_argument("--match_subjects", default=None,
                    help="JSON list of subject_ids (e.g. comparison/protgnn_explained_subjects.json). "
                         "Explains ONLY these patients, for a patient-matched comparison across "
                         "methods. Best combined with --canonical_split.")
    ap.add_argument("--top_nodes", type=int, default=4)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    set_seed(cfg.seed)
    device = torch.device("cpu")            # GraphXAI explainers: CPU only
    if args.canonical_split:
        os.environ["CANONICAL_SPLIT_JSON"] = args.canonical_split

    gsat, ckpt_structure = load_model(args.ckpt, device)
    structure = resolve_structure(args.graph_structure or ckpt_structure, "gsat")
    if structure != ckpt_structure:
        print(f"  [warn] --graph_structure {structure} != checkpoint's {ckpt_structure}")

    dataset = get_dataset(str(cfg.data_dir), cfg.dataset_name, graph_structure=structure)
    loaders = get_dataloader(dataset, batch_size=1,
                             data_split_ratio=list(cfg.split_ratio), seed=cfg.seed)
    test_loader = DataLoader(loaders["test"].dataset, batch_size=1, shuffle=False)
    layout = build_layout(dataset)

    # class-id -> disease name. protgnn load_dataset stores this as
    # dataset.label_mapping = {"0": "Acute kidney failure", ...}, built from
    # sorted(disease_1.unique()) — the same ordering comparison/build_split.py
    # uses, so ids match across methods.
    _lm = getattr(dataset, "label_mapping", {}) or {}
    id_to_class = {int(k): v for k, v in _lm.items()}

    match_set = None
    if args.match_subjects:
        match_set = _load_match_set(args.match_subjects)
        print(f"  patient matching ON: restricting to {len(match_set)} subject_ids")
        if not args.canonical_split:
            print("  [warn] --match_subjects without --canonical_split: the test fold "
                  "is seed-derived, so matched patients may not all be in it.")

    wrapper = GSATGraphXAIWrapper(gsat).to(device)
    explainers = build_explainers(wrapper)
    if not explainers:
        sys.exit("No explainers constructed — see [ctor error] above.")

    out_dir = args.out_dir or os.path.join(str(cfg.OUTPUTS_DIR), structure,
                                           "results", "explanations")
    os.makedirs(out_dir, exist_ok=True)

    agg = {k: {e: [] for e in EXPLAINERS} for k in ("sp", "fp", "fm")}
    n_correct = total = 0
    faith_fail = 0

    for gi, batch in enumerate(test_loader):
        if total >= args.max_graphs:
            break
        subject_id = int(batch.subject_id.view(-1)[0]) if hasattr(batch, "subject_id") else None
        if match_set is not None and subject_id not in match_set:
            continue                                    # skip unmatched patients
        x = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        y = int(batch.y.view(-1)[0])
        null_batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        fwd_kwargs = {"batch": null_batch}

        wrapper.set_context(x, edge_index, null_batch)
        ok, diff = wrapper.verify(x, edge_index, null_batch)
        if not ok:
            faith_fail += 1
            print(f"  [FAITHFULNESS WARN] graph {gi}: max|diff|={diff:.2e}")

        with torch.no_grad():
            pred = int(wrapper(x, edge_index, null_batch).argmax(1).item())
        total += 1
        n_correct += int(pred == y)

        N = int(x.size(0))
        rec = {
            "graph_idx": gi,
            "dataset_index": int(batch.dataset_index.view(-1)[0]) if hasattr(batch, "dataset_index") else None,
            "subject_id": subject_id,
            "pred_label": pred, "true_label": y, "correct": bool(pred == y),
            "pred_class": id_to_class.get(pred, str(pred)),
            "true_class": id_to_class.get(y, str(y)),
            "num_nodes": N, "num_edges": int(edge_index.size(1)),
            "node_tokens": [decode_node(x[i].cpu().numpy(), layout) for i in range(N)],
            "explanations": {},
        }

        # (a) GSAT's inherent explanation
        try:
            imp = wrapper.builtin_node_importance(x, edge_index, null_batch).cpu().numpy()
            _record_importance(rec["explanations"], "BuiltinAttention", imp,
                               wrapper, x, edge_index, y, null_batch, args.top_nodes, agg)
        except Exception as e:
            rec["explanations"]["BuiltinAttention"] = {"error": f"{type(e).__name__}: {e}"}

        # (b) post-hoc GraphXAI explainers
        for name, ex in explainers.items():
            try:
                exp = explain_one(ex, x, edge_index, pred, fwd_kwargs)
                imp = _importance(exp, N)
                _record_importance(rec["explanations"], name, imp,
                                   wrapper, x, edge_index, y, null_batch, args.top_nodes, agg)
            except Exception as e:
                rec["explanations"][name] = {"error": f"{type(e).__name__}: {e}"}

        with open(os.path.join(out_dir, f"graph_{gi}.json"), "w") as f:
            json.dump(rec, f, indent=2)

        tags = " ".join(
            f"{e.split('Explainer')[0][:4] if e != 'BuiltinAttention' else 'attn'}"
            f"={rec['explanations'][e].get('fidelity_plus', {}).get('prob', float('nan')):+.3f}"
            for e in EXPLAINERS if e in rec["explanations"] and "error" not in rec["explanations"][e]
        )
        print(f"  graph {gi:4d} | {'OK ' if pred == y else 'ERR'} true={y:2d} pred={pred:2d} "
              f"N={N:3d} | fid+ {tags}")

    print(f"\nDONE. explained {total} graphs -> {out_dir}")
    print(f"  accuracy on explained set : {n_correct}/{total}")
    if faith_fail:
        print(f"  [WARN] wrapper faithfulness failed on {faith_fail}/{total} graphs")
    print("  mean sparsity / fidelity+ / fidelity- per explainer:")
    for e in EXPLAINERS:
        sp, fp, fm = agg["sp"][e], agg["fp"][e], agg["fm"][e]
        if sp:
            print(f"    {e:<26}: sparsity={np.mean(sp):.4f}  "
                  f"fidelity+={np.mean(fp):+.4f}  fidelity-={np.mean(fm):+.4f}")
        else:
            print(f"    {e:<26}: (no successful runs)")


if __name__ == "__main__":
    main()
