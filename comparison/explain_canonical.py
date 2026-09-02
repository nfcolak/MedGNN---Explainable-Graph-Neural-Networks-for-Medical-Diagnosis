"""GraphXAI explanations for the canonical ProtGNN checkpoint.

protgnn_analysis/train.py explains the test split built by its own dataloader,
which is not the canonical fold. This script runs the identical three
explainers, with the same record schema, over the first N graphs of the
CANONICAL test fold, using the checkpoint trained by
comparison/protgnn_canonical.py. The output folder can be handed to
scripts/summarize_graphxai.py --explanations_dir to get the decoded clinical
factors, sparsity and inter-explainer agreement.

Run:  PYTHONPATH=. python3 -u comparison/explain_canonical.py --n 50
Out:  comparison/protgnn_canonical/explanations/graph_*.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from protgnn_analysis.config import data_args, model_args
from protgnn_analysis.models import GnnNets
from protgnn_analysis.load_dataset import get_dataset
from protgnn_analysis.train import (ProtGNNWrapper, _prototype_record, _top_items,
                                    _feature_record, _grad_feature_attribution,
                                    _integrated_grad_feature_attribution,
                                    NODE_TOP_K, FEATURE_TOP_K)
from graphxai.explainers.grad import GradExplainer
from graphxai.explainers.integrated_grad import IntegratedGradExplainer
from graphxai.explainers.gnn_explainer import GNNExplainer

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "comparison" / "protgnn_canonical"


def canonical_test_indices(n_graphs):
    split = json.load(open(REPO / "comparison" / "canonical_split.json"))
    df = pd.read_csv(REPO / "data" / "merged_ed.csv", low_memory=False,
                     usecols=["subject_id"])
    assert len(df) == n_graphs, (len(df), n_graphs)
    f = df["subject_id"].astype(str).map(split["fold"])
    return [i for i, v in enumerate(f.values) if not pd.isna(v) and int(v) == 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    data_args.dataset_name = "mimic_intra_patient_disease_cooccur"
    data_args.graph_structure = "cooccur"
    model_args.enable_prot = True
    # GNNExplainer moves weights to CPU internally, which breaks a following
    # MPS forward pass; train.py explains on CPU for the same reason.
    device = torch.device("cpu")
    model_args.device = device

    dataset = get_dataset(data_args.dataset_dir, data_args.dataset_name,
                          task=data_args.task, graph_structure="cooccur")
    test_idx = canonical_test_indices(len(dataset))
    n_classes = int(dataset.num_classes)

    model = GnnNets(dataset.num_node_features, n_classes, model_args)
    ckpt = torch.load(RUN_DIR / "gcn_latest.pth", map_location=device)
    model.update_state_dict(ckpt["net"])
    model.to(device)
    model.device = device
    if hasattr(model.model, "device"):
        model.model.device = device
    model.eval()
    print(f"  checkpoint epoch {ckpt['epoch']} | test graphs {len(test_idx)} | "
          f"explaining {min(args.n, len(test_idx))}", flush=True)

    wrapper = ProtGNNWrapper(model)
    wrapper.eval()
    criterion = nn.CrossEntropyLoss()
    grad_exp = GradExplainer(wrapper, criterion=criterion)
    integ_exp = IntegratedGradExplainer(wrapper, criterion=criterion)
    gnn_exp = GNNExplainer(wrapper)

    feature_names = getattr(dataset, "feature_cols", [])
    out = RUN_DIR / "explanations"
    out.mkdir(parents=True, exist_ok=True)

    for k, di in enumerate(test_idx[:args.n]):
        d = dataset[di]
        x, edge_index = d.x.to(device), d.edge_index.to(device)
        label = int(d.y.view(-1)[0].item())
        null_batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        fwd = {"batch": null_batch}

        with torch.no_grad():
            pred = int(wrapper(x, edge_index, null_batch).argmax(dim=-1).item())
        explain_label = torch.tensor([pred], dtype=torch.long, device=device)

        batch = d.clone()
        batch.batch = null_batch
        rec = {
            "graph_idx": k,
            "dataset_index": int(di),
            "patient_index": int(di),
            "true_label": label,
            "pred_label": pred,
            "correct": bool(pred == label),
            "num_nodes": int(x.size(0)),
            "num_edges": int(edge_index.size(1)),
            "feature_names": feature_names,
            "prototype_evidence": _prototype_record(model, batch, n_classes, pred_label=pred),
            "explanations": {},
        }

        for name, fn in (
            ("GradExplainer", lambda: (grad_exp.get_explanation_graph(
                x=x, edge_index=edge_index, label=explain_label, forward_kwargs=fwd),
                _grad_feature_attribution(wrapper, x, edge_index, explain_label, fwd, criterion), True)),
            ("IntegratedGradExplainer", lambda: (integ_exp.get_explanation_graph(
                x=x, edge_index=edge_index, label=explain_label, forward_kwargs=fwd),
                _integrated_grad_feature_attribution(wrapper, x, edge_index, explain_label, fwd, criterion), True)),
            ("GNNExplainer", lambda: (gnn_exp.get_explanation_graph(
                x=x, edge_index=edge_index, forward_kwargs=fwd), None, False)),
        ):
            try:
                exp, feat_attr, use_abs = fn()
                imp = exp.node_imp.detach().cpu().numpy()
                entry = {
                    "node_importance": imp.tolist(),
                    "top_nodes": _top_items(imp, k=NODE_TOP_K, use_abs=use_abs),
                    "min": float(imp.min()), "max": float(imp.max()),
                    "mean": float(imp.mean()), "std": float(imp.std()),
                }
                if feat_attr is not None:
                    entry["feature_importance"] = _feature_record(feat_attr, feature_names)
                else:
                    fi = exp.feature_imp.detach().cpu().numpy()
                    entry["feature_importance"] = {
                        "feature_scores": fi.tolist(),
                        "top_features": _top_items(fi, names=feature_names,
                                                   k=FEATURE_TOP_K, use_abs=False),
                    }
                rec["explanations"][name] = entry
            except Exception as e:
                rec["explanations"][name] = {"error": str(e)}

        json.dump(rec, open(out / f"graph_{k}.json", "w"))
        print(f"  [{'CORRECT' if rec['correct'] else 'WRONG  '}] graph {k:3d} "
              f"(dataset {di}) true={label} pred={pred} nodes={rec['num_nodes']}", flush=True)

    print("  saved ->", out, flush=True)


if __name__ == "__main__":
    main()
