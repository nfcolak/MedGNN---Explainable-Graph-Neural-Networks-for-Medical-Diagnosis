"""Smoke test: GraphCare's BAT-GNN runs on OUR tensor format (model-direct Path B).

GraphCare's own data pipeline (data_prepare.py) is hardwired to the authors'
absolute paths and CCSCM/CCSPROC/ATC3 per-code KG files, so we do NOT use it.
Instead we drive the upstream `GraphCare` model directly with tensors we build
ourselves from merged_ed.csv + our KG. This file proves that contract works and
documents the exact shapes the adapter (Phase 1) must produce.

Run (inside the GraphCare venv):
    PYTHONPATH=.:external/GraphCare .venv-graphcare/bin/python3 \
        graphcare_analysis/test_model_smoke.py
"""
import torch
import torch.nn.functional as F
from graphcare_.model import GraphCare


def main():
    # global KG: num_nodes nodes, num_rels relation types; single-visit ED cohort
    num_nodes, num_rels, max_visit, num_classes = 10, 3, 1, 30
    model = GraphCare(num_nodes=num_nodes, num_rels=num_rels, max_visit=max_visit,
                      embedding_dim=16, hidden_dim=16, out_channels=num_classes,
                      layers=2, patient_mode="joint", gnn="BAT")

    # ---- the exact contract the adapter must emit (here: 2 dummy patients) ----
    # patient 0 personalized-graph nodes = global ids [0,1,2]; patient 1 = [3,4,5]
    node_ids   = torch.tensor([0, 1, 2, 3, 4, 5])      # [N] global node id per batched node
    batch      = torch.tensor([0, 0, 0, 1, 1, 1])      # [N] graph id per node (PyG batch)
    edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]])  # [2,E] LOCAL node indices
    rel_ids    = torch.tensor([0, 1, 0])               # [E] relation id per edge
    visit_node = torch.zeros(2, max_visit, num_nodes)  # [graphs, max_visit, num_nodes]
    visit_node[0, 0, [0, 1, 2]] = 1.0
    visit_node[1, 0, [3, 4, 5]] = 1.0
    ehr_nodes  = torch.zeros(2, num_nodes)             # [graphs, num_nodes] EHR-code indicator
    ehr_nodes[0, [0, 1]] = 1.0
    ehr_nodes[1, [3]]    = 1.0

    model.eval()
    with torch.no_grad():
        logits = model(node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes)
    assert logits.shape == (2, num_classes), logits.shape

    model.train()
    loss = F.cross_entropy(
        model(node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes),
        torch.tensor([5, 12]))
    loss.backward()
    print(f"OK: logits {tuple(logits.shape)}, loss {loss.item():.3f}, backward ran.")


if __name__ == "__main__":
    main()
