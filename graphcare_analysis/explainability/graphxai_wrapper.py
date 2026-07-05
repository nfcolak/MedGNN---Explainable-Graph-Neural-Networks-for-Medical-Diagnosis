
"""
GraphXAI-compatible wrapper for GraphCare (BAT-GNN).
=====================================================
Sibling of protgnn_analysis/explainability/graphxai_wrapper.py, so GraphCare
can be evaluated with the SAME GraphXAI explainers, on the SAME test patients,
feeding the SAME summarize_graphxai.py metric pipeline (sparsity / top-node
agreement / fidelity) as ProtGNN — an apples-to-apples explainability comparison.

WHY GRAPHCARE NEEDS A DIFFERENT WRAPPER THAN PROTGNN
----------------------------------------------------
GraphXAI explainers call ``model(x, edge_index, **kwargs)`` and attribute over a
continuous node-feature tensor ``x``. GraphCare's real ``forward`` instead takes
``(node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes)`` and looks the
node features up from an ``nn.Embedding`` table INTERNALLY. You cannot take a
gradient wrt an integer node id, so we cannot hand GraphXAI ``node_ids``.

This wrapper therefore:
  1. Exposes the post-embedding node features ``h0 = lin(node_emb(node_ids))``
     via ``node_features()`` — this is the differentiable tensor GraphXAI
     perturbs / integrates / masks.
  2. Re-runs GraphCare's forward FROM that tensor in ``forward(x, edge_index)``,
     reusing the trained sub-modules (``conv``, ``alpha_attn``, ``beta_attn``,
     ``node_emb``, ``lin``, ``MLP``) so no weights are re-learned or duplicated.
  3. Keeps the per-patient discrete context (``node_ids``, ``rel_ids``,
     ``batch``, ``visit_node``, ``ehr_nodes``) as FIXED inputs set once via
     ``set_context()`` before each explanation — GraphXAI never differentiates
     these, it only differentiates ``x``.

SCOPE OF THE EXPLANATION 
---------------------------------------------------
Under ``patient_mode="joint"`` (the trained config in run.py) the logits mix
two channels: ``x_graph`` (GNN message passing — graph-structured) and
``x_node`` (a plain average of the patient's own code embeddings — NOT graph
structured). Because ``x_node`` is recomputed from the embedding table and does
NOT depend on the injected ``x``, gradient- and perturbation-based explainers
attribute through the GRAPH pathway only. The returned logits are still the
model's true joint logits, so the predicted class is unchanged. Report GraphXAI
results as characterising GraphCare's graph-message-passing pathway; GraphCare's
own BAT attention (store_attn=True) covers the model's native interpretability.

FAITHFULNESS
------------
``verify()`` checks that ``forward(node_features(), edge_index)`` equals the
upstream ``model(...)`` logits to numerical tolerance. On the smoke-test
contract this matches to 0.0. RUN ``verify()`` ON A REAL TEST PATIENT before
trusting any explanation — it is the guard that this re-implemented forward has
not drifted from the trained function.

USAGE
-----
    from graphcare_.model import GraphCare
    from graphcare_analysis.explainability.graphxai_wrapper import GraphCareGraphXAIWrapper
    from graphxai.explainers import GradExplainer

    model = GraphCare(...); model.load_state_dict(ckpt); model.eval()
    wrapper = GraphCareGraphXAIWrapper(model)

    # per patient (batch of one graph):
    x0 = wrapper.set_context(node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes)
    ok, maxdiff = wrapper.verify()
    assert ok, f"wrapper diverged from model: {maxdiff}"

    explainer = GradExplainer(wrapper, criterion=torch.nn.CrossEntropyLoss())
    exp = explainer.get_explanation_graph(x=x0, edge_index=edge_index, label=pred)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class GraphCareGraphXAIWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self._ctx = None

    def set_context(self, node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes):
        """Fix the per-patient discrete inputs. Returns the node-feature tensor
        ``x`` (post-embedding, post-lin) to hand to the GraphXAI explainer."""
        self._ctx = dict(node_ids=node_ids, rel_ids=rel_ids, edge_index=edge_index,
                         batch=batch, visit_node=visit_node, ehr_nodes=ehr_nodes)
        return self.node_features()

    def node_features(self) -> torch.Tensor:
        """h0 = lin(node_emb(node_ids)) — the differentiable [N, hidden] input."""
        m, c = self.model, self._ctx
        with torch.no_grad():
            return m.lin(m.node_emb(c["node_ids"]).float())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run GraphCare's forward starting from injected node features ``x``.
        ``x``: [N, hidden].  Returns logits [B, C]."""
        m, c = self.model, self._ctx
        node_ids, rel_ids = c["node_ids"], c["rel_ids"]
        batch, visit_node, ehr_nodes = c["batch"], c["visit_node"], c["ehr_nodes"]

        edge_attr = m.lin(m.rel_emb(rel_ids).float())
        h = x
        for layer in range(1, m.layers + 1):
            if m.use_alpha:
                alpha = torch.softmax(m.alpha_attn[str(layer)](visit_node.float()), dim=1)
            if m.use_beta:
                beta = torch.tanh(m.beta_attn[str(layer)](visit_node.float())) * m.lambda_j
            if m.use_alpha and m.use_beta:
                attn = alpha * beta
            elif m.use_alpha:
                attn = alpha * torch.ones((batch.max().item() + 1, m.max_visit, 1)).to(edge_index.device)
            elif m.use_beta:
                attn = beta * torch.ones((batch.max().item() + 1, m.max_visit, m.num_nodes)).to(edge_index.device)
            else:
                attn = torch.ones((batch.max().item() + 1, m.max_visit, m.num_nodes)).to(edge_index.device)
            attn = torch.sum(attn, dim=1)
            xj_node_ids = node_ids[edge_index[0]]
            xj_batch = batch[edge_index[0]]
            attn = attn[xj_batch, xj_node_ids].reshape(-1, 1)

            if m.gnn == "BAT":
                h, _w_rel = m.conv[str(layer)](h, edge_index, edge_attr, attn=attn)
            else:
                h = m.conv[str(layer)](h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=0.5, training=self.training)

        if m.patient_mode in ("joint", "graph"):
            x_graph = global_mean_pool(h, batch)
            x_graph = F.dropout(x_graph, p=m.dropout, training=self.training)
        if m.patient_mode in ("joint", "node"):
            x_node = torch.stack([
                ehr_nodes[i].view(1, -1) @ m.node_emb.weight / torch.sum(ehr_nodes[i])
                for i in range(batch.max().item() + 1)
            ])
            x_node = m.lin(x_node).squeeze(1)
            x_node = F.dropout(x_node, p=m.dropout, training=self.training)

        if m.patient_mode == "joint":
            x_concat = torch.cat((x_graph, x_node), dim=1)
            x_concat = F.dropout(x_concat, p=m.dropout, training=self.training)
            return m.MLP(x_concat)
        elif m.patient_mode == "graph":
            return m.MLP(x_graph)
        else:
            return m.MLP(x_node)

    @torch.no_grad()
    def verify(self, atol: float = 1e-5):
        """Assert forward-from-x reproduces the upstream model's own logits.
        Returns (is_faithful: bool, max_abs_diff: float). Call per patient."""
        was_training = self.training
        self.model.eval(); self.eval()
        c = self._ctx
        got = self.forward(self.node_features(), c["edge_index"])
        ref = self.model(c["node_ids"], c["rel_ids"], c["edge_index"],
                         c["batch"], c["visit_node"], c["ehr_nodes"])
        if was_training:
            self.train()
        return bool(torch.allclose(got, ref, atol=atol)), (got - ref).abs().max().item()