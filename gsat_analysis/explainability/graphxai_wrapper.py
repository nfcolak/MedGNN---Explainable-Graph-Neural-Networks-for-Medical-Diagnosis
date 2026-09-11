"""GraphXAI-compatible wrapper for GSAT.

Sibling of protgnn_analysis/explainability/graphxai_wrapper.py (`ProtGNNWrapper`)
and graphcare_analysis/explainability/graphxai_wrapper.py
(`GraphCareGraphXAIWrapper`) — same contract, so GSAT is audited with the SAME
three GraphXAI explainers on the SAME test patients feeding the SAME summary
pipeline as the other two methods.

GraphXAI explainers call ``model(x, edge_index, **kwargs)`` and attribute over a
continuous node-feature tensor ``x``. GSAT already operates on continuous node
features (unlike GraphCare, whose real features are behind an embedding table),
so this wrapper is thin: it runs GSAT's DETERMINISTIC joint forward (no Gumbel
noise — ``GSAT.predict``) and returns logits.

Two node-importance sources for GSAT, both worth reporting (cf. GraphCare):
  (a) GSAT's own stochastic attention ``p_v`` — the inherent, faithful-by-
      construction explanation (``wrapper.builtin_node_importance``).
  (b) the post-hoc GraphXAI explainers run through this wrapper — for an
      apples-to-apples comparison against ProtGNN's post-hoc explanations.

FAITHFULNESS: ``verify()`` checks the wrapper's logits equal the underlying
GSAT's own eval-mode logits to numerical tolerance. Run it per graph before
trusting an explanation.
"""
import torch
import torch.nn as nn


class GSATGraphXAIWrapper(nn.Module):
    def __init__(self, gsat: nn.Module):
        super().__init__()
        self.gsat = gsat
        self._batch = None

    def set_context(self, x, edge_index, batch=None):
        """Fix the per-graph batch vector. Returns ``x`` unchanged (GSAT's node
        features are already the differentiable tensor the explainer wants), to
        mirror GraphCareGraphXAIWrapper.set_context's return contract."""
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        self._batch = batch
        return x

    def forward(self, x, edge_index, batch=None, **kwargs):
        if batch is None:
            batch = self._batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.gsat.predict(x, edge_index, batch)

    @torch.no_grad()
    def builtin_node_importance(self, x, edge_index, batch=None):
        """GSAT's inherent explanation: deterministic per-node attention p_v."""
        from torch_geometric.data import Data
        if batch is None:
            batch = self._batch if self._batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.gsat.node_importance(Data(x=x, edge_index=edge_index, batch=batch))

    @torch.no_grad()
    def verify(self, x, edge_index, batch=None, atol: float = 1e-5):
        """(is_faithful, max_abs_diff). GSAT.predict vs GSAT.forward(training=False)."""
        was_training = self.gsat.training
        self.gsat.eval()
        if batch is None:
            batch = self._batch if self._batch is not None else \
                torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        from torch_geometric.data import Data
        got = self.forward(x, edge_index, batch)
        ref = self.gsat(Data(x=x, edge_index=edge_index, batch=batch,
                             y=torch.zeros(int(batch.max()) + 1, dtype=torch.long,
                                           device=x.device)),
                        training=False)["logits"]
        if was_training:
            self.gsat.train()
        return bool(torch.allclose(got, ref, atol=atol)), (got - ref).abs().max().item()
