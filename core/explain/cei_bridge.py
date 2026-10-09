"""CEI-GNN v3 under the clinical_graph_v2 explanation runner.

Built ON TOP of Necati's own explanation code (cei/studies/cei_v3_graphxai.py); nothing
under cei/ is edited and nothing of his is reimplemented:

- the model-facing wrapper is his `CEIV3GraphXAIWrapper` (a private copy of the network,
  the original PLE term cached and held fixed, edge identity checked on every call);
- the built-in explanation is his `native_importance`, the exact additive accounting of the
  logit (node + half of each incident edge/pair contribution), with bias and absence
  evidence exported separately rather than pushed onto nodes.

What this class adds is only the runner's interface: `set_context(batch)`,
`forward(x, edge_index, batch=None, ...)`, `verify`, `builtin_node_importance`,
`builtin_detail`, `ig_feature_spec`.

CEI has no message passing, so GNNExplainer reaches it through the edge aggregator it
already routes through PyG's mask state (read in cei_gnn_v3._evidence_blocks). Like
`ClinicalGNNGraphXAIWrapper`, `forward` therefore declares no `edge_mask` parameter: the
explainer has exactly one delivery path, and a removal mask from the fidelity code goes
through the same hook for that one call.

Departures from the contract the other models get, recorded rather than hidden:

- The explained input is CEI's continuous row (numeric features | token vectors |
  node-type vectors). Zeroing a row therefore also zeroes its token and type vectors, so
  `token_override` is ignored (there is no separate token to replace).
- Removal masks the node's incident EDGES, but CEI's within-visit PAIR contributions and
  its absence block keep seeing the node (the absence inputs are fixed, as in his study).
  Fidelity for CEI is therefore a partial removal and not strictly comparable with the
  other models'; see `intervention_scope`.
- The relation-vs-payload ablation cannot be applied (its edge metadata is fixed inside
  the bound wrapper); the runner records that as not applicable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

INTERVENTION_SCOPE = (
    "continuous node row zeroed (numeric + token + type vectors) and incident edges masked; "
    "within-visit pair contributions and the absence block still see the node. Partial "
    "removal: not strictly comparable with the other models' fidelity."
)


class CEIGraphXAIWrapper(nn.Module):
    def __init__(self, adapter, *, method: str = "cei_gnn_v3"):
        super().__init__()
        # Held in a plain list so the adapter's own network is NOT registered as a
        # submodule: GNNExplainer must install its mask only on the private copy inside
        # the wrapper that actually runs the forward pass.
        self._holder = [adapter]
        self._method = method
        self.inner = None
        self._raw_context = None
        self._features = None
        self._native = None

    @property
    def adapter(self):
        return self._holder[0]

    def set_context(self, batch) -> torch.Tensor:
        from cei.studies.cei_v3_graphxai import CEIV3GraphXAIWrapper

        adapter = self.adapter.eval()
        clinical, _, _ = adapter._read(batch)
        if clinical.graph_count != 1:
            raise ValueError(
                f"this wrapper explains one graph at a time; got graph_count={clinical.graph_count}")
        self.inner = CEIV3GraphXAIWrapper(adapter, batch).eval()
        self._raw_context = batch
        self._native = None
        with torch.no_grad():
            self._features = adapter.continuous_inputs(batch).detach().clone()
        return self._features.clone()

    def forward(self, x, edge_index, batch=None, token_override=None, edge_attr_override=None,
                **kwargs):
        if self.inner is None:
            raise RuntimeError("call set_context(batch) before forward()/verify()")
        removal_mask = kwargs.pop("edge_mask", None)
        installed = removal_mask is not None
        if installed:
            set_masks(self.inner, removal_mask, edge_index.long(), apply_sigmoid=False)
        try:
            return self.inner(x, edge_index, batch)
        finally:
            if installed:
                clear_masks(self.inner)

    @torch.no_grad()
    def verify(self, x, edge_index, batch=None, atol: float = 1e-5):
        """(is_faithful, max_abs_diff): this wrapper (no mask) vs the adapter's own forward."""
        got = self.forward(x, edge_index, batch=batch)
        ref = self.adapter(self._raw_context, epoch=0).logits
        return bool(torch.allclose(got, ref, atol=atol)), float((got - ref).abs().max().item())

    @torch.no_grad()
    def builtin_node_importance(self, x, edge_index, batch=None, target_class=None):
        """His exact additive accounting for the explained class, as |net contribution|."""
        from cei.studies.cei_v3_graphxai import native_importance

        del x, batch
        target = (int(target_class) if target_class is not None
                  else int(self.forward(self._features, edge_index).argmax(-1)))
        importance, provenance = native_importance(self.adapter, self._raw_context, target)
        if importance is None:
            raise NotImplementedError(
                f"{self._method} has no per-node built-in explanation in this adapter")
        provenance = dict(provenance, intervention_scope=INTERVENTION_SCOPE, explained_class=target)
        self._native = provenance
        return importance.detach()

    @torch.no_grad()
    def builtin_detail(self):
        """Provenance of the last built-in explanation (accounting check, bias, absence)."""
        return self._native

    def ig_feature_spec(self) -> dict:
        """Widths of CEI's composite explained row, for the IG 'not recorded' baseline."""
        adapter = self.adapter
        return {"numeric_width": int(adapter.node_dim), "zeroed_tail": int(adapter.token_dim),
                "kept_tail": int(adapter.hidden)}
