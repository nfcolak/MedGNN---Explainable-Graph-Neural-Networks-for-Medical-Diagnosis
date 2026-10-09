"""GraphXAI-compatible wrappers for clinical_graph_v2 method adapters.

`ClinicalGraphXAIWrapper` covers every method whose forward() only needs the
standard ClinicalBatch tensors (x, token, node_type, edge_index, edge_attr,
edge_relation, edge_triple, batch_index, graph_count): gsat, protgnn,
and -- via `is_method_adapter=False` -- GCHMv2/GCHMv3. The GCHM pair aren't
ClinicalMethodAdapter subclasses at all (they're not in METHOD_REGISTRY;
train.py constructs and calls them through a separate path): their
forward(data, edge_mask=None) returns a plain logits tensor directly, no
MethodOutput wrapper and no `epoch` argument, which is the one thing
`is_method_adapter` branches on -- everything else (context assembly,
set_context, verify) is identical. `GraphCareGraphXAIWrapper` is the one
method that doesn't fit this class at all -- GraphCare also requires
visit_membership_index, num_visits, and global_node_mask, which GraphXAI
never supplies, so (like every other wrapper in this project) they're fixed
once via set_context() and held as closed-over context rather than part of
the differentiable (x, edge_index) GraphXAI perturbs.

Same contract throughout this project (old graph and new): set_context(batch)
-> x, forward(x, edge_index, batch=None, edge_mask=None) -> logits,
verify(x, edge_index) -> (is_faithful, max_abs_diff). Like every wrapper here,
these score exactly one graph at a time (core/explain/fidelity.py's own
convention): graph_count must be 1.

`edge_mask` reaches the model two different ways depending on the method:
- protgnn's `_RelationPayloadLayer` (protgnn/adapter.py) is the local message
  pass this mask multiplies into exactly once (both the message numerator and
  the degree-normalisation denominator).
- GraphCare's own `_BATLayer` gets the same treatment, with the mask first
  subset to the same global-node-excluding edge filter GraphCare's own
  message pass already applies (see GraphCareGraphXAIWrapper.forward), so an
  external mask lands on the correct edges once that filter runs.
- GCHMv2/v3's `HubGatedPNALayer` aggregates four statistics (mean/min/max/std)
  per node, not one sum/mean -- "mask this edge out" means something
  different for each: sum/mean/std divide by the mask's own weighted count
  rather than the raw edge count, so a masked-out edge is truly removed
  rather than diluting the average toward zero; min/max push a masked
  message toward a saturating sentinel instead of multiplying it by the
  mask, since a naively zeroed message is still a real candidate value that
  could win the reduction outright. See HubGatedPNALayer._aggregate's own
  docstring-comment for the full reasoning.
"""
from __future__ import annotations

import inspect
import types

import torch
import torch.nn as nn

from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

from core.explain import gchm_builtin
from core.method_base import ClinicalBatch, read_clinical_batch
from core.tensorize import PAYLOAD_WIDTH

# Methods whose adapter exposes .explain(batch) -> per-node tensor as its own
# built-in node-importance readout: GSAT's deterministic attention, and ProtGNN's
# own prototype subgraph (binary membership; the model's explanation is a
# subgraph, not a ranking). Others raise rather than fabricating a score.
_SUPPORTS_BUILTIN_EXPLAIN = frozenset({"gsat", "protgnn"})
# Models that are not ClinicalMethodAdapters and keep their code untouched: their built-in
# explanation is read off their own computation by a function in core/explain/.
# GCHM-PNA v3: node-additive label-wise + wide-vote accounting. v2 has neither path.
_EXTERNAL_BUILTINS = {"gchm_v3": gchm_builtin.explain}


class ClinicalGraphXAIWrapper(nn.Module):
    def __init__(self, adapter, *, method: str, node_dim: int, edge_dim: int,
                 num_tokens: int, num_triples: int, num_relations: int,
                 is_method_adapter: bool = True):
        super().__init__()
        self.adapter = adapter
        self._method = method
        self._node_dim = int(node_dim)
        self._edge_dim = int(edge_dim)
        self._num_tokens = int(num_tokens)
        self._num_triples = int(num_triples)
        self._num_relations = int(num_relations)
        # True for every ClinicalMethodAdapter (forward(batch, *, epoch,
        # external_edge_mask) -> MethodOutput). GCHMv2/v3 are plain nn.Module
        # classes with their own simpler forward(data, edge_mask=None) ->
        # logits convention -- not registered in METHOD_REGISTRY, not part of
        # ClinicalMethodAdapter at all -- so this wrapper calls them
        # differently rather than needing a near-duplicate class.
        self._is_method_adapter = is_method_adapter
        self._context: ClinicalBatch | None = None
        self._raw_context = None
        self._external_detail = None

    def set_context(self, batch) -> torch.Tensor:
        clinical_batch = read_clinical_batch(
            batch, method=self._method, node_dim=self._node_dim, edge_dim=self._edge_dim,
            num_tokens=self._num_tokens, num_triples=self._num_triples,
            num_relations=self._num_relations,
        )
        if clinical_batch.graph_count != 1:
            raise ValueError(
                "this wrapper explains one graph at a time; got "
                f"graph_count={clinical_batch.graph_count}"
            )
        self._context = clinical_batch
        self._raw_context = batch
        self._external_detail = None
        return clinical_batch.x

    def _batch_with(self, x, edge_index, *, token=None, edge_attr=None):
        if self._context is None:
            raise RuntimeError("call set_context(batch) before forward()/verify()")
        ctx = self._context
        if x.shape != ctx.x.shape:
            raise ValueError("x must keep the shape set_context() was given")
        if edge_index.shape != ctx.edge_index.shape or not torch.equal(
            edge_index.long(), ctx.edge_index
        ):
            raise ValueError("edge_index must match the graph passed to set_context()")
        if token is not None and token.shape != ctx.token.shape:
            raise ValueError("token override must keep the shape set_context() was given")
        if edge_attr is not None and edge_attr.shape != ctx.edge_attr.shape:
            raise ValueError("edge_attr override must keep the shape set_context() was given")
        used_edge_attr = ctx.edge_attr if edge_attr is None else edge_attr
        return types.SimpleNamespace(
            x=x, token=ctx.token if token is None else token, node_type=ctx.node_type,
            edge_index=ctx.edge_index, edge_attr=used_edge_attr,
            edge_payload=used_edge_attr[:, -PAYLOAD_WIDTH:],
            edge_relation=ctx.edge_relation,
            edge_triple=ctx.edge_triple, batch=ctx.batch_index, num_graphs=ctx.graph_count,
        )

    def forward(self, x, edge_index, batch=None, edge_mask=None, token_override=None,
               edge_attr_override=None, **kwargs):
        del batch, kwargs
        view = self._batch_with(x, edge_index, token=token_override, edge_attr=edge_attr_override)
        was_training = self.adapter.training
        self.adapter.eval()
        try:
            if self._is_method_adapter:
                return self.adapter(view, epoch=0, external_edge_mask=edge_mask).logits
            return self.adapter(view, edge_mask=edge_mask)
        finally:
            if was_training:
                self.adapter.train()

    @torch.no_grad()
    def builtin_node_importance(self, x, edge_index, batch=None, target_class=None):
        """Calls the adapter's own `.explain(batch)`, for the methods that
        have one (GSAT, ProtGNN). Always a fresh, unmasked pass -- never
        reuses state from a prior forward(edge_mask=...) probe. `target_class` is
        passed only to adapters whose explanation is class-specific (ProtGNN)."""
        del x, edge_index, batch
        if self._method in _EXTERNAL_BUILTINS:
            importance, self._external_detail = _EXTERNAL_BUILTINS[self._method](
                self.adapter, self._raw_context, target_class)
            return importance
        if self._method not in _SUPPORTS_BUILTIN_EXPLAIN:
            raise NotImplementedError(
                f"{self._method} has no per-node built-in explanation in this adapter"
            )
        was_training = self.adapter.training
        self.adapter.eval()
        try:
            if target_class is not None and "target_class" in inspect.signature(
                    self.adapter.explain).parameters:
                return self.adapter.explain(self._raw_context, target_class).detach()
            return self.adapter.explain(self._raw_context).detach()
        finally:
            if was_training:
                self.adapter.train()

    @torch.no_grad()
    def builtin_detail(self):
        """The adapter's own structured explanation (e.g. ProtGNN's prototype,
        its activation and the matching subgraph), or None if it has none."""
        if self._external_detail is not None:
            return self._external_detail
        detail = getattr(self.adapter, "explain_detail", None)
        return None if detail is None else detail(self._raw_context)

    @torch.no_grad()
    def verify(self, x, edge_index, batch=None, atol: float = 1e-5):
        """(is_faithful, max_abs_diff): wrapper logits (no mask) vs the
        adapter's own direct eval-mode forward on the identical batch."""
        view = self._batch_with(x, edge_index)
        was_training = self.adapter.training
        self.adapter.eval()
        try:
            got = self.forward(x, edge_index, batch=batch)
            ref = (self.adapter(view, epoch=0).logits if self._is_method_adapter
                  else self.adapter(view))
        finally:
            if was_training:
                self.adapter.train()
        return bool(torch.allclose(got, ref, atol=atol)), float((got - ref).abs().max().item())


class ClinicalGNNGraphXAIWrapper(ClinicalGraphXAIWrapper):
    """Black-box baselines: `ClinicalGNN` with conv in (edge_conditioned, hgt, gchm).

    Their layers are genuine `torch_geometric.nn.MessagePassing` modules, so
    GNNExplainer's learned mask reaches the messages through PyG's own
    `set_masks` hook with no model change. Because of that this wrapper's
    `forward` deliberately does NOT declare an `edge_mask` parameter: if it did,
    CompatibleGNNExplainer would deliver the mask a second time through the
    keyword and the mask would be applied twice per edge (plan item 3 requires
    exactly once). An explicit removal mask from the fidelity code (an evicted
    node's incident edges) arrives as a plain keyword and is installed through
    the same PyG hook for that one call only.

    Soft-mask caveat, as for every other model here: the mask multiplies each
    message after it is computed. For HGT that is after the per-receiver softmax,
    so masked-out edges do not renormalise the remaining attention weights, and
    for GCHM's PNA-style aggregation a zeroed message still takes part in
    min/max. A soft mask is not the same intervention as deleting the edge.
    `hgt` also reads relation identity from `edge_triple`, not from `edge_attr`.
    """

    def __init__(self, adapter, *, method: str, node_dim: int, edge_dim: int,
                 num_tokens: int, num_triples: int, num_relations: int):
        super().__init__(adapter, method=method, node_dim=node_dim, edge_dim=edge_dim,
                         num_tokens=num_tokens, num_triples=num_triples,
                         num_relations=num_relations, is_method_adapter=False)

    def forward(self, x, edge_index, batch=None, token_override=None,
                edge_attr_override=None, **kwargs):
        del batch
        removal_mask = kwargs.pop("edge_mask", None)
        view = self._batch_with(x, edge_index, token=token_override, edge_attr=edge_attr_override)
        was_training = self.adapter.training
        self.adapter.eval()
        installed = removal_mask is not None
        if installed:
            set_masks(self.adapter, removal_mask, edge_index.long(), apply_sigmoid=False)
        try:
            return self.adapter(view)
        finally:
            if installed:
                clear_masks(self.adapter)
            if was_training:
                self.adapter.train()


class GraphCareGraphXAIWrapper(nn.Module):
    def __init__(self, adapter, *, node_dim: int, edge_dim: int, num_tokens: int,
                 num_triples: int, num_relations: int):
        super().__init__()
        self.adapter = adapter
        self._node_dim = int(node_dim)
        self._edge_dim = int(edge_dim)
        self._num_tokens = int(num_tokens)
        self._num_triples = int(num_triples)
        self._num_relations = int(num_relations)
        self._context: ClinicalBatch | None = None
        self._raw_context = None

    def set_context(self, batch) -> torch.Tensor:
        clinical_batch = read_clinical_batch(
            batch, method="GraphCare", node_dim=self._node_dim, edge_dim=self._edge_dim,
            num_tokens=self._num_tokens, num_triples=self._num_triples,
            num_relations=self._num_relations,
        )
        if clinical_batch.graph_count != 1:
            raise ValueError(
                "this wrapper explains one graph at a time; got "
                f"graph_count={clinical_batch.graph_count}"
            )
        for name in ("visit_membership_index", "num_visits", "global_node_mask"):
            if getattr(batch, name, None) is None:
                raise ValueError(f"GraphCare requires {name}")
        self._context = clinical_batch
        self._raw_context = batch
        return clinical_batch.x

    def _batch_with(self, x, edge_index, *, token=None, edge_attr=None):
        if self._context is None:
            raise RuntimeError("call set_context(batch) before forward()/verify()")
        ctx = self._context
        if x.shape != ctx.x.shape:
            raise ValueError("x must keep the shape set_context() was given")
        if edge_index.shape != ctx.edge_index.shape or not torch.equal(
            edge_index.long(), ctx.edge_index
        ):
            raise ValueError("edge_index must match the graph passed to set_context()")
        if token is not None and token.shape != ctx.token.shape:
            raise ValueError("token override must keep the shape set_context() was given")
        if edge_attr is not None and edge_attr.shape != ctx.edge_attr.shape:
            raise ValueError("edge_attr override must keep the shape set_context() was given")
        raw = self._raw_context
        return types.SimpleNamespace(
            x=x, token=ctx.token if token is None else token, node_type=ctx.node_type,
            edge_index=ctx.edge_index, edge_attr=ctx.edge_attr if edge_attr is None else edge_attr,
            edge_relation=ctx.edge_relation,
            edge_triple=ctx.edge_triple, batch=ctx.batch_index, num_graphs=ctx.graph_count,
            visit_membership_index=raw.visit_membership_index,
            num_visits=raw.num_visits, global_node_mask=raw.global_node_mask,
        )

    def forward(self, x, edge_index, batch=None, edge_mask=None, token_override=None,
               edge_attr_override=None, **kwargs):
        del batch, kwargs
        view = self._batch_with(x, edge_index, token=token_override, edge_attr=edge_attr_override)
        was_training = self.adapter.training
        self.adapter.eval()
        try:
            output = self.adapter(view, epoch=0, external_edge_mask=edge_mask)
        finally:
            if was_training:
                self.adapter.train()
        return output.logits

    @torch.no_grad()
    def builtin_node_importance(self, x, edge_index, batch=None, target_class=None):
        """GraphCare's own node-conditioned attention (class-independent, so
        `target_class` is accepted for interface parity and ignored), scattered to one value
        per node (last_node_attention), from a fresh, unmasked forward pass
        this call performs. NOT last_alpha: that array is indexed over only
        the direct (non-global, non-knowledge) attention pairs, not one entry
        per node, so it is unusable as a node_importance array against the
        full node indexing (x/token/node_type) fidelity and GraphXAI expect."""
        self.forward(x, edge_index, batch=batch)
        node_attention = self.adapter.last_node_attention
        if node_attention is None or not node_attention.numel():
            raise RuntimeError("GraphCare produced no node attention on this graph")
        return node_attention.detach()

    @torch.no_grad()
    def verify(self, x, edge_index, batch=None, atol: float = 1e-5):
        view = self._batch_with(x, edge_index)
        was_training = self.adapter.training
        self.adapter.eval()
        try:
            got = self.forward(x, edge_index, batch=batch)
            ref = self.adapter(view, epoch=0).logits
        finally:
            if was_training:
                self.adapter.train()
        return bool(torch.allclose(got, ref, atol=atol)), float((got - ref).abs().max().item())
