"""Fidelity with full evidence removal (GraphXAI improvement plan item #4).

`core/explain/fidelity.py` (v1, untouched here, still what the legacy star/
cooccur/ontology/full pipeline and your already-reported thesis numbers run
through) "removes" a node by zeroing its `x` row alone. The node's `token`,
its numeric value, and every edge connecting it to the rest of the graph all
survive that intervention -- so a model with any path that reads token/value
directly instead of purely through `x` (GCHM-v3's `wide_token`/`wide_value`
class-vote paths are the confirmed case, but it's not specific to them: any
method's token embedding lookup sees the "removed" node's real identity) can
still see evidence that was supposed to be gone. This module removes a node's
full identity instead: `x` zeroed, `token` replaced with the padding/unseen
id (0 -- every token embedding in this codebase uses `padding_idx=0`), and
every edge touching it masked to zero via the `edge_mask` hook already built
and verified earlier this project. A node with no feature signal, no token
identity, and no graph connectivity is evidence that is actually gone, not
quieted in one channel while two others still carry it.

Structural nodes -- `patient`, `visit` (the graph's scaffolding; removing the
index-visit hub doesn't probe "was this evidence important", it collapses
structure the model was never trained to see without) and `knowledge` (KG
grounding nodes, present only for GraphCare, already excluded the same way by
GraphCare's own direct-channel readout) -- are never candidates for removal.

Do not compare these numbers to `core/explain/fidelity.py`'s. They measure a
different, stricter intervention, exactly as the plan's own item #4 says not
to conflate.

Deletion/insertion curves (the plan's "[new]" bullet): removing nodes one at a
time off a real graph walks it off its training distribution further with
every node removed, so a single top-20% point can't distinguish "this
explanation is good" from "this model breaks under any perturbation". A curve
over k -- and its AUC -- is reported instead for the same reason ROAR/
insertion-deletion curves are standard in the wider explainability literature,
not just one point.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from core import NODE_KINDS

STRUCTURAL_NODE_KINDS = ("patient", "visit", "knowledge")
STRUCTURAL_NODE_TYPE_IDS = tuple(NODE_KINDS.index(kind) for kind in STRUCTURAL_NODE_KINDS)
PAD_TOKEN_ID = 0
NODE_TOP_K_FRACTION = 0.2


def removable_node_mask(node_type: torch.Tensor) -> torch.Tensor:
    """True for every node this intervention is allowed to remove."""
    structural = torch.zeros_like(node_type, dtype=torch.bool)
    for type_id in STRUCTURAL_NODE_TYPE_IDS:
        structural |= node_type == type_id
    return ~structural


def resolve_k(num_removable: int, fraction: float = NODE_TOP_K_FRACTION) -> int:
    if type(num_removable) is not int or num_removable < 1:
        raise ValueError("no removable nodes to select from")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1")
    return min(num_removable, max(1, int(fraction * num_removable)))


def _ordered_removable_indices(importance: np.ndarray, removable: torch.Tensor) -> np.ndarray:
    """Removable node indices, most-to-least important, ties by lower index."""
    removable_indices = np.flatnonzero(removable.detach().cpu().numpy())
    if removable_indices.size == 0:
        raise ValueError("no removable nodes available for top-k selection")
    order = sorted(removable_indices.tolist(), key=lambda i: (-abs(importance[i]), i))
    return np.asarray(order, dtype=np.int64)


def _evict(x: torch.Tensor, token: torch.Tensor, node_indices: np.ndarray):
    """x with the selected rows zeroed, token with the selected ids set to the
    pad id, and a per-node boolean 'evicted' mask (for deriving an edge_mask)."""
    evicted = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
    evicted[torch.as_tensor(node_indices, dtype=torch.long, device=x.device)] = True
    x_evicted = x.clone()
    x_evicted[evicted] = 0.0
    token_evicted = token.clone()
    token_evicted[evicted] = PAD_TOKEN_ID
    return x_evicted, token_evicted, evicted


def _edge_mask_for_eviction(edge_index: torch.Tensor, evicted: torch.Tensor) -> torch.Tensor:
    """1.0 for an edge with neither endpoint evicted, 0.0 if either is --
    reuses the edge_mask hook verified across every method earlier this
    project rather than physically deleting rows from edge_index."""
    if edge_index.numel() == 0:
        return torch.ones(0, dtype=torch.float32, device=edge_index.device)
    source, target = edge_index
    return (~(evicted[source] | evicted[target])).float()


def _predict(wrapper, x, edge_index, token=None, edge_mask=None):
    with torch.no_grad():
        logits = wrapper(x, edge_index, token_override=token, edge_mask=edge_mask)
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("fidelity_v2 scores exactly one graph at a time")
    probs = torch.softmax(logits, dim=-1)
    return int(probs.argmax(dim=-1).item()), probs


def _score_at_k(wrapper, x, token, edge_index, node_type, node_importance, target_class,
                k: int, *, keep_important: bool) -> dict[str, Any]:
    removable = removable_node_mask(node_type)
    importance = np.abs(np.asarray(node_importance, dtype=float))
    ordered = _ordered_removable_indices(importance, removable)
    k = min(k, ordered.size)
    top = ordered[:k]
    to_evict = (
        np.setdiff1d(ordered, top, assume_unique=False) if keep_important else top
    )
    orig_pred, orig_probs = _predict(wrapper, x, edge_index)
    if to_evict.size == 0:
        # Nothing left to remove (k covers every removable node, keep-mode):
        # the untouched graph is its own answer, not an error.
        evicted_pred, evicted_probs = orig_pred, orig_probs
    else:
        x_evicted, token_evicted, evicted = _evict(x, token, to_evict)
        edge_mask = _edge_mask_for_eviction(edge_index, evicted)
        evicted_pred, evicted_probs = _predict(
            wrapper, x_evicted, edge_index, token=token_evicted, edge_mask=edge_mask
        )
    return {
        "acc": int(orig_pred == target_class) - int(evicted_pred == target_class),
        "prob": float(orig_probs[0, target_class] - evicted_probs[0, target_class]),
        "k": int(k),
        "target_class": int(target_class),
        "removable_node_count": int(removable.sum().item()),
    }


def fidelity_plus_v2(wrapper, x, token, edge_index, node_type, node_importance,
                     target_class, k: int | None = None) -> dict[str, Any]:
    """Evict the top-k most important REMOVABLE nodes (x, token, and every
    incident edge); higher prob drop = more faithful. k defaults to 20% of
    removable nodes, floor 1 -- same policy core/explain/fidelity.py uses, over
    the removable set instead of every node."""
    removable_count = int(removable_node_mask(node_type).sum().item())
    resolved_k = resolve_k(removable_count) if k is None else int(k)
    return _score_at_k(wrapper, x, token, edge_index, node_type, node_importance,
                       target_class, resolved_k, keep_important=False)


def fidelity_minus_v2(wrapper, x, token, edge_index, node_type, node_importance,
                      target_class, k: int | None = None) -> dict[str, Any]:
    """Keep only the top-k most important removable nodes, evict every other
    removable node (structural nodes are never evicted either way); lower
    prob drop = more faithful."""
    removable_count = int(removable_node_mask(node_type).sum().item())
    resolved_k = resolve_k(removable_count) if k is None else int(k)
    return _score_at_k(wrapper, x, token, edge_index, node_type, node_importance,
                       target_class, resolved_k, keep_important=True)


def _curve(wrapper, x, token, edge_index, node_type, node_importance, target_class,
          *, keep_important: bool, num_points: int) -> dict[str, Any]:
    removable_count = int(removable_node_mask(node_type).sum().item())
    fractions = [i / num_points for i in range(1, num_points + 1)]
    ks = sorted({max(1, int(round(fraction * removable_count))) for fraction in fractions})
    points = [
        _score_at_k(wrapper, x, token, edge_index, node_type, node_importance,
                   target_class, k, keep_important=keep_important)
        for k in ks
    ]
    probs = [point["prob"] for point in points]
    # Trapezoidal AUC over k/removable_count in [0, 1], with an explicit (0, 0)
    # start point (zero nodes removed -> zero probability change, by
    # construction, not measured) so the curve's domain is always the same
    # regardless of how few points were requested.
    xs = [0.0] + [point["k"] / removable_count for point in points]
    ys = [0.0] + probs
    auc = float(np.trapz(ys, xs))
    return {
        "k_values": [point["k"] for point in points],
        "prob_at_k": probs,
        "removable_node_count": removable_count,
        "auc": auc,
    }


def deletion_curve(wrapper, x, token, edge_index, node_type, node_importance,
                   target_class, *, num_points: int = 5) -> dict[str, Any]:
    """Fidelity+ at increasing k (most-important-first removal); AUC over
    k/removable_count. Higher AUC = removing important evidence consistently
    hurts the prediction, not just at one arbitrary threshold."""
    return _curve(wrapper, x, token, edge_index, node_type, node_importance,
                 target_class, keep_important=False, num_points=num_points)


def relation_vs_payload_contribution(wrapper, x, edge_index, edge_attr,
                                     num_relation_columns: int,
                                     target_class: int) -> dict[str, Any]:
    """GraphXAI improvement plan item #5's second ask: how much of the edges'
    contribution to this prediction is the relation TYPE (the one-hot block,
    columns [0, num_relation_columns)) versus the numeric PAYLOAD (delta/
    interval/recency/prior-encounters, every column after it)? Answered the
    same way every other fidelity metric in this module is -- zero one block,
    measure the probability drop -- rather than a gradient, so it needs no
    machinery beyond the edge_attr_override hook already on the wrapper.
    Graph-level (both blocks across every edge at once), not per-edge: the
    question is which TYPE of edge evidence this prediction leans on, not
    which single edge.
    """
    if edge_index.numel() == 0:
        return {"relation_type_prob_drop": 0.0, "numeric_payload_prob_drop": 0.0,
               "edge_count": 0}
    if edge_attr.ndim != 2 or edge_attr.size(1) <= num_relation_columns:
        raise ValueError("edge_attr must have at least one payload column "
                        "after num_relation_columns")
    with torch.no_grad():
        orig_probs = torch.softmax(wrapper(x, edge_index), dim=-1)
        relation_zeroed = edge_attr.clone()
        relation_zeroed[:, :num_relation_columns] = 0.0
        relation_zeroed_probs = torch.softmax(
            wrapper(x, edge_index, edge_attr_override=relation_zeroed), dim=-1)
        payload_zeroed = edge_attr.clone()
        payload_zeroed[:, num_relation_columns:] = 0.0
        payload_zeroed_probs = torch.softmax(
            wrapper(x, edge_index, edge_attr_override=payload_zeroed), dim=-1)
    target = int(target_class)
    return {
        # Dropping relation-type-only isolates relation type's contribution;
        # dropping payload-only isolates the payload's. The two need not sum
        # to the full-removal drop (they can overlap or partially substitute).
        "relation_type_prob_drop": float(orig_probs[0, target] - relation_zeroed_probs[0, target]),
        "numeric_payload_prob_drop": float(orig_probs[0, target] - payload_zeroed_probs[0, target]),
        "edge_count": int(edge_index.size(1)),
        "target_class": target,
    }


def insertion_curve(wrapper, x, token, edge_index, node_type, node_importance,
                    target_class, *, num_points: int = 5) -> dict[str, Any]:
    """Fidelity- at increasing k (most-important-first retention); AUC over
    k/removable_count. Lower AUC = a small amount of important evidence
    consistently suffices to preserve the prediction."""
    return _curve(wrapper, x, token, edge_index, node_type, node_importance,
                 target_class, keep_important=True, num_points=num_points)
