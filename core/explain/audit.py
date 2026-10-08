"""Audit controls for explanations (GraphXAI improvement plan item 8) and the
edge-dependence diagnostic for edge-mask explainers (item 3's disclosure).

- weight randomisation (Adebayo et al., 2018): an explanation that survives
  re-drawing every weight is not explaining the model.
- stability: small value changes that leave the prediction alone should leave the
  explanation (nearly) alone.
- edge dependence: how much of the prediction travels along edges at all. GNNExplainer
  can only explain that share; node-level routes bypass its mask.

All comparisons use the same eligible evidence nodes as the fidelity metrics.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata

from core import NODE_KINDS
from core.explain import fidelity_v2 as fv2

VALUE_COLUMN = len(NODE_KINDS)          # scaled_value
HAS_VALUE_COLUMN = len(NODE_KINDS) + 1  # has_value


def randomise_weights(model: torch.nn.Module, seed: int, floor: float = 0.1) -> torch.nn.Module:
    """A deep copy with every parameter re-drawn from N(0, max(own std, floor)).
    Buffers (class ids, degree histograms) are structure, not weights: left alone."""
    clone = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for parameter in clone.parameters():
            scale = max(float(parameter.std()) if parameter.numel() > 1 else 0.0, floor)
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * scale)
    return clone


def perturb_values(x: torch.Tensor, sigma: float, seed: int) -> tuple[torch.Tensor, int]:
    """Add N(0, sigma) (z-score units) to the value of every measured node only."""
    generator = torch.Generator().manual_seed(int(seed))
    perturbed = x.clone()
    measured = x[:, HAS_VALUE_COLUMN] > 0.5
    noise = torch.randn(x.size(0), generator=generator) * sigma
    perturbed[measured, VALUE_COLUMN] += noise[measured]
    return perturbed, int(measured.sum())


def _top_set(importance: np.ndarray, node_type: torch.Tensor) -> tuple[set[int], int]:
    removable = fv2.removable_node_mask(node_type)
    ordered = fv2._ordered_removable_indices(np.abs(importance), removable)
    support = int((np.abs(importance)[ordered] > 0).sum())
    k = min(fv2.resolve_k(int(removable.sum())), max(support, 1))
    return set(int(i) for i in ordered[:k]), k


def agreement(a, b, node_type: torch.Tensor) -> dict[str, Any]:
    """Spearman correlation and top-k Jaccard between two importance vectors, over
    eligible evidence nodes. Spearman is None when either side is constant."""
    a, b = np.abs(np.asarray(a, dtype=float)), np.abs(np.asarray(b, dtype=float))
    eligible = fv2.removable_node_mask(node_type).detach().cpu().numpy()
    ea, eb = a[eligible], b[eligible]
    spearman = None
    if ea.size > 1 and np.ptp(ea) > 0 and np.ptp(eb) > 0:
        ra, rb = rankdata(ea), rankdata(eb)
        spearman = float(np.corrcoef(ra, rb)[0, 1])
    set_a, k_a = _top_set(a, node_type)
    set_b, k_b = _top_set(b, node_type)
    union = set_a | set_b
    return {"spearman": spearman,
            "top_k_jaccard": len(set_a & set_b) / len(union) if union else None,
            "k": [k_a, k_b]}


def compare_sources(reference: dict[str, Any], other: dict[str, Any],
                    node_type: torch.Tensor) -> dict[str, Any]:
    """`agreement` for every source present (and successful) on both sides."""
    out: dict[str, Any] = {}
    for name, vector in reference.items():
        if name in other:
            try:
                out[name] = agreement(vector, other[name], node_type)
            except ValueError as exc:   # e.g. no eligible nodes
                out[name] = {"status": "failed", "error": str(exc)}
    return out


def edge_dependence(wrapper, x: torch.Tensor, edge_index: torch.Tensor,
                    target_class: int) -> dict[str, Any]:
    """Prediction with every edge masked out vs the normal one. A small drop means
    most of the prediction does not travel along edges, so an edge-mask explainer
    (GNNExplainer) can only speak for a small part of it."""
    if edge_index.numel() == 0:
        return {"status": "not_applicable", "reason": "graph has no edges"}
    with torch.no_grad():
        full = torch.softmax(wrapper(x, edge_index), dim=-1)[0]
        none = torch.softmax(wrapper(x, edge_index, edge_mask=torch.zeros(edge_index.size(1))), dim=-1)[0]
    return {
        "probability_unmasked": float(full[target_class]),
        "probability_all_edges_masked": float(none[target_class]),
        "probability_drop": float(full[target_class] - none[target_class]),
        "prediction_unchanged_without_edges": bool(full.argmax() == none.argmax()),
        "edge_count": int(edge_index.size(1)),
    }
