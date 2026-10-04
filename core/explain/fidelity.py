"""Shared node-level fidelity, sparsity, and explanation-accuracy metrics.

All functions score one graph through a wrapper with the contract
``wrapper(x, edge_index, batch=batch) -> logits``. Fidelity+ masks out the
selected top-k nodes; Fidelity- keeps only them. Masking zeroes node-feature
rows so model wrappers do not need to support topology mutation.
"""

from numbers import Integral

import numpy as np
import torch


NODE_TOP_K_POLICY = "node_fraction_floor_min_1"
NODE_TOP_K_FRACTION = 0.2


def resolve_k(n_nodes, ground_truth_mask=None, fraction=NODE_TOP_K_FRACTION):
    """Resolve floor(fraction*n), minimum one (20% scientific default)."""
    if type(n_nodes) is not int or n_nodes < 1:
        raise ValueError("n_nodes must be a positive exact integer.")
    if ground_truth_mask is not None:
        mask = np.asarray(ground_truth_mask)
        if mask.ndim != 1 or mask.shape[0] != n_nodes:
            raise ValueError("ground_truth_mask must have one value per node.")
        return min(n_nodes, max(1, int(mask.astype(bool).sum())))
    if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
        raise ValueError("fraction must be numeric.")
    if not 0 < float(fraction) <= 1:
        raise ValueError("fraction must satisfy 0 < fraction <= 1.")
    return min(n_nodes, max(1, int(float(fraction) * n_nodes)))


def _topk_mask(node_importance, k):
    importance = np.abs(np.asarray(node_importance, dtype=float))
    if importance.ndim != 1 or importance.size == 0:
        raise ValueError("node_importance must be a non-empty one-dimensional sequence.")
    if not np.isfinite(importance).all():
        raise ValueError("node_importance must contain only finite values.")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, Integral) or k < 1:
        raise ValueError("k must be a positive exact integer.")
    # Stable tie-break: descending absolute importance, then lower node index.
    order = np.asarray(sorted(range(len(importance)), key=lambda i: (-importance[i], i)))
    mask = np.zeros(importance.shape[0], dtype=bool)
    mask[order[:min(int(k), importance.shape[0])]] = True
    return mask


def _forward(wrapper, x, edge_index, batch):
    with torch.no_grad():
        logits = wrapper(x, edge_index, batch=batch)
    if logits.ndim != 2 or logits.size(0) != 1:
        raise ValueError("Shared fidelity scores exactly one graph at a time.")
    return logits


def _pred_and_probs(logits):
    probs = torch.softmax(logits, dim=-1)
    return int(probs.argmax(dim=-1).item()), probs


def _fidelity(
    wrapper,
    x,
    edge_index,
    node_importance,
    target_class,
    batch,
    k,
    keep_important,
):
    if x.ndim != 2 or x.size(0) < 1:
        raise ValueError("x must contain at least one node-feature row.")
    if len(node_importance) != x.size(0):
        raise ValueError("node_importance must have one value per node.")
    if isinstance(target_class, (bool, np.bool_)) or not isinstance(target_class, Integral):
        raise ValueError("target_class must be an exact integer.")
    resolved_k = resolve_k(x.size(0)) if k is None else k
    important = _topk_mask(node_importance, resolved_k)
    keep = important if keep_important else ~important

    orig_pred, orig_probs = _pred_and_probs(_forward(wrapper, x, edge_index, batch))
    target = int(target_class)
    if not 0 <= target < orig_probs.size(1):
        raise ValueError(
            f"target_class must satisfy 0 <= target_class < {orig_probs.size(1)}."
        )

    x_masked = x.clone()
    x_masked[~torch.as_tensor(keep, device=x.device)] = 0.0
    masked_pred, masked_probs = _pred_and_probs(
        _forward(wrapper, x_masked, edge_index, batch)
    )
    return {
        "acc": int(orig_pred == target) - int(masked_pred == target),
        "prob": float(orig_probs[0, target] - masked_probs[0, target]),
        "k": min(int(resolved_k), x.size(0)),
        "target_class": target,
    }


def fidelity_plus(
    wrapper, x, edge_index, node_importance, target_class, batch, k=None
):
    """Mask out top-k important nodes; higher degradation is more faithful."""
    return _fidelity(
        wrapper,
        x,
        edge_index,
        node_importance,
        target_class,
        batch,
        k,
        keep_important=False,
    )


def fidelity_minus(
    wrapper, x, edge_index, node_importance, target_class, batch, k=None
):
    """Keep only top-k nodes; lower degradation is more faithful."""
    return _fidelity(
        wrapper,
        x,
        edge_index,
        node_importance,
        target_class,
        batch,
        k,
        keep_important=True,
    )


def sparsity(node_importance, mass=0.9):
    """Fraction of nodes excludable while retaining ``mass`` importance."""
    importance = np.abs(np.asarray(node_importance, dtype=float))
    if importance.ndim != 1:
        raise ValueError("node_importance must be one-dimensional.")
    if not np.isfinite(importance).all():
        raise ValueError("node_importance must contain only finite values.")
    if not isinstance(mass, (int, float)) or isinstance(mass, bool) or not 0 < mass <= 1:
        raise ValueError("mass must satisfy 0 < mass <= 1.")
    n = len(importance)
    if n == 0 or importance.sum() == 0:
        return 0.0
    ordered = np.sort(importance)[::-1]
    cumulative = np.cumsum(ordered) / importance.sum()
    selected = int(np.searchsorted(cumulative, mass) + 1)
    return round(1.0 - selected / n, 4)


def explanation_accuracy(node_importance, ground_truth_mask, k=None):
    """Jaccard overlap between top-k importance and a ground-truth node mask."""
    ground_truth = np.asarray(ground_truth_mask, dtype=bool)
    if ground_truth.ndim != 1:
        raise ValueError("ground_truth_mask must be one-dimensional.")
    resolved_k = (
        resolve_k(len(ground_truth), ground_truth_mask=ground_truth)
        if k is None
        else k
    )
    predicted = _topk_mask(node_importance, resolved_k)
    if predicted.shape != ground_truth.shape:
        raise ValueError("node_importance and ground_truth_mask lengths must match.")
    intersection = int(np.logical_and(predicted, ground_truth).sum())
    union = int(np.logical_or(predicted, ground_truth).sum())
    return round(intersection / union, 4) if union else 0.0
