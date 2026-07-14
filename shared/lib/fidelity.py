"""Fidelity / sparsity / explanation-accuracy metrics, shared by ProtGNN and
GraphCare so the two methods' explanations are audited identically (see
CLAUDE.md's shared/lib/ convention).

Formulas follow Yuan et al., "Explainability in Graph Neural Networks: A
Taxonomic Survey" (docs/literature/), Sec. 7.2.1 (Fidelity+/-) and 7.2.2
(Sparsity).

All functions score ONE graph; callers average the returned values over the
test set (same pattern as the per-graph loops in explain_checkpoint.py /
explain_graphcare.py).

Masking is done by zeroing node FEATURE rows rather than structurally
removing nodes/edges from edge_index. This is what lets these functions run
through either wrapper unchanged: both ProtGNNWrapper and
GraphCareGraphXAIWrapper accept `wrapper(x, edge_index, batch=batch) ->
logits`, and neither has to handle a changing node count mid-call.

Fidelity+ masks OUT the top-k important nodes (checks the prediction
degrades). Fidelity- masks TO KEEP ONLY the top-k important nodes (checks
the prediction is preserved). Swapping these silently gives all-zero
results — see docs/PROJECT_CONTEXT.md Sec. 2.
"""
import numpy as np
import torch


def resolve_k(n_nodes, ground_truth_mask=None, fraction=0.2):
    """k = |ground-truth motif| if available, else max(1, fraction * n_nodes).

    The fallback branch is what makes the same functions work on MIMIC (no
    ground truth) without modification.
    """
    if ground_truth_mask is not None:
        return max(1, int(np.asarray(ground_truth_mask).sum()))
    return max(1, int(fraction * n_nodes))


def _topk_mask(node_importance, k):
    imp = np.abs(np.asarray(node_importance, dtype=float))
    order = np.argsort(imp)[::-1]
    mask = np.zeros(imp.shape[0], dtype=bool)
    mask[order[:min(k, imp.shape[0])]] = True
    return mask


def _forward(wrapper, x, edge_index, batch):
    with torch.no_grad():
        return wrapper(x, edge_index, batch=batch)


def _pred_and_probs(logits):
    probs = torch.softmax(logits, dim=-1)
    return int(probs.argmax(dim=-1).item()), probs


def _fidelity(wrapper, x, edge_index, node_importance, true_label, batch, k, keep_important):
    n = x.size(0)
    k = k if k is not None else resolve_k(n)
    important = _topk_mask(node_importance, k)
    keep = important if keep_important else ~important

    orig_pred, orig_probs = _pred_and_probs(_forward(wrapper, x, edge_index, batch))

    x_masked = x.clone()
    x_masked[~torch.as_tensor(keep, device=x.device)] = 0.0
    masked_pred, masked_probs = _pred_and_probs(_forward(wrapper, x_masked, edge_index, batch))

    y = int(true_label)
    return {
        "acc": int(orig_pred == y) - int(masked_pred == y),
        "prob": float(orig_probs[0, y] - masked_probs[0, y]),
        "k": k,
    }


def fidelity_plus(wrapper, x, edge_index, node_importance, true_label, batch, k=None):
    """Mask OUT the top-k important nodes. Higher = more faithful explanation
    (removing what the explanation called important should hurt the model)."""
    return _fidelity(wrapper, x, edge_index, node_importance, true_label, batch, k,
                      keep_important=False)


def fidelity_minus(wrapper, x, edge_index, node_importance, true_label, batch, k=None):
    """Mask to KEEP ONLY the top-k important nodes. Lower = more faithful
    explanation (the important nodes alone should reproduce the original
    prediction)."""
    return _fidelity(wrapper, x, edge_index, node_importance, true_label, batch, k,
                      keep_important=True)


def sparsity(node_importance, mass=0.9):
    """Fraction of nodes excludable while still capturing `mass` of the total
    importance (higher = more concentrated explanation). Identical to
    graphcare_analysis/explainability/explain_graphcare.py's sparsity() so
    both methods report numerically comparable scores.
    """
    imp = np.abs(np.asarray(node_importance, dtype=float))
    n = len(imp)
    if n == 0 or imp.sum() == 0:
        return 0.0
    order = np.sort(imp)[::-1]
    cum = np.cumsum(order) / imp.sum()
    k = int(np.searchsorted(cum, mass) + 1)
    return round(1.0 - k / n, 4)


def explanation_accuracy(node_importance, ground_truth_mask, k=None):
    """Graph Explanation Accuracy (GEA): Jaccard index between the top-k
    predicted-important nodes and the ground-truth explanation mask. Only
    meaningful when ground truth exists (synthetic data).
    """
    gt = np.asarray(ground_truth_mask, dtype=bool)
    k = k if k is not None else resolve_k(len(gt), ground_truth_mask=gt)
    pred_mask = _topk_mask(node_importance, k)

    tp = int(np.logical_and(pred_mask, gt).sum())
    fp = int(np.logical_and(pred_mask, ~gt).sum())
    fn = int(np.logical_and(~pred_mask, gt).sum())
    denom = tp + fp + fn
    return round(tp / denom, 4) if denom else 0.0
