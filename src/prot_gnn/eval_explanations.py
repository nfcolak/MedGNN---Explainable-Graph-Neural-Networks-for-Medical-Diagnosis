"""
Explanation quality metrics that require ground-truth node/edge masks.
Only meaningful for synthetic datasets (ShapeGGen).
"""
import numpy as np
import torch
from torch_geometric.data import Data


def explanation_accuracy(node_imp: torch.Tensor,
                         ground_truth_mask: torch.Tensor,
                         threshold: float = None) -> dict:
    """
    Compare predicted explanation to ground truth mask.
    If threshold is None, uses top-k where k = number of ground truth nodes.
    """
    gt = ground_truth_mask.float().cpu()
    imp = node_imp.float().cpu()
    
    k = int(gt.sum().item())
    if k == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": 0.5}
    
    if threshold is not None:
        predicted = (imp >= threshold).float()
    else:
        # top-k prediction — match the number of true motif nodes
        topk_indices = torch.topk(imp, k).indices
        predicted = torch.zeros_like(imp)
        predicted[topk_indices] = 1.0
    
    tp = (predicted * gt).sum().item()
    fp = (predicted * (1 - gt)).sum().item()
    fn = ((1 - predicted) * gt).sum().item()
    
    precision = tp / max(tp + fp, 1e-8)
    recall    = tp / max(tp + fn, 1e-8)
    f1 = 0.0 if (precision + recall) == 0 else \
         2 * precision * recall / (precision + recall)
    
    # AUROC between imp scores and gt mask
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(gt.numpy(), imp.numpy())
    except Exception:
        auroc = float('nan')
    
    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "auroc":     round(auroc, 4),
        "k_predicted": int(predicted.sum().item()),
        "k_true":      k,
    }


def fidelity_plus(wrapper, x, edge_index, batch,
                  node_imp, threshold=None, k=None) -> float:
    """
    Remove important nodes. Prediction should change.
    Higher = explainer found genuinely important nodes.
    """
    imp = node_imp.float().cpu()
    
    if k is None and threshold is None:
        # default: remove top 20% of nodes
        k = max(1, int(0.2 * len(imp)))
    
    if threshold is not None:
        mask = (imp >= threshold).float()
    else:
        topk = torch.topk(imp, k).indices
        mask = torch.zeros_like(imp)
        mask[topk] = 1.0
    
    mask = mask.to(x.device)
    
    # original prediction
    with torch.no_grad():
        orig_logits = wrapper(x, edge_index, batch)
        orig_pred = orig_logits.argmax(-1).item()
    
    # prediction with important nodes zeroed out
    masked_x = x * (1 - mask).unsqueeze(1)
    with torch.no_grad():
        mask_logits = wrapper(masked_x, edge_index, batch)
        mask_pred = mask_logits.argmax(-1).item()
    
    return float(orig_pred != mask_pred)  # 1.0 if prediction changed


def fidelity_minus(wrapper, x, edge_index, batch,
                   node_imp, threshold=None, k=None) -> float:
    """
    Keep only important nodes. Prediction should stay the same.
    Lower = explainer identified sufficient nodes.
    """
    imp = node_imp.float().cpu()
    
    if k is None and threshold is None:
        k = max(1, int(0.2 * len(imp)))
    
    if threshold is not None:
        mask = (imp >= threshold).float()
    else:
        topk = torch.topk(imp, k).indices
        mask = torch.zeros_like(imp)
        mask[topk] = 1.0
    
    mask = mask.to(x.device)
    
    with torch.no_grad():
        orig_logits = wrapper(x, edge_index, batch)
        orig_pred = orig_logits.argmax(-1).item()
    
    # keep only important nodes
    kept_x = x * mask.unsqueeze(1)
    with torch.no_grad():
        kept_logits = wrapper(kept_x, edge_index, batch)
        kept_pred = kept_logits.argmax(-1).item()
    
    return float(orig_pred != kept_pred)  # lower is better


def sparsity(node_imp: torch.Tensor, threshold: float = None, k: int = None) -> float:
    """
    Fraction of nodes NOT selected as important.
    Higher = more sparse explanation.
    """
    n = len(node_imp)
    if threshold is not None:
        n_selected = (node_imp >= threshold).sum().item()
    elif k is not None:
        n_selected = k
    else:
        n_selected = (node_imp > node_imp.mean()).sum().item()
    return round(1.0 - n_selected / max(n, 1), 4)


def evaluate_explanation(wrapper, data, node_imp_dict: dict) -> dict:
    """
    Run all metrics for one graph, for all explainers.

    k is set to the number of ground-truth motif nodes so that
    accuracy, fidelity+, and fidelity- all use the same budget.

    Parameters
    ----------
    wrapper       : ProtGNNWrapper
    data          : Data object with .node_mask (ground truth)
    node_imp_dict : {'GradExplainer': tensor, ...}

    Returns
    -------
    dict of dicts — one per explainer
    """
    results = {}
    x          = data.x
    edge_index = data.edge_index
    batch      = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    gt_mask    = data.node_mask if hasattr(data, 'node_mask') else None

    # Use k = number of true motif nodes so all metrics share the same budget.
    # For class-0 graphs (no motif), gt_mask is all zeros so k defaults to 1.
    if gt_mask is not None:
        k = max(1, int(gt_mask.sum().item()))
    else:
        k = max(1, int(0.2 * x.size(0)))

    for explainer_name, node_imp in node_imp_dict.items():
        entry = {}

        if gt_mask is not None:
            entry["accuracy"] = explanation_accuracy(node_imp, gt_mask)
            # accuracy already uses top-k internally with k = gt_mask.sum()

        entry["fidelity_plus"]  = fidelity_plus(
            wrapper, x, edge_index, batch, node_imp, k=k)
        entry["fidelity_minus"] = fidelity_minus(
            wrapper, x, edge_index, batch, node_imp, k=k)
        entry["sparsity"]       = sparsity(node_imp, k=k)

        results[explainer_name] = entry

    return results