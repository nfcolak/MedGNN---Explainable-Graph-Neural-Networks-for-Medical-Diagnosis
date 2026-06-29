"""Shared, leakage-controlled data splitting used by BOTH analyses.

`subject_aware_split_indices` keeps every visit of a given patient
(`subject_id`) in exactly ONE fold, so the same subject never leaks across
train/val/test. This MUST be identical for protgnn_analysis and
graphcare_analysis so the comparison is fair.

Mirrors the split logic in protgnn_analysis/load_dataset.py (kept in sync).
"""
import numpy as np


def extract_graph_labels(dataset) -> np.ndarray:
    labels = []
    for idx in range(len(dataset)):
        y = dataset[idx].y
        try:
            y = y.view(-1)[0].item()
        except AttributeError:
            pass
        labels.append(int(y))
    return np.array(labels)


def extract_graph_groups(dataset, attr: str = "subject_id"):
    """Per-graph group ids (e.g. subject_id) if present, else None."""
    try:
        first = dataset[0]
    except Exception:
        return None
    if not hasattr(first, attr):
        return None
    groups = []
    for idx in range(len(dataset)):
        val = getattr(dataset[idx], attr)
        try:
            val = val.view(-1)[0].item()
        except AttributeError:
            pass
        groups.append(int(val))
    return np.array(groups)


def stratified_split_indices(labels, split_sizes, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    train, eval_, test = [], [], []
    for label in np.unique(labels):
        li = indices[labels == label]
        rng.shuffle(li)
        n = len(li)
        n_tr = int(round(split_sizes[0] * n))
        n_ev = int(round(split_sizes[1] * n))
        if n_tr + n_ev > n:
            n_ev = max(0, n - n_tr)
        train += li[:n_tr].tolist()
        eval_ += li[n_tr:n_tr + n_ev].tolist()
        test += li[n_tr + n_ev:].tolist()
    rng.shuffle(train); rng.shuffle(eval_); rng.shuffle(test)
    return train, eval_, test


def subject_aware_split_indices(labels, groups, split_sizes, seed):
    """Assign whole subjects to one fold while balancing class prior."""
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)

    group_to_idx = {}
    for idx, g in enumerate(groups):
        group_to_idx.setdefault(int(g), []).append(idx)

    group_ids = list(group_to_idx.keys())
    rng.shuffle(group_ids)
    n_total = len(labels)
    target = {
        "train": split_sizes[0] * n_total,
        "eval": split_sizes[1] * n_total,
        "test": split_sizes[2] * n_total,
    }
    global_pos = labels.mean() if n_total else 0.0
    buckets = {k: {"idx": [], "n": 0, "pos": 0} for k in ("train", "eval", "test")}

    group_ids.sort(key=lambda g: len(group_to_idx[g]), reverse=True)
    for g in group_ids:
        idxs = group_to_idx[g]
        g_n = len(idxs)
        g_pos = int(labels[idxs].sum())
        best_split, best_score = None, None
        for k in ("train", "eval", "test"):
            if target[k] <= 0:
                continue
            b = buckets[k]
            room = (target[k] - b["n"]) / target[k]
            new_pos_rate = (b["pos"] + g_pos) / (b["n"] + g_n)
            score = room - 0.5 * abs(new_pos_rate - global_pos)
            if best_score is None or score > best_score:
                best_score, best_split = score, k
        b = buckets[best_split]
        b["idx"].extend(idxs); b["n"] += g_n; b["pos"] += g_pos

    train, eval_, test = (buckets[k]["idx"] for k in ("train", "eval", "test"))
    rng.shuffle(train); rng.shuffle(eval_); rng.shuffle(test)
    return train, eval_, test


def split_indices(dataset, split_sizes, seed, group_attr="subject_id"):
    """Subject-aware split when group ids are present, else stratified."""
    labels = extract_graph_labels(dataset)
    groups = extract_graph_groups(dataset, group_attr)
    if groups is not None and len(np.unique(groups)) < len(groups):
        return subject_aware_split_indices(labels, groups, split_sizes, seed)
    return stratified_split_indices(labels, split_sizes, seed)
