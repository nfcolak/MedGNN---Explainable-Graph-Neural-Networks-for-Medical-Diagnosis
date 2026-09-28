"""CEI-GNN v2: node, endpoint-additive edge and within-visit pair evidence."""
from __future__ import annotations

import torch

from .. import NODE_KINDS

EVIDENCE_KINDS = ("complaint", "measurement", "vital")
EVIDENCE_KIND_IDS = tuple(NODE_KINDS.index(kind) for kind in EVIDENCE_KINDS)
KIND_PAIR_COUNT = len(EVIDENCE_KINDS) * (len(EVIDENCE_KINDS) + 1) // 2
PAIR_MODES = ("product", "additive", "off")


def _empty_pairs(device):
    return torch.zeros((2, 0), dtype=torch.long, device=device)


def within_visit_pairs(membership, node_type, node_count):
    """Unique unordered evidence-node pairs (i < j) that share at least one visit.

    Membership rows are (visit id, node id) after PyG batching, so visit ids are
    unique across graphs and pairs never cross graphs. Deterministic; no sampling.
    """
    device = node_type.device
    if membership.ndim != 2 or membership.size(0) != 2:
        raise ValueError("visit_membership_index must have shape [2, pairs]")
    if membership.size(1) == 0:
        return _empty_pairs(device)
    membership = membership.to(device=device, dtype=torch.long)
    visit, node = membership[0], membership[1]
    if int(node.min()) < 0 or int(node.max()) >= int(node_count) or int(visit.min()) < 0:
        raise ValueError("visit membership refers to a node outside the batch")
    evidence = torch.tensor(EVIDENCE_KIND_IDS, device=device)
    keep = torch.isin(node_type[node], evidence)
    visit, node = visit[keep], node[keep]
    if node.numel() < 2:
        return _empty_pairs(device)
    count = int(node_count)
    key = torch.unique(visit * count + node)
    visit, node = key // count, key % count
    _, sizes = torch.unique_consecutive(visit, return_counts=True)
    width = int(sizes.max())
    if width < 2:
        return _empty_pairs(device)
    group = torch.arange(sizes.numel(), device=device).repeat_interleave(sizes)
    starts = torch.cumsum(sizes, 0) - sizes
    position = torch.arange(node.numel(), device=device) - starts.repeat_interleave(sizes)
    dense = node.new_full((sizes.numel(), width), -1)
    dense[group, position] = node
    left_slot, right_slot = torch.triu_indices(width, width, offset=1, device=device)
    left, right = dense[:, left_slot], dense[:, right_slot]
    valid = (left >= 0) & (right >= 0)
    pair_key = torch.unique(left[valid] * count + right[valid])
    return torch.stack((pair_key // count, pair_key % count))


def kind_pair_index(node_type, pairs):
    """Index 0..5 of the unordered evidence-kind pair of each node pair."""
    device = node_type.device
    lookup = torch.full((len(NODE_KINDS),), -1, dtype=torch.long, device=device)
    lookup[torch.tensor(EVIDENCE_KIND_IDS, device=device)] = torch.arange(
        len(EVIDENCE_KINDS), device=device)
    first, second = lookup[node_type[pairs[0]]], lookup[node_type[pairs[1]]]
    if bool((first < 0).any()) or bool((second < 0).any()):
        raise ValueError("pair endpoint is not an evidence node")
    low, high = torch.minimum(first, second), torch.maximum(first, second)
    kinds = len(EVIDENCE_KINDS)
    return low * kinds - low * (low - 1) // 2 + (high - low)
