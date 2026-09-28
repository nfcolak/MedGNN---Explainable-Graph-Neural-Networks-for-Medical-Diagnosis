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


import torch.nn as nn


class PairEvidenceNetwork(nn.Module):
    """Temporary public-interface stub; intentionally wrong for red tests."""

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden, token_dim,
                 num_triples, num_relations, dropout, pair_rank, pair_mode, num_node_types):
        super().__init__()
        if pair_mode not in PAIR_MODES:
            raise ValueError("pair_mode must be one of pair_mode")
        self.node_dim, self.edge_dim = node_dim, edge_dim
        self.num_classes, self.hidden, self.token_dim = num_classes, hidden, token_dim
        self.pair_mode, self.pair_rank = pair_mode, pair_rank
        self.token_embedding = nn.Embedding(num_tokens, token_dim)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden)
        self.node_encoder = nn.Linear(node_dim + token_dim + hidden, hidden)
        self.node_norm = nn.LayerNorm(hidden)
        self.node_head = nn.Linear(hidden, num_classes * 2)
        self.relation_embedding = nn.Embedding(num_relations, hidden)
        self.triple_embedding = nn.Embedding(num_triples, hidden)
        self.edge_feature_projection = nn.Linear(edge_dim, hidden, bias=False)
        self.edge_source = nn.Linear(hidden, num_classes)
        self.edge_target = nn.Linear(hidden, num_classes)
        self.edge_context_vote = nn.Linear(hidden, num_classes)
        self.edge_context_gate = nn.Linear(hidden, num_classes)
        self.pair_projection = nn.Linear(hidden, pair_rank, bias=False)
        self.pair_vote = nn.Linear(pair_rank, num_classes)
        self.pair_gate = nn.Parameter(torch.zeros(KIND_PAIR_COUNT, num_classes))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self._edge_mask = None

    @property
    def continuous_width(self):
        return self.node_dim + self.token_dim + self.hidden

    def continuous_inputs(self, clinical):
        return torch.zeros((clinical.x.size(0), self.continuous_width), device=clinical.x.device)

    def set_edge_mask(self, mask):
        self._edge_mask = mask

    def forward_continuous(self, features, edge_index, metadata, membership, *, return_parts=False):
        if not torch.isfinite(features).all():
            raise ValueError("continuous features must be finite")
        pairs = within_visit_pairs(membership, metadata.node_type, features.size(0))
        graph_count, edge_count = int(metadata.graph_count), edge_index.size(1)
        node_parts = features.new_zeros((features.size(0), self.num_classes))
        edge_parts = features.new_zeros((edge_count, self.num_classes))
        pair_parts = features.new_zeros((pairs.size(1), self.num_classes))
        if edge_count and features.size(0) > 2:
            edge_parts[0] = (features[1, 0] * features[2, 0]).expand(self.num_classes)
        if self._edge_mask is not None and not bool(self._edge_mask.any()) and edge_count:
            edge_parts = torch.ones_like(edge_parts)
        logits = (features.sum() * 0.0 + self.bias.sum() * 0.0 + 1.0).expand(
            graph_count, self.num_classes).clone()
        if return_parts:
            return {"logits": logits, "node_contributions": node_parts,
                    "edge_contributions": edge_parts, "pair_contributions": pair_parts,
                    "pairs": pairs, "bias": self.bias}
        return logits

