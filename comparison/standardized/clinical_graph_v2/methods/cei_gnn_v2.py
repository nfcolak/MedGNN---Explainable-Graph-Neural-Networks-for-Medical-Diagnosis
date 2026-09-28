"""CEI-GNN v2: node, endpoint-additive edge and within-visit pair evidence."""
from __future__ import annotations

import torch

from .. import NODE_KINDS

EVIDENCE_KINDS = ("complaint", "measurement", "vital")
EVIDENCE_KIND_IDS = tuple(NODE_KINDS.index(kind) for kind in EVIDENCE_KINDS)
KIND_PAIR_COUNT = 0
PAIR_MODES = ("product", "additive", "off")


def within_visit_pairs(membership, node_type, node_count):
    return torch.zeros((2, 0), dtype=torch.long, device=node_type.device)


def kind_pair_index(node_type, pairs):
    return torch.zeros((pairs.size(1),), dtype=torch.long, device=node_type.device)
