"""CEI-GNN v3 extension E6b: additive comorbid pair term (extensions spec §5.3, §9 X5).

Plugged into ``EvidenceNetworkV3`` through the U3x ``extra_blocks`` protocol
(``methods/cei_gnn_v3.py``); the v3 core file is not edited. The adapter imports this
module lazily for ``comorbid_block=1`` and calls ``build_block``.

Pair set (§5.3): every forward-list edge whose relation is ``comorbid_with`` — the id is
resolved by NAME from the bound relation layout, never the literal index (E15) — maps to
the unordered pair ``(min(src, dst), max(src, dst))``; both directions and every repeated
encounter collapse to one pair; pairs never cross graphs.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..methods.cei_gnn_v3 import EXTRA_BLOCK_PREFIX

BLOCK_NAME = 'comorbid'
COMORBID_RELATION = 'comorbid_with'
PAIR_RANK = 16

__all__ = ['BLOCK_NAME', 'COMORBID_RELATION', 'PAIR_RANK', 'ComorbidPairBlock',
           'build_block', 'comorbid_pairs']


def comorbid_pairs(edge_index, edge_relation, relation_id, batch_index):
    """Unique unordered pairs ``long[2, pairs]`` (row 0 < row 1) of the edges whose
    relation equals ``relation_id``; pairs are ordered by ``(min, max)`` node id.

    Raises when ``edge_relation`` does not align with ``edge_index`` or when a selected
    edge joins two graphs of the batch.
    """
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError('edge_index must have shape [2, edges]')
    device = edge_index.device
    edge_index = edge_index.long()
    edge_relation = torch.as_tensor(edge_relation, device=device).long().reshape(-1)
    batch_index = torch.as_tensor(batch_index, device=device).long().reshape(-1)
    if edge_relation.numel() != edge_index.size(1):
        raise ValueError('edge_relation must have one entry per edge of edge_index')
    selected = edge_index[:, edge_relation == int(relation_id)]
    if selected.size(1) == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    if int(selected.min()) < 0 or int(selected.max()) >= batch_index.numel():
        raise ValueError('comorbid edge refers to a node outside the batch')
    if not torch.equal(batch_index[selected[0]], batch_index[selected[1]]):
        raise ValueError('comorbid pairs must not cross graph boundaries')
    low = torch.minimum(selected[0], selected[1])
    high = torch.maximum(selected[0], selected[1])
    node_count = int(batch_index.numel())
    key = torch.unique(low * node_count + high)   # sorted, deduplicated (min, max) keys
    return torch.stack((key // node_count, key % node_count)).contiguous()


class ComorbidPairBlock(nn.Module):
    """Additive comorbid pair block (E6b) satisfying the U3x ``ExtraBlock`` protocol."""

    name = BLOCK_NAME
    uses_rng = False

    def __init__(self, hidden, num_classes, *, relation_id, seed):
        super().__init__()
        self.hidden, self.num_classes = int(hidden), int(num_classes)
        self.relation_id, self.seed = int(relation_id), int(seed)

    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(f'{EXTRA_BLOCK_PREFIX}.{self.name}.{local}'
                     for local, _ in self.named_parameters())

    def forward(self, h, edge_index, edge_relation, batch_index, graph_count):
        graphs, classes = int(graph_count), self.num_classes
        pairs = comorbid_pairs(edge_index, edge_relation, self.relation_id, batch_index)
        parts: Dict[str, torch.Tensor] = {
            'comorbid_contributions': h.new_zeros((pairs.size(1), classes)),
            'comorbid_pairs': pairs,
            'comorbid_gates': h.new_zeros((classes,)),
            'comorbid_denominator': h.new_ones((graphs, classes)),
        }
        return h.new_zeros((graphs, classes)), parts


def build_block(hidden, num_classes, *, relation_layout: Dict[str, int], seed) -> ComorbidPairBlock:
    """Adapter entry point (§9 X5): the comorbid relation id is resolved by NAME (E15)."""
    layout = dict(relation_layout)
    if COMORBID_RELATION not in layout:
        raise ValueError(f'relation_layout has no {COMORBID_RELATION!r} relation; the comorbid '
                         'block needs its id resolved by name from the bound layout')
    return ComorbidPairBlock(hidden, num_classes, relation_id=layout[COMORBID_RELATION], seed=seed)
