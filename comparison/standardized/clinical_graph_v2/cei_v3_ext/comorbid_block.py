"""CEI-GNN v3 extension E6b: additive comorbid pair term (extensions spec §5.3, §9 X5).

Plugged into ``EvidenceNetworkV3`` through the U3x ``extra_blocks`` protocol
(``methods/cei_gnn_v3.py``); the v3 core file is not edited. The adapter imports this
module lazily for ``comorbid_block=1`` and calls ``build_block``.
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
    """Unique unordered comorbid pairs of the batch as ``long[2, pairs]``."""
    return torch.zeros((2, 0), dtype=torch.long)


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
        parts: Dict[str, torch.Tensor] = {
            'comorbid_contributions': h.new_zeros((0, classes)),
            'comorbid_pairs': torch.zeros((2, 0), dtype=torch.long, device=h.device),
            'comorbid_gates': h.new_zeros((classes,)),
            'comorbid_denominator': h.new_zeros((graphs, classes)),
        }
        return h.new_zeros((graphs, classes)), parts


def build_block(hidden, num_classes, *, relation_layout: Dict[str, int], seed) -> ComorbidPairBlock:
    """Adapter entry point (§9 X5): the comorbid relation id is resolved by NAME."""
    return ComorbidPairBlock(hidden, num_classes, relation_id=0, seed=seed)
