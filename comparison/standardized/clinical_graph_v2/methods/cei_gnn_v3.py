"""CEI-GNN v3 core: v2 additive evidence + PLE value basis + "no recorded result" block.

Stub (red step): compiles and returns wrong values; the green step implements it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .cei_gnn_v2 import PairEvidenceNetwork

ARMS = ('A', 'B', 'C')


class EvidenceNetworkV3(PairEvidenceNetwork):
    """v3 superset network (stub)."""

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden, token_dim,
                 num_triples, num_relations, dropout, pair_rank, num_node_types, arm, knots,
                 knot_active, slot_of_token, universe_size, feature_layout, seed,
                 encoder_depth=1, extra_blocks=(), knot_row_of_token=None):
        super().__init__(num_tokens=num_tokens, node_dim=node_dim, edge_dim=edge_dim,
                         num_classes=num_classes, hidden=hidden, token_dim=token_dim,
                         num_triples=num_triples, num_relations=num_relations, dropout=dropout,
                         pair_rank=pair_rank, pair_mode='additive',
                         num_node_types=num_node_types)
        self.arm = arm
        self.ple_projection = nn.Linear(int(knots.shape[1]), int(hidden))
        self.scaled_value_column = 8
        self.has_value_column = 9

    def forward_continuous(self, features, edge_index, metadata, membership, *,
                           return_parts=False, visit_graph=None):
        graph_count, classes = int(metadata.graph_count), self.num_classes
        logits = features.new_zeros((graph_count, classes))
        if not return_parts:
            return logits
        return {'logits': logits, 'node_contributions': features.new_zeros((features.size(0), classes)),
                'edge_contributions': features.new_zeros((edge_index.size(1), classes)),
                'pair_contributions': features.new_zeros((0, classes)),
                'pairs': torch.zeros((2, 0), dtype=torch.long), 'bias': features.new_zeros((classes,)),
                'pair_gates': features.new_zeros((0,)),
                'pair_denominator': features.new_ones((graph_count, classes))}
