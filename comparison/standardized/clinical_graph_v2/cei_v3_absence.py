"""CEI-GNN v3 unit U2: absence universe and "no recorded result" derivation.

Stub: the universe is fitted from nothing and every item is reported absent.
"""
from __future__ import annotations

import torch

UNIVERSE_VERSION = 'cei_v3_absence_universe_v1'


class Universe:
    """Frozen train-fitted set of measurement/vital identities."""

    def __init__(self, items=(), token_index=(), *, min_graphs=20, token_min_count=None,
                 num_tokens=1, graph_counts=None):
        self.items = tuple(items)
        self.token_index = tuple(token_index)
        self.min_graphs = min_graphs
        self.token_min_count = token_min_count
        self.num_tokens = num_tokens
        self.graph_counts = dict(graph_counts or {})

    def __len__(self):
        return 0

    def state(self):
        return {'version': UNIVERSE_VERSION, 'items': [], 'token_index': [],
                'min_graphs': self.min_graphs, 'token_min_count': self.token_min_count,
                'num_tokens': self.num_tokens, 'graph_counts': {}}

    def sha256(self):
        return ''

    @classmethod
    def load(cls, state):
        return cls()

    def slot_of_token(self, num_tokens=None):
        return torch.zeros(0, dtype=torch.long)


def fit_universe(identity_graph_counts, vocabulary, *, min_graphs=20, token_min_count=None):
    return Universe()
