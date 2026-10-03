"""CEI-GNN v3 unit U2: absence universe and "no recorded result" derivation.

The universe is the fixed, train-only set of measurement/vital identities whose
presence is supported in at least ``min_graphs`` distinct graphs of the frozen
TRAIN sample (v3 spec §4.3, amended by §12 F9/F10). It is preprocessing state:
``state()`` / ``sha256()`` / ``load()`` bind it, and ``slot_of_token()`` gives the
model a ``token index -> universe slot`` buffer so no string comparison happens at
forward time.

Absence means "no recorded result at this visit" (§12 F8): no admitted node of the
identity is assigned by ``visit_membership_index`` to the graph's index visit. The
index visit is the last ordinal, ``cumsum(num_visits) - 1`` in batch coordinates
(§12 F9). The derivation reads existing tensors only; no data file is opened here.
"""
from __future__ import annotations

import hashlib
import json

import torch

UNIVERSE_VERSION = 'cei_v3_absence_universe_v1'
UNK = 0  # tensorize.Vocabulary reserves index 0 for unseen tokens.
ABSENCE_LABEL = '{item}: no recorded result at this visit'


def _canonical_json(state):
    return json.dumps(state, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


class Universe:
    """Frozen train-fitted set of measurement/vital identities (sorted)."""

    def __init__(self, items, token_index, *, min_graphs, num_tokens, graph_counts,
                 token_min_count=None):
        self.items = tuple(items)
        self.token_index = tuple(int(i) for i in token_index)
        self.min_graphs = int(min_graphs)
        self.token_min_count = None if token_min_count is None else int(token_min_count)
        self.num_tokens = int(num_tokens)
        self.graph_counts = {str(k): int(v) for k, v in graph_counts.items()}
        self._validate()

    def _validate(self):
        if len(self.items) != len(self.token_index):
            raise ValueError('Universe items and token indices differ in length')
        if not self.items:
            raise ValueError('Universe is empty')
        if list(self.items) != sorted(set(self.items)):
            raise ValueError('Universe items must be unique and sorted')
        if any(index <= UNK or index >= self.num_tokens for index in self.token_index):
            raise ValueError('Universe identity maps to UNK or outside the vocabulary')
        if len(set(self.token_index)) != len(self.token_index):
            raise ValueError('Universe token indices must be unique')
        if self.min_graphs < 1:
            raise ValueError('min_graphs must be positive')
        if set(self.graph_counts) != set(self.items):
            raise ValueError('Universe graph counts do not match its items')
        if any(count < self.min_graphs for count in self.graph_counts.values()):
            raise ValueError('Universe item has fewer graphs than min_graphs')
        if self.token_min_count is not None and self.min_graphs < self.token_min_count:
            raise ValueError('min_graphs must be at least token_min_count (F10)')

    def __len__(self):
        return len(self.items)

    def state(self):
        return {'version': UNIVERSE_VERSION,
                'items': list(self.items),
                'token_index': list(self.token_index),
                'min_graphs': self.min_graphs,
                'token_min_count': self.token_min_count,
                'num_tokens': self.num_tokens,
                'graph_counts': {item: self.graph_counts[item] for item in self.items}}

    def sha256(self):
        return hashlib.sha256(_canonical_json(self.state()).encode()).hexdigest()

    @classmethod
    def load(cls, state):
        if not isinstance(state, dict) or state.get('version') != UNIVERSE_VERSION:
            raise ValueError('Unsupported absence universe state version')
        try:
            return cls(state['items'], state['token_index'], min_graphs=state['min_graphs'],
                       num_tokens=state['num_tokens'], graph_counts=state['graph_counts'],
                       token_min_count=state.get('token_min_count'))
        except (KeyError, TypeError) as error:
            raise ValueError(f'Malformed absence universe state: {error}') from error

    def slot_of_token(self, num_tokens=None):
        """long[num_tokens]: universe slot of each token index, -1 when not in universe."""
        size = self.num_tokens if num_tokens is None else int(num_tokens)
        if size < self.num_tokens:
            raise ValueError('num_tokens is smaller than the vocabulary the universe was fitted on')
        slots = torch.full((size,), -1, dtype=torch.long)
        slots[torch.tensor(self.token_index, dtype=torch.long)] = torch.arange(
            len(self.token_index), dtype=torch.long)
        return slots


def fit_universe(identity_graph_counts, vocabulary, *, min_graphs=20, token_min_count=None):
    """Fit the universe from TRAIN-only distinct-graph counts per identity.

    ``vocabulary`` maps token string -> index as ``Vocabulary.index`` does (UNK = 0
    is never a member). Raises if a supported identity has no non-UNK index.
    """
    if isinstance(min_graphs, bool) or int(min_graphs) != min_graphs or min_graphs < 1:
        raise ValueError('min_graphs must be a positive integer')
    if token_min_count is not None and min_graphs < token_min_count:
        raise ValueError('min_graphs must be at least token_min_count (F10)')
    counts = {}
    for identity, count in identity_graph_counts.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f'Graph count of {identity!r} must be a non-negative integer')
        counts[str(identity)] = count
    items = sorted(identity for identity, count in counts.items() if count >= min_graphs)
    if not items:
        raise ValueError('Absence universe is empty; no identity reaches min_graphs')
    token_index = []
    for item in items:
        index = vocabulary.get(item, UNK)
        if not isinstance(index, int) or index == UNK:
            raise ValueError(f'Universe identity {item!r} maps to UNK; universe must be a subset '
                             'of the vocabulary')
        token_index.append(index)
    num_tokens = max(vocabulary.values()) + 1 if vocabulary else 1
    return Universe(items, token_index, min_graphs=int(min_graphs), num_tokens=num_tokens,
                    graph_counts={item: counts[item] for item in items},
                    token_min_count=token_min_count)


def index_visit_absence(membership, num_visits, node_type, token, slot_of_token,
                        measurement_kinds):
    """bool[G, U]: True where a universe item has no node at the graph's index visit.

    ``membership`` is ``visit_membership_index`` in batch coordinates (visits offset
    by ``num_visits``, nodes by node count, ``tensorize.ClinicalGraphData.__inc__``).
    The index visit of graph g is ``cumsum(num_visits)[g] - 1`` (§12 F9). Presence is
    membership-derived only: an index-visit node with an invalid value counts as
    present, and several nodes of one identity count once. Only nodes whose kind is
    in ``measurement_kinds`` and whose token maps to a universe slot are considered.
    """
    device = node_type.device
    slot_of_token = slot_of_token.to(device=device, dtype=torch.long)
    universe_size = int((slot_of_token >= 0).sum())
    num_visits = num_visits.to(device=device, dtype=torch.long).view(-1)
    graph_count = int(num_visits.numel())
    if graph_count and bool((num_visits < 1).any()):
        raise ValueError('num_visits must contain one positive count per graph')
    absent = torch.ones((graph_count, universe_size), dtype=torch.bool, device=device)
    if graph_count == 0:
        return absent
    if membership.ndim != 2 or membership.size(0) != 2:
        raise ValueError('visit_membership_index must have shape [2, pairs]')
    membership = membership.to(device=device, dtype=torch.long)
    visit, node = membership[0], membership[1]
    node_count = int(node_type.numel())
    index_visit = torch.cumsum(num_visits, 0) - 1
    if node.numel():
        if int(node.min()) < 0 or int(node.max()) >= node_count or int(visit.min()) < 0:
            raise ValueError('visit membership refers to a node or visit outside the batch')
        if int(visit.max()) > int(index_visit[-1]):
            raise ValueError('visit membership refers to a visit outside the batch')
    # Membership contract (§4.3): every measurement/vital node has exactly one
    # visit membership; missing or ambiguous membership fails, it is never inferred.
    kinds = torch.tensor(tuple(measurement_kinds), dtype=torch.long, device=device)
    item_nodes = torch.isin(node_type, kinds)
    membership_count = torch.bincount(node, minlength=node_count)
    if bool((item_nodes & (membership_count == 0)).any()):
        raise ValueError('measurement/vital node has no visit membership')
    if bool((item_nodes & (membership_count > 1)).any()):
        raise ValueError('measurement/vital node has ambiguous visit membership (more than one)')
    if node.numel() == 0:
        return absent
    # Visit -> graph, then keep only pairs on their graph's index visit.
    visit_graph = torch.repeat_interleave(torch.arange(graph_count, device=device), num_visits)
    graph_of_pair = visit_graph[visit]
    at_index = visit == index_visit[graph_of_pair]
    is_kind = item_nodes[node]
    slot = slot_of_token[token.to(device=device, dtype=torch.long)[node]]
    keep = at_index & is_kind & (slot >= 0)
    if bool(keep.any()):
        absent[graph_of_pair[keep], slot[keep]] = False
    return absent
