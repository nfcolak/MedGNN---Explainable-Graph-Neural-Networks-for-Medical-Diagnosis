"""CEI-GNN v3 core: v2 additive evidence + PLE value basis + "no recorded result" block.

Spec: v3 design §4.2–§4.5 as amended by §12 (F3, F8, F9, F11, F12, F13) and extensions
spec §9 U3. The network subclasses ``PairEvidenceNetwork`` with ``pair_mode='additive'``
fixed and keeps every v2 parameter name and shape, so a v2 additive state_dict loads with
``strict=False`` (F12). The v2 forward body is reused unchanged: the PLE term enters as an
additive pre-activation term ``node_encoder(cat(x, token, type)) + ple_projection(basis)``
through a thin ``nn.Linear`` subclass that adds a pending term (F12), and the absence block
is added to the returned logits afterwards.

Arms (§4.4 / F3): ``A`` = both new paths inactive, ``B`` = PLE only, ``C`` = PLE and
absence. An inactive path is skipped entirely: no tensor derived from its parameters
enters the autograd graph, so those parameters keep ``grad=None``.

RNG (F11): the v2 tensors are initialised exactly as ``PairEvidenceNetwork`` does; every
new tensor is initialised from its own ``torch.Generator`` seeded from
``(seed, qualified tensor name)``, and no RNG operation runs inside the new paths.
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from .. import NODE_KINDS
from ..cei_v3_absence import ABSENCE_LABEL, index_visit_absence
from ..cei_v3_ple import ple_basis
from .cei_gnn_v2 import PairEvidenceNetwork

ARMS = ('A', 'B', 'C')
MEASUREMENT_KINDS = tuple(NODE_KINDS.index(kind) for kind in ('measurement', 'vital'))
NEW_PARAMETER_NAMES = ('ple_projection.weight', 'absence_vote', 'absence_gate')
ABSENCE_KEYS = ('absence_contributions', 'absence_items', 'absence_gates',
                'absence_denominator')

__all__ = ['ARMS', 'MEASUREMENT_KINDS', 'NEW_PARAMETER_NAMES', 'ABSENCE_KEYS', 'ABSENCE_LABEL',
           'EvidenceNetworkV3', 'tensor_generator', 'tensor_seed']


def tensor_seed(seed, name):
    """Deterministic 63-bit generator seed of one qualified tensor name (F11)."""
    digest = hashlib.sha256(f'{int(seed)}:{name}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') & 0x7FFF_FFFF_FFFF_FFFF


def tensor_generator(seed, name):
    return torch.Generator().manual_seed(tensor_seed(seed, name))


class _EncoderWithPreActivationTerm(nn.Linear):
    """``nn.Linear`` that adds a pending additive term to its pre-activation output.

    Registered under the v2 name ``node_encoder`` so the parameter names and shapes are
    unchanged; when no term is pending it is exactly ``nn.Linear`` (arm A parity).
    """

    _pending_term = None

    def forward(self, input):
        output = super().forward(input)
        term = self._pending_term
        if term is None:
            return output
        return output + term


class EvidenceNetworkV3(PairEvidenceNetwork):
    """v3 superset network: v2 additive + PLE projection + absence votes/gates."""

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden, token_dim,
                 num_triples, num_relations, dropout, pair_rank, num_node_types, arm, knots,
                 knot_active, slot_of_token, universe_size, feature_layout, seed,
                 encoder_depth=1, extra_blocks=(), knot_row_of_token=None):
        if arm not in ARMS:
            raise ValueError(f'arm must be one of {list(ARMS)}, got {arm!r}')
        if isinstance(encoder_depth, bool) or int(encoder_depth) != encoder_depth or encoder_depth != 1:
            raise ValueError('encoder_depth != 1 is added by U3x; U3 supports encoder_depth=1 only')
        if tuple(extra_blocks):
            raise ValueError('extra_blocks are added by U3x; U3 accepts an empty tuple only')
        super().__init__(num_tokens=num_tokens, node_dim=node_dim, edge_dim=edge_dim,
                         num_classes=num_classes, hidden=hidden, token_dim=token_dim,
                         num_triples=num_triples, num_relations=num_relations, dropout=dropout,
                         pair_rank=pair_rank, pair_mode='additive',
                         num_node_types=num_node_types)
        self.arm, self.seed, self.encoder_depth = str(arm), int(seed), 1
        self.ple_active, self.absence_active = arm != 'A', arm == 'C'
        layout = [str(name) for name in feature_layout]
        if len(layout) != self.node_dim:
            raise ValueError('node_feature_layout length differs from node_dim')
        for column in ('scaled_value', 'has_value'):
            if layout.count(column) != 1:
                raise ValueError(f'node_feature_layout must contain exactly one {column!r} column')
        self.feature_layout = tuple(layout)
        self.scaled_value_column = layout.index('scaled_value')   # F13: by name, never 8
        self.has_value_column = layout.index('has_value')          # F13: by name, never 9

        # Re-wrap the v2 encoder without touching the RNG: same Parameter objects, same name.
        encoder = nn.utils.skip_init(_EncoderWithPreActivationTerm,
                                     in_features=self.node_encoder.in_features,
                                     out_features=self.node_encoder.out_features)
        encoder.weight, encoder.bias = self.node_encoder.weight, self.node_encoder.bias
        self.node_encoder = encoder

        knots = torch.as_tensor(knots, dtype=torch.float32)
        knot_active = torch.as_tensor(knot_active, dtype=torch.bool)
        if knots.ndim != 2 or knot_active.shape != (knots.shape[0],):
            raise ValueError('knots must be float32[items, K+1] with knot_active bool[items]')
        self.K = int(knots.shape[1]) - 1
        if knot_row_of_token is None:
            if self.ple_active and knots.shape[0]:
                raise ValueError('knot_row_of_token is required when the PLE path is active')
            knot_row_of_token = torch.full((self.num_tokens,), -1, dtype=torch.long)
        knot_row_of_token = torch.as_tensor(knot_row_of_token, dtype=torch.long).reshape(-1)
        if knot_row_of_token.numel() != self.num_tokens:
            raise ValueError('knot_row_of_token must have one entry per token')
        if bool((knot_row_of_token < -1).any()) or bool((knot_row_of_token >= knots.shape[0]).any()):
            raise ValueError('knot_row_of_token refers to a row outside the knot table')
        slot_of_token = torch.as_tensor(slot_of_token, dtype=torch.long).reshape(-1)
        if slot_of_token.numel() != self.num_tokens:
            raise ValueError('slot_of_token must have one entry per token')
        self.universe_size = int(universe_size)
        if int((slot_of_token >= 0).sum()) != self.universe_size:
            raise ValueError('universe_size differs from the number of tokens with a universe slot')
        if self.universe_size and int(slot_of_token.max()) >= self.universe_size:
            raise ValueError('slot_of_token refers to a slot outside the universe')
        # Preprocessing state, not parameters (§4.4): bound through the v3_state file hash
        # in run_config, so the checkpoint stays a parameter-only superset of v2's.
        self.register_buffer('knots', knots, persistent=False)
        self.register_buffer('knot_active', knot_active, persistent=False)
        self.register_buffer('knot_row_of_token', knot_row_of_token, persistent=False)
        self.register_buffer('slot_of_token', slot_of_token, persistent=False)
        self.register_buffer('measurement_kinds', torch.tensor(MEASUREMENT_KINDS, dtype=torch.long),
                             persistent=False)

        # New tensors (registered in every arm; names/shapes identical across arms).
        self.ple_projection = nn.utils.skip_init(nn.Linear, in_features=self.K + 1,
                                                 out_features=self.hidden, bias=False)
        bound = 1.0 / float(self.K + 1) ** 0.5   # nn.Linear's default uniform bound
        with torch.no_grad():
            self.ple_projection.weight.uniform_(
                -bound, bound, generator=tensor_generator(self.seed, 'ple_projection.weight'))
        # Zero-initialised votes and gate logits (v2 `pair_gate`/`bias` pattern; F3).
        self.absence_vote = nn.Parameter(torch.zeros(self.universe_size, self.num_classes))
        self.absence_gate = nn.Parameter(torch.zeros(self.universe_size, self.num_classes))

    # ------------------------------------------------------------------ new paths

    def _ple_term(self, features, metadata):
        """PLE pre-activation term float32[N, hidden]; only called when PLE is active."""
        values = features[:, self.scaled_value_column]
        has_value = features[:, self.has_value_column]
        knot_row = self.knot_row_of_token[metadata.token]
        basis = ple_basis(values, has_value, knot_row, self.knots, self.knot_active)
        return self.ple_projection(basis)

    def _absence_block(self, features, metadata, membership, visit_graph):
        """Absence block of the batch; only called when the absence path is active.

        Returns ``(total float32[G, C], parts dict)`` with the four absence keys. An empty
        absent set yields an exact-zero total, ``[0, C]`` parts and denominator one (§4.5).
        """
        graph_count, classes = int(metadata.graph_count), self.num_classes
        device = features.device
        if visit_graph is None:
            raise ValueError('visit_graph is required for the absence block (index visit = '
                             'cumsum(num_visits) - 1 per graph, F9)')
        visit_graph = visit_graph.to(device=device, dtype=torch.long).view(-1)
        if visit_graph.numel() and bool((visit_graph[1:] < visit_graph[:-1]).any()):
            raise ValueError('visit_graph must list visits in graph order')
        num_visits = torch.bincount(visit_graph, minlength=graph_count)
        if num_visits.numel() != graph_count:
            raise ValueError('visit_graph refers to a graph outside the batch')
        absent = index_visit_absence(membership, num_visits, metadata.node_type, metadata.token,
                                     self.slot_of_token, MEASUREMENT_KINDS)
        items = absent.nonzero(as_tuple=False).t().contiguous()   # [2, n]: (graph, slot)
        graph, slot = items[0], items[1]
        denominator = features.new_ones((graph_count, classes))
        if items.size(1) == 0:
            parts = {'absence_contributions': features.new_zeros((0, classes)),
                     'absence_items': items, 'absence_gates': features.new_zeros((0, classes)),
                     'absence_denominator': denominator}
            return features.new_zeros((graph_count, classes)), parts
        gate = self.absence_gate[slot].sigmoid()                   # [n, C]
        vote = self.absence_vote[slot]                             # [n, C]
        gated = gate * vote
        denominator = denominator.index_add(0, graph, gate)        # 1 + Σ_absent g
        total = features.new_zeros((graph_count, classes)).index_add(0, graph, gated) / denominator
        parts = {'absence_contributions': gated / denominator[graph], 'absence_items': items,
                 'absence_gates': gate, 'absence_denominator': denominator}
        return total, parts

    def forward_continuous(self, features, edge_index, metadata, membership, *,
                           return_parts=False, visit_graph=None):
        if features.ndim == 2 and features.size(1) == self.continuous_width and self.ple_active:
            self.node_encoder._pending_term = self._ple_term(features, metadata)
        try:
            result = super().forward_continuous(features, edge_index, metadata, membership,
                                                return_parts=return_parts, visit_graph=visit_graph)
        finally:
            self.node_encoder._pending_term = None
        graph_count, classes = int(metadata.graph_count), self.num_classes
        if self.absence_active:
            total, absence_parts = self._absence_block(features, metadata, membership, visit_graph)
        else:  # skipped entirely (F3): no tensor derived from the absence parameters
            total = None
            absence_parts = {
                'absence_contributions': features.new_zeros((0, classes)),
                'absence_items': torch.zeros((2, 0), dtype=torch.long, device=features.device),
                'absence_gates': features.new_zeros((0, classes)),
                'absence_denominator': features.new_ones((graph_count, classes))}
        if not return_parts:
            return result if total is None else result + total
        if total is not None:
            result['logits'] = result['logits'] + total
        result.update(absence_parts)
        return result


def absence_label(item):
    """Explanation label of an absent universe item (v3 §12 F8 wording)."""
    return ABSENCE_LABEL.format(item=item)
