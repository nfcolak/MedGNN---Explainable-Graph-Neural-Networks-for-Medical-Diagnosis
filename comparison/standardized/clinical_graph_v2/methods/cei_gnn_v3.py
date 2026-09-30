"""CEI-GNN v3 core: v2 additive evidence + PLE value basis + "no recorded result" block.

Spec: v3 design §4.2–§4.5 as amended by §12 (F3, F8, F9, F11, F12, F13) and extensions
spec §9 U3. The network subclasses ``PairEvidenceNetwork`` with ``pair_mode='additive'``
fixed and keeps every v2 parameter name and shape, so a v2 additive state_dict loads with
``strict=False`` (F12). The forward reproduces the v2 body operation by operation: the PLE
term enters as an additive pre-activation term ``node_encoder(cat(x, token, type)) +
ple_projection(basis)`` (F12), the U3x hooks act on ``h`` before it is read (encoder depth,
extensions spec §2.2) and after the v2 sum (``extra_blocks``, §9 U3x), and the absence block
is added to the logits.

Arms (§4.4 / F3): ``A`` = both new paths inactive, ``B`` = PLE only, ``C`` = PLE and
absence. An inactive path is skipped entirely: no tensor derived from its parameters
enters the autograd graph, so those parameters keep ``grad=None``.

RNG (F11): the v2 tensors are initialised exactly as ``PairEvidenceNetwork`` does; every
new tensor is initialised from its own ``torch.Generator`` seeded from
``(seed, qualified tensor name)``, and no RNG operation runs inside the new paths.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, Tuple, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import NODE_KINDS
from ..cei_v3_absence import ABSENCE_LABEL, index_visit_absence
from ..cei_v3_ple import ple_basis
from .cei_gnn_v2 import PairEvidenceNetwork, kind_pair_index, within_visit_pairs

ARMS = ('A', 'B', 'C')
MEASUREMENT_KINDS = tuple(NODE_KINDS.index(kind) for kind in ('measurement', 'vital'))
NEW_PARAMETER_NAMES = ('ple_projection.weight', 'absence_vote', 'absence_gate')
ABSENCE_KEYS = ('absence_contributions', 'absence_items', 'absence_gates',
                'absence_denominator')
EXTRA_BLOCK_PREFIX = 'extra_blocks'
RESERVED_BLOCK_NAMES = frozenset(('node', 'edge', 'pair', 'absence'))

__all__ = ['ARMS', 'MEASUREMENT_KINDS', 'NEW_PARAMETER_NAMES', 'ABSENCE_KEYS', 'ABSENCE_LABEL',
           'EXTRA_BLOCK_PREFIX', 'RESERVED_BLOCK_NAMES', 'ExtraBlock', 'EvidenceNetworkV3',
           'tensor_generator', 'tensor_seed']


@runtime_checkable
class ExtraBlock(Protocol):
    """Additive logit block plugged into ``EvidenceNetworkV3`` (extensions spec §9 U3x).

    Contract (X5's ``comorbid_block.build_block`` returns one of these):

    * ``name`` — identifier, no ``.``, not in ``RESERVED_BLOCK_NAMES``, unique per network;
      the block is registered as ``extra_blocks.<name>`` and its parameters appear as
      ``extra_blocks.<name>.<local>`` in ``state_dict``/``parameter_inventory``.
    * ``uses_rng`` — must be ``False``: the block draws no dropout mask or other random
      number from the global stream (the network asserts the global RNG state is unchanged
      by the call; a dedicated ``torch.Generator`` is allowed). Its tensors are initialised
      by the block itself from ``tensor_generator(seed, 'extra_blocks.<name>.<local>')``.
    * ``parameter_names()`` — exactly the qualified names of its registered parameters.
    * ``__call__(h, edge_index, edge_relation, batch_index, graph_count)`` with
      ``h: float32[N, hidden]`` (after the encoder and the residual blocks),
      ``edge_index: long[2, E]``, ``edge_relation: long[E]``, ``batch_index: long[N]``,
      ``graph_count: int`` → ``(total: float32[G, C], parts: dict[str, Tensor])``. ``total``
      is added to the logit sum; every ``parts`` key must start with ``'<name>_'`` and must
      not collide with a v2/absence/other-block key.

    Blocks must be ``nn.Module`` instances (registration) that also satisfy this protocol.
    """

    name: str
    uses_rng: bool

    def parameter_names(self) -> Tuple[str, ...]: ...

    def __call__(self, h, edge_index, edge_relation, batch_index, graph_count): ...


V2_PART_KEYS = frozenset(('logits', 'node_contributions', 'edge_contributions',
                          'pair_contributions', 'pairs', 'bias', 'pair_gates', 'pair_denominator'))


def tensor_seed(seed, name):
    """Deterministic 63-bit generator seed of one qualified tensor name (F11)."""
    digest = hashlib.sha256(f'{int(seed)}:{name}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') & 0x7FFF_FFFF_FFFF_FFFF


def tensor_generator(seed, name):
    return torch.Generator().manual_seed(tensor_seed(seed, name))


class _ResidualBlock(nn.Module):
    """Node-local residual block ``h + Dropout(GELU(LayerNorm(Linear(h))))`` (§2.2).

    The Linear weight and bias are initialised from per-tensor generators (E6); LayerNorm
    starts at the deterministic ``nn.LayerNorm`` values. The dropout mask is drawn from the
    block's own generator seeded by ``(seed, '<qualified name>.dropout')``, so the global
    RNG stream is consumed exactly as in arm C.
    """

    def __init__(self, width, *, dropout, seed, name):
        super().__init__()
        self.linear = nn.utils.skip_init(nn.Linear, in_features=width, out_features=width)
        self.norm = nn.utils.skip_init(nn.LayerNorm, normalized_shape=width)
        self.dropout_rate = float(dropout)
        bound = 1.0 / float(width) ** 0.5   # nn.Linear's default uniform bound (fan_in = width)
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound,
                                        generator=tensor_generator(seed, f'{name}.linear.weight'))
            self.linear.bias.uniform_(-bound, bound,
                                      generator=tensor_generator(seed, f'{name}.linear.bias'))
            self.norm.weight.fill_(1.0)
            self.norm.bias.zero_()
        self._dropout_generator = tensor_generator(seed, f'{name}.dropout')

    def forward(self, h):
        update = F.gelu(self.norm(self.linear(h)))
        if self.training and self.dropout_rate > 0.0:
            keep = torch.rand(update.shape, generator=self._dropout_generator) >= self.dropout_rate
            update = update * keep.to(device=update.device, dtype=update.dtype) / (1.0 - self.dropout_rate)
        return h + update


class EvidenceNetworkV3(PairEvidenceNetwork):
    """v3 superset network: v2 additive + PLE projection + absence votes/gates."""

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden, token_dim,
                 num_triples, num_relations, dropout, pair_rank, num_node_types, arm, knots,
                 knot_active, slot_of_token, universe_size, feature_layout, seed,
                 encoder_depth=1, extra_blocks=(), knot_row_of_token=None, control_shapes=None):
        if arm not in ARMS:
            raise ValueError(f'arm must be one of {list(ARMS)}, got {arm!r}')
        if isinstance(encoder_depth, bool) or int(encoder_depth) != encoder_depth or encoder_depth < 1:
            raise ValueError(f'encoder_depth must be an integer >= 1, got {encoder_depth!r}')
        extra_blocks = tuple(extra_blocks)
        actual = {'num_relations': int(num_relations), 'num_triples': int(num_triples),
                  'edge_dim': int(edge_dim)}
        control = self._validated_control_shapes(control_shapes, actual)
        # E6a (§2.2 / §5 E6a): the relation, triple and payload tensors are registered by the
        # v2 constructor BEFORE the later common tensors, so widening them would shift every
        # later tensor along the global stream. The v2 constructor therefore runs at the
        # control's shapes (global RNG consumed exactly as C) and the widened tensors are
        # replaced afterwards, initialised from their own per-tensor generators.
        super().__init__(num_tokens=num_tokens, node_dim=node_dim,
                         edge_dim=control.get('edge_dim', actual['edge_dim']),
                         num_classes=num_classes, hidden=hidden, token_dim=token_dim,
                         num_triples=control.get('num_triples', actual['num_triples']),
                         num_relations=control.get('num_relations', actual['num_relations']),
                         dropout=dropout, pair_rank=pair_rank, pair_mode='additive',
                         num_node_types=num_node_types)
        self.arm, self.seed, self.encoder_depth = str(arm), int(seed), int(encoder_depth)
        self.control_shapes = dict(control) if control else None
        self.widened_tensors = self._widen_relation_tensors(control, actual)
        # E2d residual blocks (§2.2): registered only for depth >= 2, so depth 1 keeps the
        # exact U3 module set; parameters live under `encoder_blocks.<i>.*`.
        if self.encoder_depth > 1:
            self.encoder_blocks = nn.ModuleList(
                [_ResidualBlock(self.hidden, dropout=self.dropout_rate, seed=self.seed,
                                name=f'encoder_blocks.{index}')
                 for index in range(self.encoder_depth - 1)])
        # `extra_blocks` protocol (§9 U3x): validated, registered as extra_blocks.<name>.
        self.extra_block_names = tuple(str(block.name) for block in
                                       self._validated_extra_blocks(extra_blocks))
        if extra_blocks:
            self.extra_blocks = nn.ModuleDict({block.name: block for block in extra_blocks})
            for block in extra_blocks:
                registered = tuple(f'{EXTRA_BLOCK_PREFIX}.{block.name}.{local}'
                                   for local, _ in block.named_parameters())
                declared = tuple(block.parameter_names())
                if sorted(declared) != sorted(registered):
                    raise ValueError(f'extra block {block.name!r}: parameter_names() {declared} '
                                     f'differ from the registered parameters {registered}')
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

    WIDENABLE = ('num_relations', 'num_triples', 'edge_dim')

    @classmethod
    def _validated_control_shapes(cls, control_shapes, actual):
        """Control (arm C) values of the view-dependent dimensions; equal values drop out."""
        if not control_shapes:
            return {}
        control = {}
        for key, value in dict(control_shapes).items():
            if key not in cls.WIDENABLE:
                raise ValueError(f'control_shapes key {key!r} is not one of {list(cls.WIDENABLE)}')
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f'control_shapes[{key!r}] must be a positive integer')
            if int(value) > actual[key]:
                raise ValueError(f'control_shapes[{key!r}]={value} is wider than the actual '
                                 f'{key}={actual[key]}; only widening relative to C is allowed')
            if int(value) != actual[key]:
                control[key] = int(value)
        return control

    def _widen_relation_tensors(self, control, actual):
        """Replace the view-dependent v2 tensors at their actual (wider) shapes (E6a)."""
        widened = []
        width = self.hidden
        if 'num_relations' in control:
            self.num_relations = actual['num_relations']
            self.relation_embedding = nn.utils.skip_init(
                nn.Embedding, num_embeddings=self.num_relations, embedding_dim=width)
            with torch.no_grad():   # nn.Embedding default: standard normal
                self.relation_embedding.weight.normal_(
                    generator=tensor_generator(self.seed, 'relation_embedding.weight'))
            widened.append('relation_embedding.weight')
        if 'num_triples' in control:
            self.num_triples = actual['num_triples']
            self.triple_embedding = nn.utils.skip_init(
                nn.Embedding, num_embeddings=self.num_triples, embedding_dim=width)
            with torch.no_grad():
                self.triple_embedding.weight.normal_(
                    generator=tensor_generator(self.seed, 'triple_embedding.weight'))
            widened.append('triple_embedding.weight')
        if 'edge_dim' in control:
            self.edge_dim = actual['edge_dim']
            self.edge_feature_projection = nn.utils.skip_init(
                nn.Linear, in_features=self.edge_dim, out_features=width, bias=False)
            bound = 1.0 / float(self.edge_dim) ** 0.5   # nn.Linear default bound at the new fan-in
            with torch.no_grad():
                self.edge_feature_projection.weight.uniform_(
                    -bound, bound,
                    generator=tensor_generator(self.seed, 'edge_feature_projection.weight'))
            widened.append('edge_feature_projection.weight')
        return tuple(widened)

    @staticmethod
    def _validated_extra_blocks(blocks):
        """Protocol checks that need no registration: type, name, uses_rng, uniqueness."""
        seen = set()
        for block in blocks:
            if not isinstance(block, nn.Module) or not isinstance(block, ExtraBlock):
                raise ValueError('every extra block must be an nn.Module satisfying ExtraBlock '
                                 '(name, uses_rng, parameter_names, __call__)')
            name = block.name
            if (not isinstance(name, str) or not name or '.' in name
                    or name in RESERVED_BLOCK_NAMES):
                raise ValueError(f'extra block name {name!r} must be a non-empty identifier '
                                 f'without "." and outside {sorted(RESERVED_BLOCK_NAMES)}')
            if block.uses_rng is not False:
                raise ValueError(f'extra block {name!r}: uses_rng must be False (§2.2: no RNG '
                                 'operation inside extension blocks)')
            if name in seen:
                raise ValueError(f'duplicate extra block name {name!r}')
            seen.add(name)
        return tuple(blocks)

    def _extra_block_terms(self, h, edge_index, metadata, existing_keys):
        """Call every extra block; returns ``(total or None, merged parts)`` (§9 U3x)."""
        graph_count, classes = int(metadata.graph_count), self.num_classes
        total, parts, keys = None, {}, set(existing_keys)
        for name in self.extra_block_names:
            block = self.extra_blocks[name]
            before = torch.get_rng_state()
            block_total, block_parts = block(h, edge_index, metadata.edge_relation,
                                             metadata.batch_index, graph_count)
            if not torch.equal(torch.get_rng_state(), before):
                raise ValueError(f'extra block {name!r} consumed the global RNG stream '
                                 '(uses_rng=False requires a dedicated generator)')
            if (not torch.is_tensor(block_total)
                    or tuple(block_total.shape) != (graph_count, classes)):
                raise ValueError(f'extra block {name!r} must return a total of shape '
                                 f'[{graph_count}, {classes}]')
            for key, value in dict(block_parts).items():
                if not key.startswith(f'{name}_'):
                    raise ValueError(f'extra block {name!r} part key {key!r} must start with '
                                     f'{name + "_"!r}')
                if key in keys:
                    raise ValueError(f'extra block {name!r} part key {key!r} collides with an '
                                     'existing return_parts key')
                keys.add(key)
                parts[key] = value
            total = block_total if total is None else total + block_total
        return total, parts

    def parameter_inventory(self):
        """name -> (shape, active) for every registered parameter (§4.4 inventory).

        v2 tensors are always active; the PLE projection is active in arms B and C, the
        absence tables in arm C only (F3). Residual and extra blocks are always active
        (they are absent, not inactive, in C — extensions spec §1 / E17).
        """
        active_by_name = {'ple_projection.weight': self.ple_active,
                          'absence_vote': self.absence_active,
                          'absence_gate': self.absence_active}
        return {name: (tuple(parameter.shape), bool(active_by_name.get(name, True)))
                for name, parameter in self.named_parameters()}

    def inactive_parameter_count(self):
        inventory = self.parameter_inventory()
        return int(sum(parameter.numel() for name, parameter in self.named_parameters()
                       if not inventory[name][1]))

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

    def _evidence_blocks(self, h, features, edge_index, metadata, membership, visit_graph):
        """v2 additive evidence on a node representation ``h`` (``cei_gnn_v2.py`` lines
        180–239 verbatim: same operations and the same global-RNG consumption order).

        Split out of the v2 forward so that U3x can transform ``h`` before every read of
        it (node head, edge votes, pair votes), which the v2 body does not allow.
        """
        node_count, graph_count = int(features.size(0)), int(metadata.graph_count)
        batch_index, classes = metadata.batch_index, self.num_classes
        node_vote, node_gate = self.node_head(h).chunk(2, dim=-1)
        node_gate = node_gate.sigmoid()
        node_num = h.new_zeros((graph_count, classes)).index_add(0, batch_index, node_gate * node_vote)
        node_den = h.new_ones((graph_count, classes)).index_add(0, batch_index, node_gate)
        node_parts = (node_gate * node_vote) / node_den[batch_index]

        edge_count = edge_index.size(1)
        context = (self.relation_embedding(metadata.edge_relation) + self.triple_embedding(metadata.edge_triple)
                   + self.edge_feature_projection(metadata.edge_attr))
        src, dst = edge_index
        edge_vote = self.edge_source(h[src]) + self.edge_target(h[dst]) + self.edge_context_vote(context)
        edge_gate = self.edge_context_gate(context).sigmoid()
        vote_sum, gate_sum = self.edge_aggregator(edge_index, edge_vote, edge_gate, node_count)
        edge_vote_graph = h.new_zeros((graph_count, classes)).index_add(0, batch_index, vote_sum)
        edge_gate_graph = h.new_zeros((graph_count, classes)).index_add(0, batch_index, gate_sum)
        edge_denominator = 1.0 + edge_gate_graph
        edge_mask = self.edge_aggregator._edge_mask
        if edge_mask is None:
            effective_gate_vote = edge_gate * edge_vote
        else:
            edge_mask = edge_mask.to(device=edge_gate.device, dtype=edge_gate.dtype)
            if getattr(self.edge_aggregator, "_apply_sigmoid", False):
                edge_mask = edge_mask.sigmoid()
            effective_gate_vote = edge_mask[:, None] * edge_gate * edge_vote
        edge_parts = (effective_gate_vote / edge_denominator[batch_index[src]] if edge_count else edge_vote)

        pairs = within_visit_pairs(
            membership, metadata.node_type, node_count,
            **({"node_graph": batch_index, "visit_graph": visit_graph}
               if visit_graph is not None else {}))
        pair_count = pairs.size(1)
        if pair_count == 0:   # pair_mode is fixed to 'additive' (never 'off')
            pair_total = h.new_zeros((graph_count, classes))
            pair_parts = h.new_zeros((pair_count, classes))
            pair_gates = h.new_empty((0,))
            pair_denominator = h.new_ones((graph_count, classes))
        else:
            z = torch.tanh(self.pair_projection(h))
            left, right = pairs
            q = z[left] + z[right]
            q = F.dropout(q, p=self.dropout_rate, training=self.training)
            pair_vote = self.pair_vote(q)
            pair_gate = self.pair_gate[kind_pair_index(metadata.node_type, pairs)].sigmoid()
            pair_gates = pair_gate
            pair_graph = batch_index[left]
            pair_num = h.new_zeros((graph_count, classes)).index_add(0, pair_graph, pair_gate * pair_vote)
            pair_den = h.new_ones((graph_count, classes)).index_add(0, pair_graph, pair_gate)
            pair_denominator = pair_den
            pair_total = pair_num / pair_den
            pair_parts = (pair_gate * pair_vote) / pair_den[pair_graph]

        logits = self.bias + node_num / node_den + edge_vote_graph / edge_denominator + pair_total
        return {"logits": logits, "node_contributions": node_parts, "edge_contributions": edge_parts,
                "pair_contributions": pair_parts, "pairs": pairs, "bias": self.bias,
                "pair_gates": pair_gates, "pair_denominator": pair_denominator}

    def forward_continuous(self, features, edge_index, metadata, membership, *,
                           return_parts=False, visit_graph=None):
        self._validate(features, edge_index, metadata)
        graph_count, classes = int(metadata.graph_count), self.num_classes
        # v2 encoder (`cei_gnn_v2.py` lines 177–179) with the PLE pre-activation term (F12).
        numeric, token_vectors, type_vectors = torch.split(
            features, (self.node_dim, self.token_dim, self.hidden), dim=-1)
        pre_activation = self.node_encoder(torch.cat((numeric, token_vectors, type_vectors), dim=-1))
        if self.ple_active:
            pre_activation = pre_activation + self._ple_term(features, metadata)
        h = F.gelu(self.node_norm(pre_activation))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        # E2d hook: node-local residual blocks before every read of h (§2.2).
        if self.encoder_depth > 1:
            for block in self.encoder_blocks:
                h = block(h)
        result = self._evidence_blocks(h, features, edge_index, metadata, membership, visit_graph)
        if self.absence_active:
            total, absence_parts = self._absence_block(features, metadata, membership, visit_graph)
        else:  # skipped entirely (F3): no tensor derived from the absence parameters
            total = None
            absence_parts = {
                'absence_contributions': features.new_zeros((0, classes)),
                'absence_items': torch.zeros((2, 0), dtype=torch.long, device=features.device),
                'absence_gates': features.new_zeros((0, classes)),
                'absence_denominator': features.new_ones((graph_count, classes))}
        # `extra_blocks` hook (§9 U3x): additive terms after the v2 sum and the absence block.
        extra_parts = {}
        if self.extra_block_names:
            extra_total, extra_parts = self._extra_block_terms(
                h, edge_index, metadata, set(result) | set(absence_parts))
            total = extra_total if total is None else total + extra_total
        if total is not None:
            result['logits'] = result['logits'] + total
        if not return_parts:
            return result['logits']
        result.update(absence_parts)
        result.update(extra_parts)
        return result


def absence_label(item):
    """Explanation label of an absent universe item (v3 §12 F8 wording)."""
    return ABSENCE_LABEL.format(item=item)
