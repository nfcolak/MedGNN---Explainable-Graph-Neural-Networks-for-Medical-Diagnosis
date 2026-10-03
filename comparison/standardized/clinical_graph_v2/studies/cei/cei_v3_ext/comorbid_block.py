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

from ....methods.cei_gnn_v3 import EXTRA_BLOCK_PREFIX, tensor_generator

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
    """Additive comorbid pair block (E6b) satisfying the U3x ``ExtraBlock`` protocol.

    For each unique unordered comorbid pair ``(i, j)`` of graph ``G`` (§5.3, D10.6)::

        a_ij        = tanh(P_a h_i) + tanh(P_a h_j)     P_a: hidden × 16, no bias
        v_ij        = V_a a_ij + b_a                     V_a: 16 × C, b_a: C
        g_c         = sigmoid(γ_c)                       one gate logit per class
        comorbid_Gc = Σ_pairs g_c v_ijc / (1 + Σ_pairs g_c)

    Dedicated parameters (never the v2 ``pair_projection``/``pair_vote``): ``P_a`` =
    ``pair_projection.weight``, ``V_a``/``b_a`` = ``vote.weight``/``vote.bias``, ``γ`` =
    ``gate``; 2,228 parameters at hidden 128 and 10 classes. Every tensor is initialised
    from ``tensor_generator(seed, 'extra_blocks.comorbid.<local>')`` (nn.Linear's default
    uniform bound at its fan-in; the gate logits start at zero like the v2 ``pair_gate``),
    so the global RNG stream is untouched; the forward has no dropout and no RNG
    operation. An empty pair set gives an exact-zero total and denominator one.
    """

    name = BLOCK_NAME
    uses_rng = False

    def __init__(self, hidden, num_classes, *, relation_id, seed):
        super().__init__()
        self.hidden, self.num_classes = int(hidden), int(num_classes)
        self.relation_id, self.seed = int(relation_id), int(seed)
        self.pair_projection = nn.utils.skip_init(nn.Linear, in_features=self.hidden,
                                                  out_features=PAIR_RANK, bias=False)
        self.vote = nn.utils.skip_init(nn.Linear, in_features=PAIR_RANK,
                                       out_features=self.num_classes)
        self.gate = nn.Parameter(torch.zeros(self.num_classes))
        prefix = f'{EXTRA_BLOCK_PREFIX}.{self.name}'
        with torch.no_grad():
            bound = 1.0 / float(self.hidden) ** 0.5   # nn.Linear default bound, fan_in = hidden
            self.pair_projection.weight.uniform_(
                -bound, bound, generator=tensor_generator(seed, f'{prefix}.pair_projection.weight'))
            bound = 1.0 / float(PAIR_RANK) ** 0.5    # fan_in = 16 for V_a and b_a
            self.vote.weight.uniform_(
                -bound, bound, generator=tensor_generator(seed, f'{prefix}.vote.weight'))
            self.vote.bias.uniform_(
                -bound, bound, generator=tensor_generator(seed, f'{prefix}.vote.bias'))

    def parameter_names(self) -> Tuple[str, ...]:
        """Qualified names in the §5.3 order P_a, V_a, b_a, γ."""
        prefix = f'{EXTRA_BLOCK_PREFIX}.{self.name}'
        return (f'{prefix}.pair_projection.weight', f'{prefix}.vote.weight',
                f'{prefix}.vote.bias', f'{prefix}.gate')

    def forward(self, h, edge_index, edge_relation, batch_index, graph_count):
        graphs, classes = int(graph_count), self.num_classes
        pairs = comorbid_pairs(edge_index, edge_relation, self.relation_id, batch_index)
        gates = self.gate.sigmoid()                                    # [C], shared by all pairs
        denominator = h.new_ones((graphs, classes))
        if pairs.size(1) == 0:
            parts: Dict[str, torch.Tensor] = {
                'comorbid_contributions': h.new_zeros((0, classes)),
                'comorbid_pairs': pairs, 'comorbid_gates': gates,
                'comorbid_denominator': denominator}
            return h.new_zeros((graphs, classes)), parts
        left, right = pairs
        z = torch.tanh(self.pair_projection(h))                        # tanh(P_a h)  [N, 16]
        votes = self.vote(z[left] + z[right])                          # v_ij         [pairs, C]
        gated = votes * gates                                          # g_c v_ijc
        graph = batch_index[left]
        denominator = denominator.index_add(0, graph, gates.expand(pairs.size(1), classes))
        total = h.new_zeros((graphs, classes)).index_add(0, graph, gated) / denominator
        parts = {'comorbid_contributions': gated / denominator[graph],
                 'comorbid_pairs': pairs, 'comorbid_gates': gates,
                 'comorbid_denominator': denominator}
        return total, parts


def build_block(hidden, num_classes, *, relation_layout: Dict[str, int], seed) -> ComorbidPairBlock:
    """Adapter entry point (§9 X5): the comorbid relation id is resolved by NAME (E15)."""
    layout = dict(relation_layout)
    if COMORBID_RELATION not in layout:
        raise ValueError(f'relation_layout has no {COMORBID_RELATION!r} relation; the comorbid '
                         'block needs its id resolved by name from the bound layout')
    return ComorbidPairBlock(hidden, num_classes, relation_id=layout[COMORBID_RELATION], seed=seed)
