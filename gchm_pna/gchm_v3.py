"""GCHM-PNA v3: v2's hub-gated compact PNA plus three read-side mechanisms.

Why v3 exists (measured on a scratch *design* split: 5,000 TRAIN visits whose patients
are disjoint from the protocol's 10k training sample AND its dev split; validation and
test were never encoded):

* v2 reached 0.633 design macro-F1 while a bag-of-token XGBoost reached 0.646 on the
  same rows. v2 had no path that could express "this token alone votes for class c":
  every token had to survive three rounds of gated mixing and a sum/mean pool first.
* Per-class evidence is sparse (one complaint, one abnormal lab). A single pooled graph
  vector shared by all ten classes dilutes it; a class-specific attention over nodes
  (CAML-style label-wise attention) lets each diagnosis pick its own evidence.
* The deepest layer is not always the best one; jumping knowledge keeps every depth.

Mechanisms, each removable by one switch while everything else stays fixed:

1. `wide=True`   A linear per-token class-vote path summed over the graph's nodes,
                 `sum_i W_tok[t_i] + v_i * W_val[t_i]`, zero-initialised. It reads only
                 this graph's own node tokens/values -- the same primitives every arm
                 receives -- and never another model's predictions.
2. `readout='labelwise'`  one learned query per class attends over final node states
                 (per-graph softmax); its class-specific context adds to the pooled head.
                 `readout='pool'` keeps v2's [sum, mean, hub] head only.
3. `jk=True`     final node state = Linear(concat(encoder, layer1..L)).
4. `edge_dropout`  training-time random edge removal (graph-level augmentation).

Everything upstream -- encoder, HubGatedPNALayer stack, relation vocabulary checks --
is v2's code, imported, not copied.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as segment_softmax

from core import NODE_KINDS
from .gchm_v2 import HUB_KIND, HubGatedPNALayer, average_log_degree

READOUTS = ('labelwise', 'pool')
DEFAULT_HIDDEN = 88  # largest width below the frozen ProtGNN parameter count
VALUE_COLUMN = len(NODE_KINDS)          # scaled numeric value (tensorize layout)
HAS_VALUE_COLUMN = len(NODE_KINDS) + 1  # its presence flag


class GCHMv3(nn.Module):
    """Graph classifier: v2 encoder + hub-gated PNA, JK, label-wise readout, wide path."""

    conv = 'gchm_v3'

    def __init__(self, num_tokens, node_dim, edge_dim, num_classes, *, num_relations,
                 degree_histogram, hidden=DEFAULT_HIDDEN, layers=3, dropout=0.3,
                 token_dim=32, use_edge_payload=True, use_message_passing=True,
                 readout='labelwise', wide=True, jk=True, edge_dropout=0.1):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f'readout must be one of {READOUTS}')
        if min(int(hidden), int(layers), int(num_classes), int(num_tokens)) < 1:
            raise ValueError('hidden, layers, num_classes and num_tokens must be positive')
        if not 0.0 <= float(edge_dropout) < 1.0:
            raise ValueError('edge_dropout must be in [0, 1)')
        self.use_message_passing = bool(use_message_passing)
        self.readout, self.wide, self.jk = readout, bool(wide), bool(jk)
        self.edge_dropout = float(edge_dropout)
        self.num_relations = int(num_relations)
        self.num_classes = int(num_classes)
        self.register_buffer('degree_histogram', torch.as_tensor(degree_histogram).float())
        avg_deg_log = average_log_degree(self.degree_histogram)
        self.token = nn.Embedding(num_tokens, token_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(node_dim + token_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.kind = nn.Embedding(len(NODE_KINDS), hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([
            HubGatedPNALayer(hidden, edge_dim, num_relations, avg_deg_log, dropout,
                             use_edge_payload=use_edge_payload)
            for _ in range(layers)])
        self.jk_projection = nn.Linear(hidden * (layers + 1), hidden) if self.jk else None
        self.pool_norm = nn.LayerNorm(3 * hidden)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))
        if readout == 'labelwise':
            self.class_query = nn.Parameter(torch.randn(num_classes, hidden) / math.sqrt(hidden))
            self.class_weight = nn.Parameter(torch.zeros(num_classes, hidden))
            self.class_bias = nn.Parameter(torch.zeros(num_classes))
            self.labelwise_norm = nn.LayerNorm(hidden)
        if self.wide:
            self.wide_token = nn.Embedding(num_tokens, num_classes)
            self.wide_value = nn.Embedding(num_tokens, num_classes)
            nn.init.zeros_(self.wide_token.weight)
            nn.init.zeros_(self.wide_value.weight)
        self.dropout = nn.Dropout(dropout)
        self.settings = {'hidden': int(hidden), 'layers': int(layers),
                         'dropout': float(dropout), 'token_dim': int(token_dim),
                         'num_relations': self.num_relations, 'avg_deg_log': avg_deg_log,
                         'readout': readout, 'wide': self.wide, 'jk': self.jk,
                         'edge_dropout': self.edge_dropout,
                         'edge_payload': bool(use_edge_payload),
                         'message_passing': self.use_message_passing}

    @staticmethod
    def hub_state(h, batch, hub_mask, graph_count):
        """Index-visit hub state per graph; a graph without a hub gets zeros."""
        return scatter(h[hub_mask], batch[hub_mask], 0, dim_size=graph_count, reduce='mean')

    def _edges(self, data):
        edge_index, edge_attr, relation = data.edge_index, data.edge_attr, data.edge_relation
        if relation.numel() and (int(relation.min()) < 0
                                 or int(relation.max()) >= self.num_relations):
            raise ValueError("edge_relation id is outside this model's relation "
                             'vocabulary; edge_direction differs from construction')
        if self.training and self.edge_dropout > 0 and relation.numel():
            keep = torch.rand(relation.numel(), device=relation.device) >= self.edge_dropout
            edge_index, edge_attr, relation = edge_index[:, keep], edge_attr[keep], relation[keep]
        return edge_index, edge_attr, relation

    def node_states(self, data, edge_mask=None):
        """Final (JK-combined) node states, batch vector, hub mask and graph count."""
        h = self.encoder(torch.cat([data.x, self.token(data.token)], dim=-1))
        h = self.input_norm(h + self.kind(data.node_type))
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        size = int(batch.max()) + 1
        hub_mask = data.node_type == HUB_KIND
        states = [h]
        if self.use_message_passing:
            edge_index, edge_attr, relation = self._edges(data)
            if edge_mask is not None and edge_mask.numel() != edge_index.size(1):
                raise ValueError(
                    "edge_mask must contain one value per edge of the unmodified "
                    "edge_index -- this model applies training-time edge_dropout "
                    "before message passing, but that branch never runs in eval "
                    "mode (the only mode a GraphXAI explainer uses), so the two "
                    "never actually conflict in practice; this check only guards "
                    "against a caller misusing eval()-only state."
                )
            for layer in self.layers:
                hub = self.hub_state(h, batch, hub_mask, size)
                h = layer(h, edge_index, edge_attr, relation, hub[batch], edge_mask=edge_mask)
                states.append(h)
        if self.jk:
            while len(states) < len(self.layers) + 1:  # NOMP: repeat the encoder state
                states.append(states[0])
            h = self.jk_projection(torch.cat(states, dim=-1))
        return h, batch, hub_mask, size

    def labelwise_logits(self, h, batch, size):
        """Per-class attention over each graph's nodes -> [graphs, classes]."""
        states = self.labelwise_norm(h)
        score = states @ self.class_query.t() / math.sqrt(states.size(-1))
        alpha = segment_softmax(score, batch, num_nodes=size, dim=0)
        context = scatter(alpha.unsqueeze(-1) * states.unsqueeze(1), batch, 0,
                          dim_size=size, reduce='sum')
        return (self.dropout(context) * self.class_weight).sum(-1) + self.class_bias

    def wide_logits(self, data, batch, size):
        """Linear per-token class votes of this graph's own nodes, summed per graph."""
        value = (data.x[:, VALUE_COLUMN] * data.x[:, HAS_VALUE_COLUMN]).unsqueeze(-1)
        votes = self.wide_token(data.token) + value * self.wide_value(data.token)
        return scatter(votes, batch, 0, dim_size=size, reduce='sum')

    def forward(self, data, edge_mask=None):
        h, batch, hub_mask, size = self.node_states(data, edge_mask=edge_mask)
        pooled = torch.cat([scatter(h, batch, 0, dim_size=size, reduce='sum'),
                            scatter(h, batch, 0, dim_size=size, reduce='mean'),
                            self.hub_state(h, batch, hub_mask, size)], dim=-1)
        logits = self.head(self.pool_norm(pooled))
        if self.readout == 'labelwise':
            logits = logits + self.labelwise_logits(h, batch, size)
        if self.wide:
            logits = logits + self.wide_logits(data, batch, size)
        return logits

    def parameter_count(self):
        """All allocated parameters, including bypassed NOMP layers."""
        return sum(p.numel() for p in self.parameters())

    def active_parameter_count(self):
        """Parameters in enabled forward blocks (NOMP bypasses the layer stack)."""
        if self.use_message_passing:
            return self.parameter_count()
        bypassed = {id(p) for p in self.layers.parameters()}
        return sum(p.numel() for p in self.parameters() if id(p) not in bypassed)

    def relation_separation(self):
        """No HGT layers; kept for the runner's shared reporting call."""
        return []

    def gate_report(self):
        return [None if layer.last_gate_mean is None else float(layer.last_gate_mean)
                for layer in self.layers]
