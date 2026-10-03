"""GCHM-PNA v2: hub-gated, relation-aware, compact PNA over the typed clinical graph.

Why a second version exists (measured on the frozen max6 artifact before any design;
see ProjectOS 10-Projects/medgnn/Reports/gchm_pna/superpowers/specs/2026-09-24-gchm-pna-v2-design.md):

* The producer emits every visit->evidence relation in ONE direction, so under
  source->target message passing no complaint, vital or laboratory node reached the
  index-visit hub at any depth. v2 is meant for the `edge_direction='bidirectional'`
  view (typed reverse edges, tensorize.py); the forward view is its ablation.
* 71% of nodes received at most one message. For those nodes v1's four aggregators
  collapse (mean = min = max, std = 0), so v1's 12 aggregator x scaler blocks were
  mostly redundant width: 557k parameters, the largest arm, and it overfit.

Mechanism. Every part has a switch that removes it while the rest stays fixed:

1. Multiplicative hub gate: message = content * sigmoid(W_r x_i + W_h hub + e_rel).
   The receiver state, the graph's index-visit hub state and the relation type jointly
   SCALE each message. `modulation='additive'` is the capacity-matched control (same
   parameters; the gate can only shift). `hub_gate=False` drops only the W_h term.
2. Compact PNA: mean/min/max/std are aggregated, projected 4d->d once, then scaled by
   learned per-channel identity/amplification/attenuation weights initialised to the
   identity scaler. Degree information is kept at 4d*d cost instead of 13d*d.
   `aggregation='sum'` is the no-PNA control (smaller; its count is recorded).
3. Hub readout: [sum pool, mean pool, index-visit hub state]. `readout='pool'` drops
   the hub term and matches the v1 readout.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from core import NODE_KINDS
from core.tensorize import PAYLOAD_WIDTH

HUB_KIND = NODE_KINDS.index('visit')
MODULATIONS = ('multiplicative', 'additive')
AGGREGATIONS = ('pna', 'sum')
READOUTS = ('hub', 'pool')
# Default width. With the bidirectional view (24 relations, 31-wide edge_attr) and the
# frozen 391-token vocabulary this gives 379,786 parameters, under ProtGNN's 399,884,
# the largest rival GNN. The runner records the realised count in every binding.
DEFAULT_HIDDEN = 92


def average_log_degree(degree_histogram):
    """PyG PNA's normaliser: mean of log(deg + 1) under the train in-degree histogram."""
    hist = torch.as_tensor(degree_histogram, dtype=torch.float64).flatten()
    if hist.numel() < 1 or bool((hist < 0).any()) or float(hist.sum()) <= 0:
        raise ValueError('degree_histogram must be a nonempty nonnegative count vector')
    degrees = torch.arange(hist.numel(), dtype=torch.float64)
    value = float((torch.log(degrees + 1.0) * hist).sum() / hist.sum())
    # A fold whose training graphs are genuinely edgeless has average log degree
    # zero; keep the histogram exact and only floor the numerical denominator.
    return max(value, 1e-6)


class HubGatedPNALayer(nn.Module):
    """One message-passing layer; see the module docstring for the mechanism."""

    def __init__(self, dim, edge_dim, num_relations, avg_deg_log, dropout=0.1, *,
                 use_edge_payload=True, modulation='multiplicative', aggregation='pna',
                 hub_gate=True):
        super().__init__()
        if modulation not in MODULATIONS:
            raise ValueError(f'modulation must be one of {MODULATIONS}')
        if aggregation not in AGGREGATIONS:
            raise ValueError(f'aggregation must be one of {AGGREGATIONS}')
        if int(num_relations) < 1 or int(edge_dim) < PAYLOAD_WIDTH:
            raise ValueError('num_relations must be positive and edge_dim must hold the payload')
        if not math.isfinite(avg_deg_log) or avg_deg_log <= 0:
            raise ValueError('avg_deg_log must be finite and positive')
        self.use_edge_payload = bool(use_edge_payload)
        self.modulation = modulation
        self.aggregation = aggregation
        self.avg_deg_log = float(avg_deg_log)
        self.num_relations = int(num_relations)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.receiver_gate = nn.Linear(dim, dim)
        self.hub_gate = nn.Linear(dim, dim, bias=False) if hub_gate else None
        self.relation_gate = nn.Embedding(self.num_relations, dim)
        if aggregation == 'pna':
            self.aggregate = nn.Linear(4 * dim, dim)
            scaler = torch.zeros(3, dim)
            scaler[0] = 1.0  # identity scaler; amplification/attenuation start at 0
            self.scaler_weight = nn.Parameter(scaler)
        else:
            self.aggregate = nn.Linear(dim, dim)
            self.register_parameter('scaler_weight', None)
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.last_gate_mean = None

    def gate(self, receiver, relation, hub):
        """Pre-sigmoid gate logits for receivers, relation ids and receiver-graph hubs."""
        logits = self.receiver_gate(receiver) + self.relation_gate(relation)
        if self.hub_gate is not None:
            logits = logits + self.hub_gate(hub)
        return logits

    def message(self, x_source, x_target, payload, relation, hub_target):
        """Per-edge message; exposed so the mechanism checks can probe it directly."""
        content = self.message_mlp(torch.cat([x_source, x_target, payload], dim=-1))
        gate = torch.sigmoid(self.gate(x_target, relation, hub_target))
        self.last_gate_mean = gate.detach().mean() if gate.numel() else None
        if self.modulation == 'multiplicative':
            return content * gate
        return content + gate

    def forward(self, x, edge_index, edge_attr, edge_relation, hub_context):
        source, target = edge_index
        payload = edge_attr
        if not self.use_edge_payload:
            payload = edge_attr.clone()
            payload[:, -PAYLOAD_WIDTH:] = 0
        message = self.message(x[source], x[target], payload, edge_relation,
                               hub_context[target])
        aggregated = self._aggregate(message, target, x.size(0))
        update = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        return self.norm(x + self.dropout(update))

    def _aggregate(self, message, target, num_nodes):
        if self.aggregation == 'sum':
            return self.aggregate(scatter(message, target, 0, dim_size=num_nodes, reduce='sum'))
        mean = scatter(message, target, 0, dim_size=num_nodes, reduce='mean')
        minimum = scatter(message, target, 0, dim_size=num_nodes, reduce='min')
        maximum = scatter(message, target, 0, dim_size=num_nodes, reduce='max')
        square = scatter(message * message, target, 0, dim_size=num_nodes, reduce='mean')
        std = torch.sqrt(torch.relu(square - mean * mean) + 1e-5)
        projected = self.aggregate(torch.cat([mean, minimum, maximum, std], dim=-1))
        degree = torch.bincount(target, minlength=num_nodes).clamp(min=1).to(message.dtype)
        log_degree = torch.log(degree + 1.0).unsqueeze(-1)
        weight = self.scaler_weight
        scale = (weight[0] + weight[1] * (log_degree / self.avg_deg_log)
                 + weight[2] * (self.avg_deg_log / log_degree))
        return projected * scale


class GCHMv2(nn.Module):
    """Graph classifier: shared encoder, HubGatedPNALayer stack, hub-aware readout."""

    conv = 'gchm_v2'

    def __init__(self, num_tokens, node_dim, edge_dim, num_classes, *, num_relations,
                 degree_histogram, hidden=DEFAULT_HIDDEN, layers=3, dropout=0.1,
                 token_dim=32, use_edge_payload=True, use_message_passing=True,
                 modulation='multiplicative', aggregation='pna', readout='hub',
                 hub_gate=True):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f'readout must be one of {READOUTS}')
        if min(int(hidden), int(layers), int(num_classes), int(num_tokens)) < 1:
            raise ValueError('hidden, layers, num_classes and num_tokens must be positive')
        self.use_message_passing = bool(use_message_passing)
        self.readout = readout
        self.num_relations = int(num_relations)
        self.register_buffer('degree_histogram', torch.as_tensor(degree_histogram).float())
        avg_deg_log = average_log_degree(self.degree_histogram)
        self.token = nn.Embedding(num_tokens, token_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(node_dim + token_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([
            HubGatedPNALayer(hidden, edge_dim, num_relations, avg_deg_log, dropout,
                             use_edge_payload=use_edge_payload, modulation=modulation,
                             aggregation=aggregation, hub_gate=hub_gate)
            for _ in range(layers)])
        width = (3 if readout == 'hub' else 2) * hidden
        self.pool_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))
        self.settings = {'hidden': int(hidden), 'layers': int(layers),
                         'dropout': float(dropout), 'token_dim': int(token_dim),
                         'num_relations': self.num_relations,
                         'avg_deg_log': avg_deg_log, 'modulation': modulation,
                         'aggregation': aggregation, 'readout': readout,
                         'hub_gate': bool(hub_gate),
                         'edge_payload': bool(use_edge_payload),
                         'message_passing': self.use_message_passing}

    @staticmethod
    def hub_state(h, batch, hub_mask, graph_count):
        """Index-visit hub state per graph; a graph without a hub gets zeros."""
        return scatter(h[hub_mask], batch[hub_mask], 0, dim_size=graph_count, reduce='mean')

    def node_states(self, data):
        """Final node states, batch vector, hub mask and graph count of one batch."""
        h = self.encoder(torch.cat([data.x, self.token(data.token)], dim=-1))
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        size = int(batch.max()) + 1
        hub_mask = data.node_type == HUB_KIND
        if self.use_message_passing:
            relation = data.edge_relation
            if relation.numel() and (int(relation.min()) < 0
                                     or int(relation.max()) >= self.num_relations):
                raise ValueError('edge_relation id is outside this model\'s relation '
                                 'vocabulary; edge_direction differs from construction')
            for layer in self.layers:
                hub = self.hub_state(h, batch, hub_mask, size)
                h = layer(h, data.edge_index, data.edge_attr, relation, hub[batch])
        return h, batch, hub_mask, size

    def forward(self, data):
        h, batch, hub_mask, size = self.node_states(data)
        pooled = [scatter(h, batch, 0, dim_size=size, reduce='sum'),
                  scatter(h, batch, 0, dim_size=size, reduce='mean')]
        if self.readout == 'hub':
            pooled.append(self.hub_state(h, batch, hub_mask, size))
        return self.head(self.pool_norm(torch.cat(pooled, dim=-1)))

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
        """Mean sigmoid gate per layer from the latest forward pass (diagnostic only)."""
        return [None if layer.last_gate_mean is None else float(layer.last_gate_mean)
                for layer in self.layers]
