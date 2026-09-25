"""A small edge-conditioned GNN for the clinical decision-point graph.

Deliberately small (~100k parameters at default width) because the point of this run
is to compare INPUTS and EDGE SETS, not to win a benchmark. Every arm shares this
class and differs only by a constructor flag, so parameter count stays identical
across arms and no difference can be attributed to capacity.

Message function, the part that matters:

    m(i<-j) = MLP([x_j, x_i, e_ji])

It takes the sender, the RECEIVER, and the edge payload. This is not decoration: an
additively separable message m(x_j) + f(e) can never express "this creatinine is high
*relative to this patient's own baseline*", which is exactly what `baseline_of`
carries. `test_message_not_separable` in the mechanism check verifies the residual

    m(x_i, x_j, e) - [m(x_i, 0, 0) + m(0, x_j, 0) + m(0, 0, e) - 2*m(0, 0, 0)]

is far above float32 roundoff, i.e. the layer really does mix its inputs.

Readout is sum + mean concatenated: sum keeps multiplicity (how many abnormal results
support a class), mean keeps intensity, and graphs here vary hugely in size.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter


class EdgeConditionedLayer(MessagePassing):
    """One round of edge-conditioned message passing with a residual update."""

    def __init__(self, dim, edge_dim, dropout=0.1, use_edge_payload=True):
        super().__init__(aggr='add', node_dim=0)
        self.use_edge_payload = use_edge_payload
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        if self.use_edge_payload:
            payload = edge_attr
        else:
            # Ablation: keep the same tensor shape and parameter count, but blank the
            # payload so any gain must come from topology alone, not from the deltas.
            payload = torch.zeros_like(edge_attr)
        out = self.propagate(edge_index, x=x, edge_attr=payload)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x + self.dropout(out))

    def message(self, x_j, x_i, edge_attr):
        return self.message_mlp(torch.cat([x_j, x_i, edge_attr], dim=-1))


class ClinicalGNN(nn.Module):
    def __init__(self, num_tokens, node_dim, edge_dim, num_classes=30,
                 hidden=96, layers=3, dropout=0.1, token_dim=32,
                 use_edge_payload=True, use_message_passing=True):
        super().__init__()
        self.use_message_passing = use_message_passing
        self.token = nn.Embedding(num_tokens, token_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(node_dim + token_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([
            EdgeConditionedLayer(hidden, edge_dim, dropout, use_edge_payload)
            for _ in range(layers)])
        # Sum pooling scales with graph size, and these graphs range from 1 to
        # thousands of nodes. Without normalising the pooled vector the logit scale
        # tracks node count (measured |z|max ~80, std ~9 at init), the softmax
        # saturates and gradients vanish -- the run then sits at the majority class
        # with a frozen loss. Normalising here makes the readout size-independent
        # while keeping sum's multiplicity information in the direction of the vector.
        self.pool_norm = nn.LayerNorm(2 * hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))

    def forward(self, data):
        h = self.encoder(torch.cat([data.x, self.token(data.token)], dim=-1))
        if self.use_message_passing:
            for layer in self.layers:
                h = layer(h, data.edge_index, data.edge_attr)
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        size = int(batch.max()) + 1
        pooled = torch.cat([scatter(h, batch, dim=0, dim_size=size, reduce='sum'),
                            scatter(h, batch, dim=0, dim_size=size, reduce='mean')], dim=-1)
        return self.head(self.pool_norm(pooled))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
