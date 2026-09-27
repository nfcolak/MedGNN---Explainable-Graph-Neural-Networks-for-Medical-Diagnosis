"""Independent class-specific evidence interaction computation for clinical graphs."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EdgeEvidenceAggregator(MessagePassing):
    """Sum [gate * vote, gate] messages to target nodes using PyG's mask hook."""

    def __init__(self, num_classes: int):
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)
        self.num_classes = int(num_classes)

    def forward(self, edge_index, votes, gates, node_count):
        messages = self.propagate(
            edge_index, votes=votes, gates=gates, size=(node_count, node_count))
        return messages[:, :self.num_classes], messages[:, self.num_classes:]

    def message(self, votes, gates):
        return torch.cat((gates * votes, gates), dim=-1)


class EvidenceInteractionNetwork(nn.Module):
    """CEI-GNN's exact continuous-input predictive core."""

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 token_dim, num_triples, num_relations, dropout, interaction_rank,
                 use_interactions, num_node_types):
        super().__init__()
        self.num_tokens, self.node_dim, self.edge_dim = int(num_tokens), int(node_dim), int(edge_dim)
        self.num_classes, self.hidden = int(num_classes), int(hidden)
        self.token_dim, self.num_triples = int(token_dim), int(num_triples)
        self.num_relations = int(num_relations)
        self.dropout_rate, self.interaction_rank = float(dropout), int(interaction_rank)
        self.use_interactions = bool(use_interactions)
        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(int(num_node_types), self.hidden)
        self.node_encoder = nn.Linear(self.node_dim + self.token_dim + self.hidden, self.hidden)
        self.node_norm = nn.LayerNorm(self.hidden)
        self.relation_embedding = nn.Embedding(self.num_relations, self.hidden)
        self.triple_embedding = nn.Embedding(self.num_triples, self.hidden)
        self.edge_feature_projection = nn.Linear(self.edge_dim, self.hidden, bias=False)
        self.endpoint_source = nn.Linear(self.hidden, self.interaction_rank, bias=False)
        self.endpoint_target = nn.Linear(self.hidden, self.interaction_rank, bias=False)
        self.interaction_context = nn.Linear(self.hidden, self.interaction_rank)
        edge_width = self.hidden * 3 + self.interaction_rank
        self.edge_head = nn.Sequential(
            nn.Linear(edge_width, self.hidden), nn.GELU(), nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden, self.num_classes * 2))
        self.node_head = nn.Linear(self.hidden, self.num_classes * 2)
        self.edge_aggregator = EdgeEvidenceAggregator(self.num_classes)
        self.edge_aggregator._edge_mask = None
        self.edge_aggregator._apply_sigmoid = False
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    @property
    def continuous_width(self):
        return self.node_dim + self.token_dim + self.hidden

    def continuous_inputs(self, clinical):
        return torch.cat((clinical.x, self.token_embedding(clinical.token),
                          self.node_type_embedding(clinical.node_type)), dim=-1)

    def set_edge_mask(self, mask):
        """Install a direct probability mask or restore PyG's neutral mask state."""
        aggregator = self.edge_aggregator
        if mask is not None:
            if mask.ndim != 1 or not torch.isfinite(mask).all():
                raise ValueError("edge mask must be a finite vector")
            if (mask < 0).any() or (mask > 1).any():
                raise ValueError("edge mask values must be in [0, 1]")
        # PyG registers its explanation mask as a Parameter. Direct masks are
        # probabilities, so remove that registration before accepting a tensor.
        if "_edge_mask" in aggregator._parameters:
            del aggregator._parameters["_edge_mask"]
        aggregator._edge_mask = mask
        aggregator.explain = None if mask is None else True
        aggregator._apply_sigmoid = False if mask is not None else True

    def forward_continuous(self, features, edge_index, metadata, *, return_parts=False):
        node_count = int(features.size(0))
        batch_index = metadata.batch_index
        graph_count = int(metadata.graph_count)
        if features.ndim != 2 or tuple(features.shape) != (node_count, self.continuous_width):
            raise ValueError("continuous features have an incompatible shape")
        if node_count == 0:
            raise ValueError("CEI-GNN requires at least one node")
        if not torch.isfinite(features).all():
            raise ValueError("continuous features must be finite")
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be a long tensor with shape [2, edges]")
        edge_count = edge_index.size(1)
        edge_mask = self.edge_aggregator._edge_mask
        if edge_mask is not None and (edge_mask.ndim != 1 or edge_mask.numel() != edge_count):
            raise ValueError("edge mask length differs from the edge list")
        if edge_count:
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count:
                raise ValueError("edge_index refers to a node outside the batch")
            if not torch.equal(batch_index[edge_index[0]], batch_index[edge_index[1]]):
                raise ValueError("cross-graph edges are not allowed")
        if graph_count < 1 or batch_index.numel() != node_count:
            raise ValueError("invalid graph membership metadata")
        if not torch.isfinite(metadata.edge_attr).all():
            raise ValueError("edge_attr must be finite")

        numeric, token_vectors, type_vectors = torch.split(
            features, (self.node_dim, self.token_dim, self.hidden), dim=-1)
        h = F.gelu(self.node_norm(self.node_encoder(torch.cat(
            (numeric, token_vectors, type_vectors), dim=-1))))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        raw_node = self.node_head(h)
        node_vote, node_gate = raw_node.chunk(2, dim=-1)
        node_gate = node_gate.sigmoid()
        node_num = h.new_zeros((graph_count, self.num_classes)).index_add(
            0, batch_index, node_gate * node_vote)
        node_den = h.new_ones((graph_count, self.num_classes)).index_add(
            0, batch_index, node_gate)
        node_parts = (node_gate * node_vote) / node_den[batch_index]

        relation = metadata.edge_relation
        triple = metadata.edge_triple
        if edge_count and (int(relation.min()) < 0 or int(relation.max()) >= self.num_relations):
            raise ValueError("edge_relation index is outside the fitted vocabulary")
        if edge_count and (int(triple.min()) < 0 or int(triple.max()) >= self.num_triples):
            raise ValueError("edge_triple index is outside the fitted vocabulary")
        context = (self.relation_embedding(relation) + self.triple_embedding(triple)
                   + self.edge_feature_projection(metadata.edge_attr))
        src, dst = edge_index
        q = (self.endpoint_source(h[src]).tanh() * self.endpoint_target(h[dst]).tanh()
             * self.interaction_context(context).sigmoid())
        if not self.use_interactions:
            q = torch.zeros_like(q)
        edge_input = torch.cat((h[src], h[dst], context, q), dim=-1)
        edge_input = F.dropout(edge_input, p=self.dropout_rate, training=self.training)
        raw_edge = self.edge_head(edge_input)
        edge_vote, edge_gate = raw_edge.chunk(2, dim=-1)
        edge_gate = edge_gate.sigmoid()
        node_vote_sum, node_gate_sum = self.edge_aggregator(edge_index, edge_vote, edge_gate, node_count)
        edge_vote_graph = h.new_zeros((graph_count, self.num_classes)).index_add(
            0, batch_index, node_vote_sum)
        edge_gate_graph = h.new_zeros((graph_count, self.num_classes)).index_add(
            0, batch_index, node_gate_sum)
        edge_denominator = 1.0 + edge_gate_graph
        edge_mask = self.edge_aggregator._edge_mask
        if edge_mask is None:
            effective_gate_vote = edge_gate * edge_vote
            effective_gate = edge_gate
        else:
            edge_mask = edge_mask.to(device=edge_gate.device, dtype=edge_gate.dtype)
            if edge_mask.numel() != edge_count:
                raise ValueError("edge mask length differs from the edge list")
            if getattr(self.edge_aggregator, "_apply_sigmoid", False):
                edge_mask = edge_mask.sigmoid()
            effective_gate_vote = edge_mask[:, None] * edge_gate * edge_vote
            effective_gate = edge_mask[:, None] * edge_gate
        edge_parts = (effective_gate_vote / edge_denominator[batch_index[src]]
                      if edge_count else edge_vote)
        # A fixed sentinel denominator makes both edgeless graph totals finite.
        logits = self.bias + node_num / node_den + edge_vote_graph / edge_denominator
        if not return_parts:
            return logits
        return {"logits": logits, "node_contributions": node_parts,
                "edge_contributions": edge_parts, "bias": self.bias}



