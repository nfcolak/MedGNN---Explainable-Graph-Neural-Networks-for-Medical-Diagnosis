"""CEI-GNN v2: node, endpoint-additive edge and within-visit pair evidence."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import NODE_KINDS
from .cei_gnn import EdgeEvidenceAggregator

EVIDENCE_KINDS = ("complaint", "measurement", "vital")
EVIDENCE_KIND_IDS = tuple(NODE_KINDS.index(kind) for kind in EVIDENCE_KINDS)
KIND_PAIR_COUNT = len(EVIDENCE_KINDS) * (len(EVIDENCE_KINDS) + 1) // 2
PAIR_MODES = ("product", "additive", "off")


def _empty_pairs(device):
    return torch.zeros((2, 0), dtype=torch.long, device=device)


def within_visit_pairs(membership, node_type, node_count, *, node_graph=None, visit_graph=None):
    """Unique unordered evidence-node pairs (i < j) that share at least one visit."""
    device = node_type.device
    if (node_graph is None) != (visit_graph is None):
        raise ValueError("node_graph and visit_graph must be supplied together")
    if node_graph is not None:
        node_graph = node_graph.to(device=device, dtype=torch.long)
        visit_graph = visit_graph.to(device=device, dtype=torch.long)
        if node_graph.ndim != 1 or node_graph.numel() != int(node_count):
            raise ValueError("node_graph must contain one graph id per node")
        if visit_graph.ndim != 1:
            raise ValueError("visit_graph must be a vector")
    if membership.ndim != 2 or membership.size(0) != 2:
        raise ValueError("visit_membership_index must have shape [2, pairs]")
    if membership.size(1) == 0:
        return _empty_pairs(device)
    membership = membership.to(device=device, dtype=torch.long)
    visit, node = membership[0], membership[1]
    if int(node.min()) < 0 or int(node.max()) >= int(node_count) or int(visit.min()) < 0:
        raise ValueError("visit membership refers to a node or visit outside the batch")
    if node_graph is not None:
        if int(visit.max()) >= visit_graph.numel():
            raise ValueError("visit membership refers to a visit outside the batch")
        if not torch.equal(visit_graph[visit], node_graph[node]):
            raise ValueError("visit membership crosses graph boundaries")
    evidence = torch.tensor(EVIDENCE_KIND_IDS, device=device)
    keep = torch.isin(node_type[node], evidence)
    visit, node = visit[keep], node[keep]
    if node.numel() < 2:
        return _empty_pairs(device)
    count = int(node_count)
    key = torch.unique(visit * count + node)
    visit, node = key // count, key % count
    _, sizes = torch.unique_consecutive(visit, return_counts=True)
    starts = torch.cumsum(sizes, 0) - sizes
    pair_chunks = []
    for width_tensor in torch.unique(sizes, sorted=True):
        width = int(width_tensor)
        if width < 2:
            continue
        selected_starts = starts[sizes == width]
        positions = selected_starts[:, None] + torch.arange(width, device=device)
        nodes = node[positions]
        left_slot, right_slot = torch.triu_indices(width, width, offset=1, device=device)
        left, right = nodes[:, left_slot], nodes[:, right_slot]
        pair_chunks.append((left * count + right).reshape(-1))
    if not pair_chunks:
        return _empty_pairs(device)
    pair_key = torch.unique(torch.cat(pair_chunks))
    return torch.stack((pair_key // count, pair_key % count))


def kind_pair_index(node_type, pairs):
    """Index 0..5 of the unordered evidence-kind pair of each node pair."""
    device = node_type.device
    lookup = torch.full((len(NODE_KINDS),), -1, dtype=torch.long, device=device)
    lookup[torch.tensor(EVIDENCE_KIND_IDS, device=device)] = torch.arange(
        len(EVIDENCE_KINDS), device=device)
    first, second = lookup[node_type[pairs[0]]], lookup[node_type[pairs[1]]]
    if bool((first < 0).any()) or bool((second < 0).any()):
        raise ValueError("pair endpoint is not an evidence node")
    low, high = torch.minimum(first, second), torch.maximum(first, second)
    kinds = len(EVIDENCE_KINDS)
    return low * kinds - low * (low - 1) // 2 + (high - low)


class PairEvidenceNetwork(nn.Module):
    """Exact continuous-input core of CEI-GNN v2."""

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden, token_dim,
                 num_triples, num_relations, dropout, pair_rank, pair_mode, num_node_types):
        super().__init__()
        if pair_mode not in PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {list(PAIR_MODES)}")
        self.num_tokens, self.node_dim, self.edge_dim = int(num_tokens), int(node_dim), int(edge_dim)
        self.num_classes, self.hidden = int(num_classes), int(hidden)
        self.token_dim, self.num_triples = int(token_dim), int(num_triples)
        self.num_relations = int(num_relations)
        self.dropout_rate, self.pair_rank, self.pair_mode = float(dropout), int(pair_rank), pair_mode
        classes, width = self.num_classes, self.hidden
        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(int(num_node_types), width)
        self.node_encoder = nn.Linear(self.node_dim + self.token_dim + width, width)
        self.node_norm = nn.LayerNorm(width)
        self.node_head = nn.Linear(width, classes * 2)
        self.relation_embedding = nn.Embedding(self.num_relations, width)
        self.triple_embedding = nn.Embedding(self.num_triples, width)
        self.edge_feature_projection = nn.Linear(self.edge_dim, width, bias=False)
        self.edge_source = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(self.dropout_rate), nn.Linear(width, classes))
        self.edge_target = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(self.dropout_rate), nn.Linear(width, classes))
        self.edge_context_vote = nn.Linear(width, classes)
        self.edge_context_gate = nn.Linear(width, classes)
        self.edge_aggregator = EdgeEvidenceAggregator(classes)
        self.edge_aggregator._edge_mask = None
        self.edge_aggregator._apply_sigmoid = False
        self.pair_projection = nn.Linear(width, int(pair_rank), bias=False)
        self.pair_vote = nn.Linear(int(pair_rank), classes)
        self.pair_gate = nn.Parameter(torch.zeros(KIND_PAIR_COUNT, classes))
        self.bias = nn.Parameter(torch.zeros(classes))

    @property
    def continuous_width(self):
        return self.node_dim + self.token_dim + self.hidden

    def continuous_inputs(self, clinical):
        return torch.cat((clinical.x, self.token_embedding(clinical.token),
                          self.node_type_embedding(clinical.node_type)), dim=-1)

    def set_edge_mask(self, mask):
        aggregator = self.edge_aggregator
        if mask is not None:
            if mask.ndim != 1 or not torch.isfinite(mask).all():
                raise ValueError("edge mask must be a finite vector")
            if (mask < 0).any() or (mask > 1).any():
                raise ValueError("edge mask values must be in [0, 1]")
        if "_edge_mask" in aggregator._parameters:
            del aggregator._parameters["_edge_mask"]
        aggregator._edge_mask = mask
        aggregator.explain = None if mask is None else True
        aggregator._apply_sigmoid = False if mask is not None else True

    def _validate(self, features, edge_index, metadata):
        node_count = int(features.size(0))
        if features.ndim != 2 or tuple(features.shape) != (node_count, self.continuous_width):
            raise ValueError("continuous features have an incompatible shape")
        if node_count == 0:
            raise ValueError("CEI-GNN v2 requires at least one node")
        if not torch.isfinite(features).all():
            raise ValueError("continuous features must be finite")
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be a long tensor with shape [2, edges]")
        edge_count = edge_index.size(1)
        edge_mask = self.edge_aggregator._edge_mask
        if edge_mask is not None and (edge_mask.ndim != 1 or edge_mask.numel() != edge_count):
            raise ValueError("edge mask length differs from the edge list")
        batch_index = metadata.batch_index
        if edge_count:
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count:
                raise ValueError("edge_index refers to a node outside the batch")
            if not torch.equal(batch_index[edge_index[0]], batch_index[edge_index[1]]):
                raise ValueError("cross-graph edges are not allowed")
            relation, triple = metadata.edge_relation, metadata.edge_triple
            if int(relation.min()) < 0 or int(relation.max()) >= self.num_relations:
                raise ValueError("edge_relation index is outside the fitted vocabulary")
            if int(triple.min()) < 0 or int(triple.max()) >= self.num_triples:
                raise ValueError("edge_triple index is outside the fitted vocabulary")
        if int(metadata.graph_count) < 1 or batch_index.numel() != node_count:
            raise ValueError("invalid graph membership metadata")
        if not torch.isfinite(metadata.edge_attr).all():
            raise ValueError("edge_attr must be finite")

    def forward_continuous(self, features, edge_index, metadata, membership, *,
                           return_parts=False, visit_graph=None):
        self._validate(features, edge_index, metadata)
        node_count, graph_count = int(features.size(0)), int(metadata.graph_count)
        batch_index, classes = metadata.batch_index, self.num_classes
        numeric, token_vectors, type_vectors = torch.split(features, (self.node_dim, self.token_dim, self.hidden), dim=-1)
        h = F.gelu(self.node_norm(self.node_encoder(torch.cat((numeric, token_vectors, type_vectors), dim=-1))))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
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
        if self.pair_mode == "off" or pair_count == 0:
            pair_total = h.new_zeros((graph_count, classes))
            pair_parts = h.new_zeros((pair_count, classes))
        else:
            z = torch.tanh(self.pair_projection(h))
            left, right = pairs
            q = z[left] * z[right] if self.pair_mode == "product" else z[left] + z[right]
            q = F.dropout(q, p=self.dropout_rate, training=self.training)
            pair_vote = self.pair_vote(q)
            pair_gate = self.pair_gate[kind_pair_index(metadata.node_type, pairs)].sigmoid()
            pair_graph = batch_index[left]
            pair_num = h.new_zeros((graph_count, classes)).index_add(0, pair_graph, pair_gate * pair_vote)
            pair_den = h.new_ones((graph_count, classes)).index_add(0, pair_graph, pair_gate)
            pair_total = pair_num / pair_den
            pair_parts = (pair_gate * pair_vote) / pair_den[pair_graph]

        logits = self.bias + node_num / node_den + edge_vote_graph / edge_denominator + pair_total
        if not return_parts:
            return logits
        return {"logits": logits, "node_contributions": node_parts, "edge_contributions": edge_parts,
                "pair_contributions": pair_parts, "pairs": pairs, "bias": self.bias}
