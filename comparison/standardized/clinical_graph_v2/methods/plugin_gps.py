"""GraphGPS clinical adapter: local typed messages plus patient-local attention."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .. import NODE_KINDS
from .base import (
    ClinicalMethodAdapter,
    MethodOutput,
    diagnostic_float,
    graph_mean,
    method_option,
    parameter_count,
    read_clinical_batch,
    reject_unknown_options,
    relation_count,
)
from .protgnn import _RelationPayloadLayer


KNOWN_OPTIONS = ("rwse", "rwse_steps")
_OPTION_DEFAULTS = {"rwse": True, "rwse_steps": 8}


class GPSAdapter(ClinicalMethodAdapter):
    """GraphGPS-inspired layers over the shared clinical node/edge encoder."""

    adaptation_version = "clinical_graph_v2_gps_v1"
    runner_defaults = dict(
        hidden=80, layers=3, dropout=0.3, lr=1.79e-3, weight_decay=4.3e-5,
        batch_size=128, epochs=40, patience=10, min_delta=0.005,
    )
    grad_clip_value = 2.0

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        reject_unknown_options(args, KNOWN_OPTIONS)
        if min(num_tokens, node_dim, edge_dim, num_classes, hidden, layers,
               token_dim, num_triples) < 1:
            raise ValueError("GPS dimensions and layer count must be positive")
        self.num_tokens = int(num_tokens)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.layers_count = int(layers)
        self.dropout_rate = float(dropout)
        if not math.isfinite(self.dropout_rate) or not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.token_dim = int(token_dim)
        self.num_triples = int(num_triples)
        self.num_relations = relation_count(args)
        self.heads = 4
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by four attention heads")
        self.rwse = method_option(args, "rwse", True, bool)
        self.rwse_steps = method_option(args, "rwse_steps", 8, int, minimum=1)

        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim, padding_idx=0)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim + self.token_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.node_type_embedding = nn.Embedding(len(NODE_KINDS), self.hidden)
        self.input_norm = nn.LayerNorm(self.hidden)
        if self.rwse:
            self.rwse_projection = nn.Linear(self.rwse_steps, self.hidden)
        else:
            self.rwse_projection = None
        self.relation_embedding = nn.Embedding(self.num_relations, self.hidden)
        self.triple_embedding = nn.Embedding(self.num_triples, self.hidden)
        self.edge_feature_projection = nn.Linear(self.edge_dim, self.hidden, bias=False)
        self.local_layers = nn.ModuleList([
            _RelationPayloadLayer(self.hidden, self.dropout_rate)
            for _ in range(self.layers_count)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(self.hidden, self.heads, dropout=self.dropout_rate,
                                  batch_first=True)
            for _ in range(self.layers_count)
        ])
        self.global_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden) for _ in range(self.layers_count)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(self.hidden, 2 * self.hidden), nn.GELU(),
                          nn.Linear(2 * self.hidden, self.hidden))
            for _ in range(self.layers_count)
        ])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden) for _ in range(self.layers_count)
        ])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), nn.GELU(), nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden, self.num_classes),
        )
        self._last_attention = None

    def early_stopping_start(self) -> int:
        return 20

    def _random_walk_encoding(self, data):
        """Per-graph diagonal return probabilities for undirected random walks."""
        result = data.x.new_zeros((data.x.size(0), self.rwse_steps))
        edge_index = data.edge_index
        for graph_id in range(data.graph_count):
            node_ids = torch.nonzero(data.batch_index == graph_id, as_tuple=False).view(-1)
            if not node_ids.numel():
                continue
            local = torch.full((data.x.size(0),), -1, dtype=torch.long,
                               device=data.x.device)
            local[node_ids] = torch.arange(node_ids.numel(), device=data.x.device)
            keep = ((data.batch_index[edge_index[0]] == graph_id)
                    & (data.batch_index[edge_index[1]] == graph_id))
            edges = local[edge_index[:, keep]]
            n = node_ids.numel()
            adjacency = data.x.new_zeros((n, n))
            if edges.numel():
                adjacency[edges[0], edges[1]] = 1.0
                adjacency[edges[1], edges[0]] = 1.0
            degree = adjacency.sum(dim=1, keepdim=True)
            transition = adjacency / degree.clamp_min(1.0)
            walk = torch.eye(n, dtype=data.x.dtype, device=data.x.device)
            for step in range(self.rwse_steps):
                walk = walk @ transition
                result[node_ids, step] = walk.diagonal()
        return result

    def _input_state(self, data):
        state = self.node_encoder(torch.cat(
            [data.x, self.token_embedding(data.token)], dim=-1))
        state = F.gelu(self.input_norm(state + self.node_type_embedding(data.node_type)))
        if self.rwse_projection is not None:
            state = state + self.rwse_projection(self._random_walk_encoding(data))
        return state

    def _global_attention(self, state, batch_index, graph_count, layer_index,
                          *, return_weights=False):
        dense, valid = to_dense_batch(state, batch_index, batch_size=graph_count)
        output, weights = self.attention_layers[layer_index](
            dense, dense, dense, key_padding_mask=~valid,
            need_weights=True, average_attn_weights=False,
        )
        output = self.global_norms[layer_index](dense + self.dropout(output))
        output = output[valid]
        if return_weights:
            return output, weights, valid
        return output

    def _encode(self, data, *, capture_attention=False):
        state = self._input_state(data)
        attention_records = []
        edge_context = (self.relation_embedding(data.edge_relation)
                        + self.triple_embedding(data.edge_triple)
                        + self.edge_feature_projection(data.edge_attr))
        for index, local_layer in enumerate(self.local_layers):
            previous = state
            local = local_layer(previous, data.edge_index, edge_context)
            if capture_attention:
                global_state, weights, valid = self._global_attention(
                    previous, data.batch_index, data.graph_count, index,
                    return_weights=True)
                attention_records.append((weights, valid))
            else:
                global_state = self._global_attention(
                    previous, data.batch_index, data.graph_count, index)
            state = local + global_state
            state = self.ffn_norms[index](state + self.dropout(self.ffns[index](state)))
        self._last_attention = attention_records if capture_attention else None
        return state

    def _validated_batch(self, batch):
        return read_clinical_batch(
            batch, method="GPS", node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations,
        )

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        data = self._validated_batch(batch)
        node_state = self._encode(data)
        pooled = graph_mean(node_state, data.batch_index, data.graph_count)
        logits = self.classifier(pooled)
        zero = logits.sum() * 0.0
        return MethodOutput(
            logits=logits,
            auxiliary_loss=zero,
            diagnostics={"auxiliary_loss": diagnostic_float(zero, "GPS")},
        )

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        del train_loader
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")

    def optimizer_groups(self, args):
        del args
        return [{"params": list(self.parameters())}]

    def explain(self, batch) -> torch.Tensor:
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                data = self._validated_batch(batch)
                self._encode(data, capture_attention=True)
                scores = data.x.new_zeros((data.x.size(0),))
                for graph_id in range(data.graph_count):
                    node_ids = torch.nonzero(data.batch_index == graph_id,
                                             as_tuple=False).view(-1)
                    if not node_ids.numel():
                        continue
                    local_scores = []
                    for weights, valid in self._last_attention:
                        count = int(valid[graph_id].sum().item())
                        received = weights[graph_id, :, :count, :count].sum(dim=1).mean(dim=0)
                        local_scores.append(received)
                    scores[node_ids] = torch.stack(local_scores).mean(dim=0)
                return scores.clamp_min_(0.0)
        finally:
            self.train(was_training)

    def run_config(self) -> dict:
        count = parameter_count(self)
        return {
            "method": "gps",
            "adaptation_version": self.adaptation_version,
            "native_defaults": dict(self.runner_defaults, **_OPTION_DEFAULTS),
            "effective_settings": {"rwse": bool(self.rwse),
                                   "rwse_steps": int(self.rwse_steps)},
            "architecture": {"parameter_count": count, "hidden": self.hidden,
                             "layers": self.layers_count, "dropout": self.dropout_rate,
                             "heads": self.heads, "token_dim": self.token_dim,
                             "num_relations": self.num_relations,
                             "num_triples": self.num_triples,
                             "rwse_projection": self.rwse_projection is not None},
            "mechanism_settings": {
                "paper": "Recipe for a General, Powerful, Scalable Graph Transformer",
                "authors": "Rampasek et al.", "venue": "NeurIPS", "year": 2022,
                "local_messages": "ProtGNN relation/payload-conditioned residual mean message passing",
                "global_attention": "dense multi-head self-attention restricted to nodes of one patient graph",
                "random_walk_structural_encoding": "diagonal return probabilities of P^1..P^k on undirected graph",
                "readout": "mean pooling then MLP classifier",
            },
        }


REGISTER = {"gps": GPSAdapter}
