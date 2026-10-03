"""Graph Multiset Transformer clinical adapter.

Uses the shared ProtGNN relation/payload clinical encoder, then Set Transformer
pooling. For fidelity to GMT's GMPool_G, the original paper uses GNN-based
keys/values; this adaptation simplifies those projections to linear K/V layers
on the already encoded node states.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from .. import NODE_KINDS
from ..core.tensorize import PAYLOAD_WIDTH
from .base import (ClinicalMethodAdapter, MethodOutput, diagnostic_float, graph_mean,
                   method_option, parameter_count, read_clinical_batch,
                   reject_unknown_options, relation_count)
from .protgnn import _RelationPayloadLayer


KNOWN_OPTIONS = frozenset(("seeds", "heads", "sab", "mean_skip"))
_OPTION_DEFAULTS = {"seeds": 8, "heads": 4, "sab": True, "mean_skip": True}


class _MAB(nn.Module):
    """Multihead attention block followed by Set Transformer residual norms/FFN."""

    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, hidden)
        self.attention_norm = nn.LayerNorm(hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden), nn.Dropout(dropout))
        self.feed_forward_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context, mask=None):
        batch, query_count, _ = query.shape
        context_count = context.size(1)
        q = self.query(query).view(batch, query_count, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(context).view(batch, context_count, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(context).view(batch, context_count, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().view(
            batch, query_count, self.hidden)
        hidden = self.attention_norm(query + self.dropout(self.output(attended)))
        result = self.feed_forward_norm(hidden + self.feed_forward(hidden))
        return result, weights


class _PMA(nn.Module):
    def __init__(self, hidden: int, heads: int, seed_count: int, dropout: float):
        super().__init__()
        self.seeds = nn.Parameter(torch.empty(1, seed_count, hidden))
        nn.init.xavier_uniform_(self.seeds)
        self.mab = _MAB(hidden, heads, dropout)

    def forward(self, nodes, mask):
        query = self.seeds.expand(nodes.size(0), -1, -1)
        return self.mab(query, nodes, mask)


class _SAB(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.mab = _MAB(hidden, heads, dropout)

    def forward(self, seeds):
        return self.mab(seeds, seeds)[0]


class GMTAdapter(ClinicalMethodAdapter):
    """ProtGNN clinical encoder with masked Graph Multiset Transformer readout."""

    adaptation_version = "clinical_graph_v2_gmt_v1"
    runner_defaults = dict(hidden=76, layers=3, dropout=0.3, lr=1.79e-3,
                           weight_decay=4.3e-5, batch_size=128, epochs=40,
                           patience=10, min_delta=0.005)
    grad_clip_value = 2.0

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        reject_unknown_options(args, KNOWN_OPTIONS)
        self.num_tokens, self.node_dim, self.edge_dim = int(num_tokens), int(node_dim), int(edge_dim)
        self.num_classes, self.hidden = int(num_classes), int(hidden)
        self.layers_count, self.dropout_rate = int(layers), float(dropout)
        self.token_dim, self.num_triples = int(token_dim), int(num_triples)
        self.num_relations = relation_count(args)
        if min(self.num_tokens, self.node_dim, self.edge_dim, self.num_classes,
               self.hidden, self.layers_count, self.token_dim, self.num_triples) < 1:
            raise ValueError("GMT dimensions and layer count must be positive")
        if self.edge_dim < PAYLOAD_WIDTH:
            raise ValueError("edge_dim is narrower than the clinical numeric payload")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.seeds = method_option(args, "seeds", _OPTION_DEFAULTS["seeds"], int,
                                   minimum=1, maximum=64)
        self.heads = method_option(args, "heads", _OPTION_DEFAULTS["heads"], int,
                                   minimum=1, maximum=16)
        self.sab_enabled = method_option(args, "sab", _OPTION_DEFAULTS["sab"], bool)
        self.mean_skip = method_option(args, "mean_skip", _OPTION_DEFAULTS["mean_skip"], bool)
        if self.hidden % self.heads:
            raise ValueError("hidden must be divisible by heads")

        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(len(NODE_KINDS), self.hidden)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim + self.token_dim, self.hidden), nn.GELU(),
            nn.Linear(self.hidden, self.hidden))
        self.input_norm = nn.LayerNorm(self.hidden)
        self.relation_embedding = nn.Embedding(self.num_relations, self.hidden)
        self.triple_embedding = nn.Embedding(self.num_triples, self.hidden)
        self.edge_feature_projection = nn.Linear(self.edge_dim, self.hidden, bias=False)
        self.layers = nn.ModuleList([_RelationPayloadLayer(self.hidden, self.dropout_rate)
                                     for _ in range(self.layers_count)])
        self.gmpool_g = _PMA(self.hidden, self.heads, self.seeds, self.dropout_rate)
        self.sab = _SAB(self.hidden, self.heads, self.dropout_rate) if self.sab_enabled else None
        self.gmpool_i = _PMA(self.hidden, self.heads, 1, self.dropout_rate)
        classifier_input = self.hidden * (2 if self.mean_skip else 1)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, self.hidden), nn.GELU(), nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden, self.num_classes))
        self._last_attention = None

    def _encode(self, batch):
        clinical = read_clinical_batch(
            batch, method="GMT", node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations)
        state = self.node_encoder(torch.cat(
            [clinical.x, self.token_embedding(clinical.token)], dim=-1))
        state = F.gelu(self.input_norm(state + self.node_type_embedding(clinical.node_type)))
        edge_context = (self.relation_embedding(clinical.edge_relation)
                        + self.triple_embedding(clinical.edge_triple)
                        + self.edge_feature_projection(clinical.edge_attr))
        for layer in self.layers:
            state = layer(state, clinical.edge_index, edge_context)
        dense, mask = to_dense_batch(state, clinical.batch_index,
                                     batch_size=clinical.graph_count)
        return state, dense, mask, clinical.batch_index, clinical.graph_count

    def _pool(self, dense, mask):
        pooled_seeds, attention = self.gmpool_g(dense, mask)
        if self.sab is not None:
            pooled_seeds = self.sab(pooled_seeds)
        graph_vector = self.gmpool_i(pooled_seeds, torch.ones(
            pooled_seeds.shape[:2], dtype=torch.bool, device=pooled_seeds.device))[0][:, 0]
        self._last_attention = attention
        return graph_vector, attention

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        state, dense, mask, graph_ids, graph_count = self._encode(batch)
        graph_vector, _ = self._pool(dense, mask)
        if self.mean_skip:
            graph_vector = torch.cat([graph_vector, graph_mean(state, graph_ids, graph_count)], dim=-1)
        logits = self.classifier(graph_vector)
        zero = logits.sum() * 0.0
        return MethodOutput(logits, zero, {"auxiliary_loss": diagnostic_float(zero, "gmt")})

    def explain(self, batch) -> torch.Tensor:
        """Mean GMPool_G attention over heads/seeds, restored to node order."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                _, dense, mask, _, _ = self._encode(batch)
                _, attention = self._pool(dense, mask)
                dense_scores = attention.mean(dim=(1, 2)) * mask.to(attention.dtype)
                return dense_scores[mask]
        finally:
            self.train(was_training)

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        del train_loader
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")

    def optimizer_groups(self, args):
        del args
        return [{"params": list(self.parameters())}]

    def early_stopping_start(self) -> int:
        return 20

    def run_config(self) -> dict:
        return {
            "method": "gmt", "adaptation_version": self.adaptation_version,
            "native_defaults": dict(self.runner_defaults, **_OPTION_DEFAULTS),
            "effective_settings": {"seeds": self.seeds, "heads": self.heads,
                                   "sab": self.sab_enabled, "mean_skip": self.mean_skip},
            "architecture": {"parameter_count": parameter_count(self),
                             "hidden": self.hidden, "layers": self.layers_count,
                             "dropout": self.dropout_rate, "node_dim": self.node_dim,
                             "edge_dim": self.edge_dim, "num_classes": self.num_classes,
                             "token_dim": self.token_dim},
            "mechanism_settings": {
                "base_paper": "Baek, Kang & Hwang, Accurate Learning of Graph Representations with Graph Multiset Pooling, ICLR 2021",
                "set_transformer": "Lee et al., Set Transformer, ICML 2019",
                "graph_pool": "masked PMA with learned seeds, optional one SAB, then one-seed PMA",
                "pma_keys_values": "linear projections on encoded node states (simplification; paper uses GNN-based keys/values)",
                "classifier_input": "concat GMT vector and mean-pooled vector" if self.mean_skip else "GMT vector",
            },
        }


REGISTER = {"gmt": GMTAdapter}
