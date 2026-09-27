"""CAML-style label-wise attention over relation/payload-aware clinical nodes."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax

from .. import NODE_KINDS
from .base import (
    ClinicalMethodAdapter,
    MethodOutput,
    diagnostic_float,
    epoch_index,
    graph_mean,
    method_option,
    parameter_count,
    read_clinical_batch,
    reject_unknown_options,
    relation_count,
)
from .protgnn import _RelationPayloadLayer


_OPTIONS = {
    "mean_path": True,
    "attn_temperature": 1.0,
    "attn_dropout": 0.0,
}


class LabelAttentionAdapter(ClinicalMethodAdapter):
    """Shared ProtGNN clinical encoder with one CAML query per diagnosis class."""

    adaptation_version = "clinical_graph_v2_labelattn_v1"
    runner_defaults = dict(
        hidden=128, layers=3, dropout=0.3, lr=1.79e-3, weight_decay=4.3e-5,
        batch_size=128, epochs=40, patience=10, min_delta=0.005,
    )
    grad_clip_value = 2.0

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        reject_unknown_options(args, _OPTIONS)
        if min(num_tokens, node_dim, edge_dim, num_classes, hidden, layers,
               token_dim, num_triples) < 1:
            raise ValueError("all adapter dimensions must be positive")
        self.num_tokens = int(num_tokens)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.layers_count = int(layers)
        self.dropout_rate = float(dropout)
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.token_dim = int(token_dim)
        self.num_triples = int(num_triples)
        self.num_relations = relation_count(args)

        self.mean_path = method_option(args, "mean_path", _OPTIONS["mean_path"], bool)
        self.attn_temperature = method_option(
            args, "attn_temperature", _OPTIONS["attn_temperature"], float, minimum=0.0)
        if self.attn_temperature <= 0.0:
            raise ValueError("method option attn_temperature must be > 0")
        self.attn_dropout = method_option(
            args, "attn_dropout", _OPTIONS["attn_dropout"], float,
            minimum=0.0, maximum=1.0)
        if self.attn_dropout >= 1.0:
            raise ValueError("method option attn_dropout must be in [0, 1)")

        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(len(NODE_KINDS), self.hidden)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim + self.token_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.input_norm = nn.LayerNorm(self.hidden)
        self.relation_embedding = nn.Embedding(self.num_relations, self.hidden)
        self.triple_embedding = nn.Embedding(self.num_triples, self.hidden)
        self.edge_feature_projection = nn.Linear(self.edge_dim, self.hidden, bias=False)
        self.layers = nn.ModuleList([
            _RelationPayloadLayer(self.hidden, self.dropout_rate)
            for _ in range(self.layers_count)
        ])
        self.attention_query = nn.Parameter(torch.empty(self.num_classes, self.hidden))
        self.classifier_weight = nn.Parameter(torch.empty(self.num_classes, self.hidden))
        self.classifier_bias = nn.Parameter(torch.zeros(self.num_classes))
        nn.init.xavier_uniform_(self.attention_query)
        nn.init.xavier_uniform_(self.classifier_weight)
        self.mean_classifier = nn.Linear(self.hidden, self.num_classes) if self.mean_path else None
        self.last_attention = None

    def _encode(self, clinical):
        node_state = self.node_encoder(torch.cat(
            [clinical.x, self.token_embedding(clinical.token)], dim=-1))
        node_state = F.gelu(self.input_norm(
            node_state + self.node_type_embedding(clinical.node_type)))
        edge_context = (
            self.relation_embedding(clinical.edge_relation)
            + self.triple_embedding(clinical.edge_triple)
            + self.edge_feature_projection(clinical.edge_attr)
        )
        for layer in self.layers:
            node_state = layer(node_state, clinical.edge_index, edge_context)
        return node_state

    def _attention_readout(self, node_state, batch_index, graph_count):
        scores = (node_state @ self.attention_query.t()) / self.attn_temperature
        alpha = softmax(scores, batch_index, num_nodes=graph_count, dim=0)
        self.last_attention = alpha
        weights = F.dropout(alpha, p=self.attn_dropout, training=self.training)
        weighted_nodes = weights.unsqueeze(-1) * node_state.unsqueeze(1)
        pooled = node_state.new_zeros((graph_count, self.num_classes, self.hidden))
        pooled.index_add_(0, batch_index, weighted_nodes)
        logits = torch.einsum("bch,ch->bc", pooled, self.classifier_weight)
        logits = logits + self.classifier_bias
        return logits, alpha

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        clinical = read_clinical_batch(
            batch, method="labelattn", node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations,
        )
        node_state = self._encode(clinical)
        logits, alpha = self._attention_readout(
            node_state, clinical.batch_index, clinical.graph_count)
        if self.mean_classifier is not None:
            pooled_mean = graph_mean(node_state, clinical.batch_index, clinical.graph_count)
            logits = logits + self.mean_classifier(pooled_mean)
        zero = logits.sum() * 0.0
        diagnostics = {
            "attention_entropy": diagnostic_float(
                -(alpha.clamp_min(1e-12) * alpha.clamp_min(1e-12).log()).sum(0).mean(),
                "labelattn"),
            "mean_attention_max": diagnostic_float(alpha.max(), "labelattn"),
        }
        return MethodOutput(logits=logits, auxiliary_loss=zero, diagnostics=diagnostics)

    def explain(self, batch) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                clinical = read_clinical_batch(
                    batch, method="labelattn", node_dim=self.node_dim,
                    edge_dim=self.edge_dim, num_tokens=self.num_tokens,
                    num_triples=self.num_triples, num_relations=self.num_relations,
                )
                node_state = self._encode(clinical)
                _, alpha = self._attention_readout(
                    node_state, clinical.batch_index, clinical.graph_count)
                return alpha.max(dim=1).values
        finally:
            self.train(was_training)

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        del train_loader
        epoch_index(epoch)

    def optimizer_groups(self, args) -> list[dict]:
        del args
        return [{"params": list(self.parameters())}]

    def early_stopping_start(self) -> int:
        return 20

    def run_config(self) -> dict:
        defaults = dict(self.runner_defaults)
        defaults.update(_OPTIONS)
        return {
            "method": "labelattn",
            "adaptation_version": self.adaptation_version,
            "native_defaults": defaults,
            "effective_settings": {
                "mean_path": bool(self.mean_path),
                "attn_temperature": float(self.attn_temperature),
                "attn_dropout": float(self.attn_dropout),
            },
            "architecture": {
                "parameter_count": parameter_count(self),
                "node_dim": self.node_dim,
                "edge_dim": self.edge_dim,
                "num_tokens": self.num_tokens,
                "num_triples": self.num_triples,
                "num_relations": self.num_relations,
                "num_classes": self.num_classes,
                "hidden": self.hidden,
                "layers": self.layers_count,
                "dropout": self.dropout_rate,
                "token_dim": self.token_dim,
            },
            "mechanism_settings": {
                "base_paper": "Mullenbach et al., Explainable Prediction of Medical Codes from Clinical Text, NAACL 2018",
                "attention": "class-specific softmax over nodes, then weighted node sum",
                "class_scoring": "CAML label-wise context dot class-specific beta plus bias",
                "mean_path": bool(self.mean_path),
            },
        }


REGISTER = {"labelattn": LabelAttentionAdapter}
