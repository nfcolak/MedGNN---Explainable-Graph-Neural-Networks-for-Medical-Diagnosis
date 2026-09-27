"""Wide & Deep clinical graph classifier with patient-level token counts."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import NODE_KINDS
from .base import (ClinicalMethodAdapter, MethodOutput, diagnostic_float, graph_mean,
                   method_option, parameter_count, read_clinical_batch, relation_count,
                   reject_unknown_options)
from .protgnn import _RelationPayloadLayer


KNOWN_OPTIONS = frozenset(("count_transform", "wide_l1", "tab_mlp", "tab_hidden"))
OPTION_DEFAULTS = {
    "count_transform": "log1p",
    "wide_l1": 1e-5,
    "tab_mlp": True,
    "tab_hidden": 64,
}


class TokenFusionAdapter(ClinicalMethodAdapter):
    """ProtGNN clinical encoder jointly trained with wide and tabular count paths."""

    adaptation_version = "clinical_graph_v2_tokenfusion_v1"
    runner_defaults = dict(hidden=64, layers=3, dropout=0.3, lr=1.79e-3,
                           weight_decay=4.3e-5, batch_size=128, epochs=40,
                           patience=10, min_delta=0.005)
    grad_clip_value = 2.0

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        reject_unknown_options(args, KNOWN_OPTIONS)
        if min(num_tokens, node_dim, edge_dim, num_classes, hidden, layers,
               token_dim, num_triples) < 1:
            raise ValueError("token, feature, class, hidden, layer, and triple dimensions must be positive")
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
        self.count_transform = method_option(
            args, "count_transform", OPTION_DEFAULTS["count_transform"], str,
            choices=("log1p", "binary", "raw"))
        self.wide_l1 = method_option(
            args, "wide_l1", OPTION_DEFAULTS["wide_l1"], float, minimum=0.0)
        self.tab_mlp_enabled = method_option(
            args, "tab_mlp", OPTION_DEFAULTS["tab_mlp"], bool)
        self.tab_hidden = method_option(
            args, "tab_hidden", OPTION_DEFAULTS["tab_hidden"], int, minimum=1)

        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(len(NODE_KINDS), self.hidden)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim + self.token_dim, self.hidden), nn.GELU(),
            nn.Linear(self.hidden, self.hidden))
        self.input_norm = nn.LayerNorm(self.hidden)
        self.relation_embedding = nn.Embedding(self.num_relations, self.hidden)
        self.triple_embedding = nn.Embedding(self.num_triples, self.hidden)
        self.edge_feature_projection = nn.Linear(self.edge_dim, self.hidden, bias=False)
        self.layers = nn.ModuleList([
            _RelationPayloadLayer(self.hidden, self.dropout_rate)
            for _ in range(self.layers_count)])
        self.deep_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), nn.GELU(),
            nn.Dropout(self.dropout_rate), nn.Linear(self.hidden, self.num_classes))
        self.wide = nn.Linear(self.num_tokens, self.num_classes)
        nn.init.zeros_(self.wide.weight)
        nn.init.zeros_(self.wide.bias)
        if self.tab_mlp_enabled:
            self.tab_head = nn.Sequential(
                nn.Linear(self.num_tokens, self.tab_hidden), nn.GELU(),
                nn.Dropout(0.3), nn.Linear(self.tab_hidden, self.num_classes))
            nn.init.zeros_(self.tab_head[-1].weight)
            nn.init.zeros_(self.tab_head[-1].bias)
        else:
            self.tab_head = None

    def _encode(self, batch):
        clinical = read_clinical_batch(
            batch, method="tokenfusion", node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations)
        state = self.node_encoder(torch.cat(
            (clinical.x, self.token_embedding(clinical.token)), dim=-1))
        state = F.gelu(self.input_norm(
            state + self.node_type_embedding(clinical.node_type)))
        context = (self.relation_embedding(clinical.edge_relation)
                   + self.triple_embedding(clinical.edge_triple)
                   + self.edge_feature_projection(clinical.edge_attr))
        for layer in self.layers:
            state = layer(state, clinical.edge_index, context)
        pooled = graph_mean(state, clinical.batch_index, clinical.graph_count)
        return clinical, state, pooled

    def _count_features(self, clinical):
        counts = clinical.x.new_zeros((clinical.graph_count, self.num_tokens))
        keep = clinical.token != 0
        if keep.any():
            graph_ids = clinical.batch_index[keep]
            token_ids = clinical.token[keep]
            counts.index_put_((graph_ids, token_ids), counts.new_ones(token_ids.shape),
                              accumulate=True)
        if self.count_transform == "log1p":
            return torch.log1p(counts)
        if self.count_transform == "binary":
            return counts.clamp_max(1.0)
        return counts

    def _forward_parts(self, batch):
        clinical, state, pooled = self._encode(batch)
        counts = self._count_features(clinical)
        deep_logits = self.deep_head(pooled)
        wide_logits = self.wide(counts)
        tab_logits = self.tab_head(counts) if self.tab_head is not None else torch.zeros_like(wide_logits)
        return clinical, state, counts, deep_logits, wide_logits, tab_logits

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        _, _, _, deep_logits, wide_logits, tab_logits = self._forward_parts(batch)
        logits = deep_logits + wide_logits + tab_logits
        auxiliary_loss = self.wide_l1 * self.wide.weight.abs().sum()
        diagnostics = {}
        magnitudes = [part.detach().abs().mean() for part in
                      (deep_logits, wide_logits, tab_logits)]
        denominator = sum(magnitudes).clamp_min(torch.finfo(logits.dtype).eps)
        for name, magnitude in zip(("deep", "wide", "tab"), magnitudes):
            diagnostics["logit_share_" + name] = diagnostic_float(magnitude / denominator, "tokenfusion")
        diagnostics["auxiliary_loss"] = diagnostic_float(auxiliary_loss, "tokenfusion")
        return MethodOutput(logits, auxiliary_loss, diagnostics)

    def explain(self, batch) -> torch.Tensor:
        """Allocate each graph's absolute wide-token evidence across its occurrences."""
        clinical = read_clinical_batch(
            batch, method="tokenfusion", node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations)
        with torch.no_grad():
            count_per_node = clinical.x.new_zeros(clinical.token.shape)
            keep = clinical.token != 0
            if keep.any():
                indices = (clinical.batch_index[keep], clinical.token[keep])
                counts = clinical.x.new_zeros((clinical.graph_count, self.num_tokens))
                counts.index_put_(indices, counts.new_ones(indices[0].shape), accumulate=True)
                count_per_node[keep] = counts[indices]
            token_weight = self.wide.weight.detach().abs().sum(dim=0)
            scores = clinical.x.new_zeros(clinical.token.shape)
            scores[keep] = token_weight[clinical.token[keep]] / count_per_node[keep].clamp_min(1.0)
            return scores

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
        native_defaults = dict(self.runner_defaults)
        native_defaults.update(OPTION_DEFAULTS)
        return {
            "method": "tokenfusion",
            "adaptation_version": self.adaptation_version,
            "native_defaults": native_defaults,
            "effective_settings": {
                "count_transform": self.count_transform,
                "wide_l1": float(self.wide_l1),
                "tab_mlp": bool(self.tab_mlp_enabled),
                "tab_hidden": int(self.tab_hidden),
            },
            "architecture": {
                "parameter_count": parameter_count(self),
                "node_dim": self.node_dim,
                "edge_dim": self.edge_dim,
                "num_tokens": self.num_tokens,
                "num_triples": self.num_triples,
                "num_relations": self.num_relations,
                "num_classes": self.num_classes,
                "token_dim": self.token_dim,
                "hidden": self.hidden,
                "layers": self.layers_count,
                "dropout": self.dropout_rate,
                "tab_hidden": self.tab_hidden,
            },
            "mechanism_settings": {
                "base_paper": "Wide & Deep Learning for Recommender Systems, DLRS 2016",
                "clinical_encoder": "ProtGNN relation/payload clinical encoder",
                "graph_readout": "mean",
                "count_feature": "per-patient token counts excluding padding, configured transform",
                "wide_feature": "linear count model with L1 auxiliary penalty",
                "joint_training": True,
            },
        }


REGISTER = {"tokenfusion": TokenFusionAdapter}
