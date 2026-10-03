"""Clinical relation/payload-aware GSAT adaptation.

The adapter keeps GSAT's shared GIN predictor, node-level graph-aware extractor,
binary-concrete attention, and Bernoulli information bottleneck. Clinical relation
identity, meta-relation identity, and numeric edge payload are added to every GIN
message before the sampled edge attention scales that message exactly once.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import InstanceNorm

from ... import NODE_KINDS
from ...core.tensorize import PAYLOAD_WIDTH
from ..base import (
    ClinicalMethodAdapter,
    MethodOutput,
    diagnostic_float,
    epoch_index,
    graph_mean,
    method_setting,
    parameter_count,
    read_clinical_batch,
    relation_count,
)


_NATIVE_DEFAULTS = {
    "temperature": 1.0,
    "info_loss_coef": 1.0,
    "init_r": 0.9,
    "final_r": 0.7,
    "decay_interval": 10,
    "decay_r": 0.1,
    "max_epochs": 100,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 128,
    "patience": 10,
    "min_delta": 0.005,
    "extractor_dropout": 0.5,
}


class _SingletonSafeBatchNorm1d(nn.BatchNorm1d):
    """Preserve BatchNorm while avoiding undefined one-row training statistics."""

    def forward(self, inputs):
        if self.training and inputs.ndim == 2 and inputs.size(0) == 1:
            return F.batch_norm(
                inputs,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                momentum=0.0,
                eps=self.eps,
            )
        return super().forward(inputs)


class _ClinicalGINLayer(nn.Module):
    """GIN sum aggregation with one relation/payload-aware message intervention."""

    def __init__(self, hidden: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            _SingletonSafeBatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, node_state, edge_index, edge_context, edge_attention=None):
        aggregate = torch.zeros_like(node_state)
        if edge_index.numel():
            source, target = edge_index
            messages = node_state[source] + edge_context
            if edge_attention is not None:
                if edge_attention.ndim != 1 or edge_attention.numel() != source.numel():
                    raise ValueError("GSAT edge attention must contain one value per edge")
                messages = messages * edge_attention.unsqueeze(-1)
            aggregate.index_add_(0, target, messages)
        return self.mlp(aggregate + (1.0 + self.eps) * node_state)


class _ClinicalGIN(nn.Module):
    """Shared GSAT predictor used once for extraction and once for prediction."""

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, num_relations):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.layers_count = int(layers)
        self.dropout_rate = float(dropout)
        self.token_dim = int(token_dim)
        self.num_triples = int(num_triples)
        self.num_relations = int(num_relations)

        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(len(NODE_KINDS), self.hidden)
        self.node_encoder = nn.Linear(self.node_dim + self.token_dim, self.hidden)
        self.input_norm = nn.LayerNorm(self.hidden)
        self.relation_embedding = nn.Embedding(self.num_relations, self.hidden)
        self.triple_embedding = nn.Embedding(self.num_triples, self.hidden)
        self.edge_feature_projection = nn.Linear(self.edge_dim, self.hidden, bias=False)
        self.layers = nn.ModuleList(
            _ClinicalGINLayer(self.hidden) for _ in range(self.layers_count)
        )
        self.classifier = nn.Linear(self.hidden, self.num_classes)

    def encode_inputs(self, clinical_batch):
        node_state = self.node_encoder(torch.cat([
            clinical_batch.x,
            self.token_embedding(clinical_batch.token),
        ], dim=-1))
        node_state = self.input_norm(
            node_state + self.node_type_embedding(clinical_batch.node_type)
        )
        node_state = F.relu(node_state)
        edge_context = (
            self.relation_embedding(clinical_batch.edge_relation)
            + self.triple_embedding(clinical_batch.edge_triple)
            + self.edge_feature_projection(clinical_batch.edge_attr)
        )
        return node_state, edge_context

    def get_emb(self, clinical_batch, edge_attention=None):
        node_state, edge_context = self.encode_inputs(clinical_batch)
        for layer in self.layers:
            node_state = layer(
                node_state,
                clinical_batch.edge_index,
                edge_context,
                edge_attention=edge_attention,
            )
            node_state = F.relu(node_state)
            node_state = F.dropout(
                node_state,
                p=self.dropout_rate,
                training=self.training,
            )
        return node_state

    def predict_from_emb(self, node_state, clinical_batch):
        pooled = graph_mean(
            node_state,
            clinical_batch.batch_index,
            clinical_batch.graph_count,
        )
        return self.classifier(pooled)

    def forward(self, clinical_batch, edge_attention=None):
        return self.predict_from_emb(
            self.get_emb(clinical_batch, edge_attention=edge_attention),
            clinical_batch,
        )


class _BatchSequential(nn.Sequential):
    """Sequential container that supplies graph membership to InstanceNorm."""

    def forward(self, inputs, batch_index):
        for module in self:
            if isinstance(module, InstanceNorm):
                inputs = module(inputs, batch_index)
            else:
                inputs = module(inputs)
        return inputs


class _NodeExtractor(nn.Module):
    """GSAT's node-level hidden -> 2*hidden -> hidden -> 1 extractor."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.mlp = _BatchSequential(
            nn.Linear(hidden, 2 * hidden),
            InstanceNorm(2 * hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
            InstanceNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_state, batch_index):
        return self.mlp(node_state, batch_index)


class GSATAdapter(ClinicalMethodAdapter):
    """Method-faithful node-attention GSAT over shared clinical graph tensors."""

    adaptation_version = "clinical_graph_v2_gsat_v1"

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        dimensions = {
            "num_tokens": num_tokens,
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "num_classes": num_classes,
            "hidden": hidden,
            "layers": layers,
            "token_dim": token_dim,
            "num_triples": num_triples,
        }
        if any(isinstance(value, bool) or int(value) != value or value < 1
               for value in dimensions.values()):
            raise ValueError("GSAT dimensions must be positive integers")
        if edge_dim < PAYLOAD_WIDTH:
            raise ValueError("edge_dim is narrower than the clinical numeric payload")
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

        self.temperature = method_setting(
            args, ("gsat_temperature", "temperature"),
            _NATIVE_DEFAULTS["temperature"], float, minimum=1e-12)
        self.info_loss_coef = method_setting(
            args, ("gsat_info_loss_coef", "info_loss_coef"),
            _NATIVE_DEFAULTS["info_loss_coef"], float, minimum=0.0)
        self.init_r = method_setting(
            args, ("gsat_init_r", "init_r"),
            _NATIVE_DEFAULTS["init_r"], float, minimum=1e-6, maximum=1.0 - 1e-6)
        self.final_r = method_setting(
            args, ("gsat_final_r", "final_r"),
            _NATIVE_DEFAULTS["final_r"], float, minimum=1e-6, maximum=1.0 - 1e-6)
        self.decay_interval = method_setting(
            args, ("gsat_decay_interval", "decay_interval"),
            _NATIVE_DEFAULTS["decay_interval"], int, minimum=1)
        self.decay_r = method_setting(
            args, ("gsat_decay_r", "decay_r"),
            _NATIVE_DEFAULTS["decay_r"], float, minimum=0.0)
        if self.final_r > self.init_r:
            raise ValueError("GSAT final_r cannot exceed init_r")
        self.max_epochs = method_setting(
            args, ("max_epochs", "epochs"),
            _NATIVE_DEFAULTS["max_epochs"], int, minimum=1)
        self.learning_rate = method_setting(
            args, ("lr", "learning_rate"), _NATIVE_DEFAULTS["lr"], float, minimum=0.0)
        self.weight_decay = method_setting(
            args, ("weight_decay",), _NATIVE_DEFAULTS["weight_decay"],
            float, minimum=0.0)
        self.batch_size = method_setting(
            args, ("batch_size",), _NATIVE_DEFAULTS["batch_size"], int, minimum=1)
        self.patience = method_setting(
            args, ("patience", "early_stopping"),
            _NATIVE_DEFAULTS["patience"], int, minimum=1)
        self.min_delta = method_setting(
            args, ("min_delta", "early_stop_min_delta"),
            _NATIVE_DEFAULTS["min_delta"], float, minimum=0.0)
        self.extractor_dropout = method_setting(
            args, ("gsat_extractor_dropout", "extractor_dropout"),
            _NATIVE_DEFAULTS["extractor_dropout"], float, minimum=0.0,
            maximum=1.0 - 1e-12)

        self.predictor = _ClinicalGIN(
            num_tokens=self.num_tokens,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            num_classes=self.num_classes,
            hidden=self.hidden,
            layers=self.layers_count,
            dropout=self.dropout_rate,
            token_dim=self.token_dim,
            num_triples=self.num_triples,
            num_relations=self.num_relations,
        )
        self.extractor = _NodeExtractor(self.hidden, self.extractor_dropout)
        self.last_node_attention = None
        self.last_edge_attention = None

    def _clinical_batch(self, batch):
        return read_clinical_batch(
            batch,
            method="GSAT",
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            num_tokens=self.num_tokens,
            num_triples=self.num_triples,
            num_relations=self.num_relations,
        )

    def r_for_epoch(self, epoch: int) -> float:
        epoch = epoch_index(epoch)
        return max(
            self.init_r - (epoch // self.decay_interval) * self.decay_r,
            self.final_r,
        )

    def _sample_attention(self, attention_logits):
        if not self.training:
            return attention_logits.sigmoid()
        uniform = torch.empty_like(attention_logits).uniform_(1e-10, 1.0 - 1e-10)
        gumbel = torch.log(uniform) - torch.log1p(-uniform)
        return torch.sigmoid((attention_logits + gumbel) / self.temperature)

    @staticmethod
    def _lift_node_attention(node_attention, edge_index):
        if edge_index.numel() == 0:
            return node_attention.new_empty((0,))
        source, target = edge_index
        return node_attention[source] * node_attention[target]

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        epoch = epoch_index(epoch)
        clinical_batch = self._clinical_batch(batch)
        extractor_embeddings = self.predictor.get_emb(clinical_batch)
        attention_logits = self.extractor(
            extractor_embeddings,
            clinical_batch.batch_index,
        )
        attention = self._sample_attention(attention_logits).squeeze(-1)
        edge_attention = self._lift_node_attention(
            attention,
            clinical_batch.edge_index,
        )
        logits = self.predictor(clinical_batch, edge_attention=edge_attention)

        r = self.r_for_epoch(epoch)
        probability = attention.clamp(1e-6, 1.0 - 1e-6)
        ib_loss = self.info_loss_coef * (
            probability * (probability / r).log()
            + (1.0 - probability)
            * ((1.0 - probability) / (1.0 - r)).log()
        ).mean()
        self.last_node_attention = attention
        self.last_edge_attention = edge_attention
        diagnostics = {
            "auxiliary_loss": diagnostic_float(ib_loss, "GSAT"),
            "information_bottleneck": diagnostic_float(ib_loss, "GSAT"),
            "r": float(r),
            "mean_node_attention": diagnostic_float(attention.detach().mean(), "GSAT"),
            "mean_edge_attention": diagnostic_float(
                edge_attention.detach().mean() if edge_attention.numel()
                else attention.detach().sum() * 0.0,
                "GSAT",
            ),
        }
        return MethodOutput(logits, ib_loss, diagnostics)

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        del train_loader
        epoch_index(epoch)

    def optimizer_groups(self, args) -> list[dict]:
        del args
        return [{"params": list(self.parameters())}]

    def run_config(self) -> dict:
        return {
            "method": "gsat",
            "adaptation_version": self.adaptation_version,
            "native_defaults": dict(_NATIVE_DEFAULTS),
            "effective_settings": {
                "temperature": self.temperature,
                "info_loss_coef": self.info_loss_coef,
                "init_r": self.init_r,
                "final_r": self.final_r,
                "decay_interval": self.decay_interval,
                "decay_r": self.decay_r,
                "extractor_dropout": self.extractor_dropout,
            },
            "architecture": {
                "node_dim": self.node_dim,
                "edge_dim": self.edge_dim,
                "num_tokens": self.num_tokens,
                "num_triples": self.num_triples,
                "num_relations": self.num_relations,
                "num_node_types": len(NODE_KINDS),
                "num_classes": self.num_classes,
                "hidden": self.hidden,
                "layers": self.layers_count,
                "dropout": self.dropout_rate,
                "token_dim": self.token_dim,
                "extractor": [self.hidden, 2 * self.hidden, self.hidden, 1],
                "extractor_dropout": self.extractor_dropout,
                "graph_readout": "mean",
                "parameter_count": parameter_count(self),
            },
            "native_schedule": {
                "optimizer": "Adam",
                "max_epochs": self.max_epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "patience": self.patience,
                "min_delta": self.min_delta,
            },
            "objective_coefficients": {
                "information_bottleneck": self.info_loss_coef,
            },
            "mechanism_settings": {
                "attention_level": "node",
                "temperature": self.temperature,
                "init_r": self.init_r,
                "final_r": self.final_r,
                "decay_interval": self.decay_interval,
                "decay_r": self.decay_r,
                "epoch_indexing": "zero_based",
                "stochastic_training_attention": True,
                "deterministic_eval_attention": True,
                "edge_attention": "alpha_source * alpha_target",
                "message_attention_applications": 1,
                "message_inputs": [
                    "sender_state",
                    "edge_relation",
                    "edge_triple",
                    "edge_attr",
                ],
                "edge_payload_width": PAYLOAD_WIDTH,
                "shared_predictor_for_extraction_and_prediction": True,
                "upstream_checkpoint_compatible": False,
            },
        }
