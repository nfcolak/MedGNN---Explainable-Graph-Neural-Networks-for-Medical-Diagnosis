"""Independent evidence-interaction adapter for the clinical graph runner."""
from __future__ import annotations

import torch

from .. import NODE_KINDS
from .base import (ClinicalMethodAdapter, MethodOutput, diagnostic_float,
                   method_option, parameter_count, read_clinical_batch,
                   relation_count, reject_unknown_options)
from .cei_gnn import EvidenceInteractionNetwork


KNOWN_OPTIONS = frozenset(("interaction_rank", "use_interactions"))


class EvidenceInteractionAdapter(ClinicalMethodAdapter):
    """Independent CEI-GNN with signed node and typed endpoint evidence."""

    adaptation_version = "clinical_graph_v2_cei_gnn_v1"
    runner_defaults = dict(hidden=128, layers=1, dropout=0.3, lr=1.79e-3,
                           weight_decay=4.3e-5, batch_size=128, epochs=40,
                           patience=10, min_delta=0.005)
    grad_clip_value = 2.0

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        reject_unknown_options(args, KNOWN_OPTIONS)
        dimensions = (num_tokens, node_dim, edge_dim, num_classes, hidden,
                      token_dim, num_triples)
        if any(isinstance(value, bool) or int(value) != value or int(value) < 1
               for value in dimensions):
            raise ValueError("model dimensions must be positive integers")
        if isinstance(layers, bool) or int(layers) != 1:
            raise ValueError("CEI-GNN v1 supports layers=1 only")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.num_tokens, self.node_dim, self.edge_dim = map(int, (num_tokens, node_dim, edge_dim))
        self.num_classes, self.hidden = map(int, (num_classes, hidden))
        self.layers_count, self.dropout_rate = int(layers), float(dropout)
        self.token_dim, self.num_triples = int(token_dim), int(num_triples)
        self.num_relations = relation_count(args)
        self.interaction_rank = method_option(
            args, "interaction_rank", 16, int, minimum=1)
        self.use_interactions = method_option(
            args, "use_interactions", True, bool)
        self.network = EvidenceInteractionNetwork(
            num_tokens=self.num_tokens, node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_classes=self.num_classes, hidden=self.hidden, token_dim=self.token_dim,
            num_triples=self.num_triples, num_relations=self.num_relations,
            dropout=self.dropout_rate, interaction_rank=self.interaction_rank,
            use_interactions=self.use_interactions, num_node_types=len(NODE_KINDS))

    def _read(self, batch):
        return read_clinical_batch(
            batch, method="cei_gnn", node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations)

    def continuous_inputs(self, batch) -> torch.Tensor:
        clinical = self._read(batch)
        return self.network.continuous_inputs(clinical)

    def forward_continuous(self, features, edge_index, metadata, *, return_parts=False):
        clinical = self._read(metadata)
        if edge_index.shape != clinical.edge_index.shape or not torch.equal(
                edge_index.to(clinical.edge_index.device), clinical.edge_index):
            raise ValueError("edge_index differs from the fixed metadata edge list")
        return self.network.forward_continuous(features, clinical.edge_index, clinical,
                                               return_parts=return_parts)

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        clinical = self._read(batch)
        features = self.network.continuous_inputs(clinical)
        parts = self.network.forward_continuous(
            features, clinical.edge_index, clinical, return_parts=True)
        logits = parts["logits"]
        auxiliary_loss = logits.sum() * 0.0
        return MethodOutput(logits=logits, auxiliary_loss=auxiliary_loss,
                            diagnostics={"auxiliary_loss": diagnostic_float(auxiliary_loss, "cei_gnn")})

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        del train_loader
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")

    def optimizer_groups(self, args):
        del args
        return [{"params": list(self.parameters())}]

    def run_config(self) -> dict:
        total = parameter_count(self)
        inactive = (self.network.endpoint_source.weight.numel()
                    + self.network.endpoint_target.weight.numel()
                    + self.network.interaction_context.weight.numel()
                    + self.network.interaction_context.bias.numel()) if not self.use_interactions else 0
        return {
            "method": "cei_gnn",
            "adaptation_version": self.adaptation_version,
            "native_defaults": dict(self.runner_defaults),
            "effective_settings": {"interaction_rank": self.interaction_rank,
                                   "use_interactions": self.use_interactions},
            "architecture": {
                "node_dim": self.node_dim, "edge_dim": self.edge_dim,
                "num_tokens": self.num_tokens, "num_triples": self.num_triples,
                "num_relations": self.num_relations, "num_node_types": len(NODE_KINDS),
                "num_classes": self.num_classes, "hidden": self.hidden,
                "layers": self.layers_count, "dropout": self.dropout_rate,
                "token_dim": self.token_dim, "interaction_rank": self.interaction_rank,
                "parameter_count": total, "active_parameter_count": total - inactive,
                "inactive_parameter_count": inactive,
            },
        }


REGISTER = {"cei_gnn": EvidenceInteractionAdapter}
