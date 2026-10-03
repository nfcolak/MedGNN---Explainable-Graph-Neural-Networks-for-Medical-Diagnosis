"""CEI-GNN v2 adapter: node, endpoint-additive edge and within-visit pair evidence."""
from __future__ import annotations

import torch

from core import NODE_KINDS
from core.method_base import (ClinicalMethodAdapter, MethodOutput, diagnostic_float, method_option,
                   parameter_count, read_clinical_batch, relation_count,
                   reject_unknown_options)
from .cei_gnn_v2 import PAIR_MODES, PairEvidenceNetwork

KNOWN_OPTIONS = frozenset(("pair_rank", "pair_mode"))
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_INDEX_FIELDS = ("token", "node_type", "edge_index", "edge_relation", "edge_triple", "batch",
                 "visit_membership_index")


class PairEvidenceAdapter(ClinicalMethodAdapter):
    """CEI-GNN v2 with a within-visit evidence-pair interaction block."""

    adaptation_version = "clinical_graph_v2_cei_gnn_v2_pairs"
    runner_defaults = dict(hidden=128, layers=1, dropout=0.3, lr=1.79e-3,
                           weight_decay=4.3e-5, batch_size=128, epochs=40,
                           patience=10, min_delta=0.005)
    grad_clip_value = 2.0

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        reject_unknown_options(args, KNOWN_OPTIONS)
        dimensions = (num_tokens, node_dim, edge_dim, num_classes, hidden, token_dim, num_triples)
        if any(isinstance(value, bool) or int(value) != value or int(value) < 1
               for value in dimensions):
            raise ValueError("model dimensions must be positive integers")
        if isinstance(layers, bool) or int(layers) != 1:
            raise ValueError("CEI-GNN v2 supports layers=1 only")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.num_tokens, self.node_dim, self.edge_dim = map(int, (num_tokens, node_dim, edge_dim))
        self.num_classes, self.hidden = map(int, (num_classes, hidden))
        self.layers_count, self.dropout_rate = int(layers), float(dropout)
        self.token_dim, self.num_triples = int(token_dim), int(num_triples)
        self.num_relations = relation_count(args)
        self.pair_rank = method_option(args, "pair_rank", 16, int, minimum=1)
        self.pair_mode = method_option(args, "pair_mode", "product", str, choices=PAIR_MODES)
        self.network = PairEvidenceNetwork(
            num_tokens=self.num_tokens, node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_classes=self.num_classes, hidden=self.hidden, token_dim=self.token_dim,
            num_triples=self.num_triples, num_relations=self.num_relations,
            dropout=self.dropout_rate, pair_rank=self.pair_rank, pair_mode=self.pair_mode,
            num_node_types=len(NODE_KINDS))

    def _read(self, batch):
        for field in _INDEX_FIELDS:
            value = getattr(batch, field, None)
            if value is not None and (not torch.is_tensor(value) or value.dtype not in _INTEGER_DTYPES):
                raise ValueError(f"{field} must use an integer dtype")
        membership = getattr(batch, "visit_membership_index", None)
        if membership is None:
            raise ValueError("cei_gnn_v2 requires visit_membership_index")
        visit_counts = getattr(batch, "num_visits", None)
        if (not torch.is_tensor(visit_counts)
                or visit_counts.dtype not in _INTEGER_DTYPES):
            raise ValueError("num_visits must be an integer tensor")
        clinical = read_clinical_batch(
            batch, method="cei_gnn_v2", node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations)
        visit_counts = visit_counts.to(device=clinical.x.device).view(-1)
        if visit_counts.numel() != clinical.graph_count or (visit_counts < 1).any():
            raise ValueError("num_visits must contain one positive count per graph")
        visit_graph = torch.repeat_interleave(
            torch.arange(clinical.graph_count, device=clinical.x.device), visit_counts.long())
        return clinical, membership.long(), visit_graph

    def continuous_inputs(self, batch) -> torch.Tensor:
        clinical, _, _ = self._read(batch)
        return self.network.continuous_inputs(clinical)

    def forward_continuous(self, features, edge_index, metadata, *, return_parts=False):
        clinical, membership, visit_graph = self._read(metadata)
        if edge_index.shape != clinical.edge_index.shape or not torch.equal(
                edge_index.to(clinical.edge_index.device), clinical.edge_index):
            raise ValueError("edge_index differs from the fixed metadata edge list")
        return self.network.forward_continuous(features, clinical.edge_index, clinical,
                                               membership, return_parts=return_parts,
                                               visit_graph=visit_graph)

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        clinical, membership, visit_graph = self._read(batch)
        features = self.network.continuous_inputs(clinical)
        parts = self.network.forward_continuous(features, clinical.edge_index, clinical,
                                                membership, return_parts=True,
                                                visit_graph=visit_graph)
        logits = parts["logits"]
        auxiliary_loss = logits.sum() * 0.0
        pairs_per_graph = parts["pairs"].size(1) / max(int(clinical.graph_count), 1)
        return MethodOutput(logits=logits, auxiliary_loss=auxiliary_loss, diagnostics={
            "auxiliary_loss": diagnostic_float(auxiliary_loss, "cei_gnn_v2"),
            "pairs_per_graph": float(pairs_per_graph)})

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        del train_loader
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")

    def optimizer_groups(self, args):
        del args
        return [{"params": list(self.parameters())}]

    def inactive_parameter_count(self) -> int:
        if self.pair_mode != "off":
            return 0
        net = self.network
        return int(net.pair_projection.weight.numel() + net.pair_vote.weight.numel()
                   + net.pair_vote.bias.numel() + net.pair_gate.numel())

    def run_config(self) -> dict:
        total, inactive = parameter_count(self), self.inactive_parameter_count()
        return {
            "method": "cei_gnn_v2",
            "adaptation_version": self.adaptation_version,
            "native_defaults": dict(self.runner_defaults),
            "effective_settings": {"pair_rank": self.pair_rank, "pair_mode": self.pair_mode},
            "architecture": {
                "node_dim": self.node_dim, "edge_dim": self.edge_dim,
                "num_tokens": self.num_tokens, "num_triples": self.num_triples,
                "num_relations": self.num_relations, "num_node_types": len(NODE_KINDS),
                "num_classes": self.num_classes, "hidden": self.hidden,
                "layers": self.layers_count, "dropout": self.dropout_rate,
                "token_dim": self.token_dim, "pair_rank": self.pair_rank,
                "parameter_count": total, "active_parameter_count": total - inactive,
                "inactive_parameter_count": inactive,
            },
        }


REGISTER = {"cei_gnn_v2": PairEvidenceAdapter}
