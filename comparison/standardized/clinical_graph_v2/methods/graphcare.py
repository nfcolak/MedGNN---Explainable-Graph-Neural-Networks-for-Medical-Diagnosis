"""GraphCare-style BAT-GNN over the clinical v3 tensor contract.

This is an adaptation rather than an upstream-checkpoint-compatible port. It keeps
BAT-style sender and relation terms, but replaces vocabulary-square visit attention
with hidden-width projections over source-derived sparse visit membership. Numeric
node/context features and numeric edge payload remain active throughout the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter, softmax

from .. import NODE_KINDS
from ..core.tensorize import PAYLOAD_WIDTH
from .base import (
    ClinicalBatch,
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
    "max_epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "patience": 10,
    "min_delta": 0.005,
    "decay_rate": 0.03,
    "message_dropout": 0.5,
}


@dataclass(frozen=True)
class _VisitBatch:
    visit_membership_index: torch.Tensor
    visit_counts: torch.Tensor
    visit_graph: torch.Tensor
    visit_ordinal: torch.Tensor
    recency_rank: torch.Tensor
    global_node_mask: torch.Tensor
    visit_count: int


class _BATLayer(nn.Module):
    """BAT sender gating plus relation/payload-conditioned additive messages."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.relation_gate = nn.Linear(hidden, 1)
        self.update = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_state, edge_index, edge_context, node_attention):
        aggregate = torch.zeros_like(node_state)
        if edge_index.numel():
            source, target = edge_index
            sender = node_state[source] * node_attention[source].unsqueeze(-1)
            relation = self.relation_gate(edge_context) * edge_context
            messages = F.relu(sender + relation)
            aggregate.index_add_(0, target, messages)
        updated = self.update(aggregate + (1.0 + self.eps) * node_state)
        return F.relu(self.dropout(updated))


class GraphCareAdapter(ClinicalMethodAdapter):
    """Sparse-visit, relation/payload-aware GraphCare-style clinical adapter."""

    adaptation_version = "clinical_graph_v2_graphcare_v1"

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
            raise ValueError("GraphCare dimensions must be positive integers")
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
        self.recency_decay = method_setting(
            args, ("graphcare_decay_rate", "decay_rate"),
            _NATIVE_DEFAULTS["decay_rate"], float, minimum=0.0)
        self.message_dropout = method_setting(
            args, ("graphcare_message_dropout", "message_dropout"),
            _NATIVE_DEFAULTS["message_dropout"], float, minimum=0.0,
            maximum=1.0 - 1e-12)

        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(len(NODE_KINDS), self.hidden)
        self.node_projection = nn.Sequential(
            nn.Linear(self.node_dim + self.token_dim, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.input_norm = nn.LayerNorm(self.hidden)
        self.relation_embedding = nn.Embedding(self.num_relations, self.hidden)
        self.triple_embedding = nn.Embedding(self.num_triples, self.hidden)
        self.edge_feature_projection = nn.Linear(self.edge_dim, self.hidden, bias=False)

        self.alpha_projection = nn.ModuleList(
            nn.Linear(2 * self.hidden, 1) for _ in range(self.layers_count)
        )
        self.beta_projection = nn.ModuleList(
            nn.Linear(self.hidden, 1) for _ in range(self.layers_count)
        )
        self.layers = nn.ModuleList(
            _BATLayer(self.hidden, self.message_dropout)
            for _ in range(self.layers_count)
        )
        self.global_projection = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.direct_projection = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.ReLU(),
        )
        self.readout_dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(3 * self.hidden, self.num_classes)

        self.last_alpha = None
        self.last_beta = None
        self.last_visit_weight = None
        self.last_visit_state = None
        self.last_node_context = None
        self.last_graph_state = None
        self.last_direct_state = None
        self.last_global_state = None

    def _clinical_batch(self, batch) -> ClinicalBatch:
        return read_clinical_batch(
            batch,
            method="GraphCare",
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            num_tokens=self.num_tokens,
            num_triples=self.num_triples,
            num_relations=self.num_relations,
        )

    def _visit_batch(self, batch, clinical_batch: ClinicalBatch) -> _VisitBatch:
        membership = getattr(batch, "visit_membership_index", None)
        visit_counts = getattr(batch, "num_visits", None)
        global_node_mask = getattr(batch, "global_node_mask", None)
        if membership is None:
            raise ValueError("GraphCare requires visit_membership_index")
        if visit_counts is None:
            raise ValueError("GraphCare requires num_visits")
        if global_node_mask is None:
            raise ValueError("GraphCare requires global_node_mask")
        membership = membership.to(device=clinical_batch.x.device, dtype=torch.long)
        if membership.ndim != 2 or membership.size(0) != 2:
            raise ValueError("visit_membership_index must have shape [2, pairs]")
        visit_counts = visit_counts.to(
            device=clinical_batch.x.device,
            dtype=torch.long,
        ).view(-1)
        if visit_counts.numel() != clinical_batch.graph_count:
            raise ValueError("num_visits must contain one count per graph")
        if (visit_counts < 1).any():
            raise ValueError("every clinical graph must retain at least its index visit")
        visit_count = int(visit_counts.sum().item())
        global_node_mask = global_node_mask.to(
            device=clinical_batch.x.device,
            dtype=torch.bool,
        ).view(-1)
        if global_node_mask.numel() != clinical_batch.x.size(0):
            raise ValueError("global_node_mask must contain one flag per node")

        visit_graph = torch.repeat_interleave(
            torch.arange(
                clinical_batch.graph_count,
                device=clinical_batch.x.device,
            ),
            visit_counts,
        )
        visit_starts = visit_counts.cumsum(0) - visit_counts
        visit_ordinal = (
            torch.arange(visit_count, device=clinical_batch.x.device)
            - torch.repeat_interleave(visit_starts, visit_counts)
        )
        recency_rank = visit_counts[visit_graph] - 1 - visit_ordinal

        if membership.size(1):
            visit_index, node_index = membership
            if visit_index.min() < 0 or visit_index.max() >= visit_count:
                raise ValueError("visit membership refers to a visit outside the batch")
            if node_index.min() < 0 or node_index.max() >= clinical_batch.x.size(0):
                raise ValueError("visit membership refers to a node outside the batch")
            if not torch.equal(
                visit_graph[visit_index],
                clinical_batch.batch_index[node_index],
            ):
                raise ValueError("visit membership crosses graph boundaries")
            if global_node_mask[node_index].any():
                raise ValueError("global nodes cannot also carry visit membership")
            covered = torch.zeros_like(global_node_mask)
            covered[node_index] = True
        else:
            covered = torch.zeros_like(global_node_mask)
        if not torch.all(covered | global_node_mask):
            raise ValueError("every node must be visit-mapped or explicitly global")
        return _VisitBatch(
            visit_membership_index=membership,
            visit_counts=visit_counts,
            visit_graph=visit_graph,
            visit_ordinal=visit_ordinal,
            recency_rank=recency_rank,
            global_node_mask=global_node_mask,
            visit_count=visit_count,
        )

    def _encode_inputs(self, clinical_batch: ClinicalBatch):
        token_state = self.token_embedding(clinical_batch.token)
        node_state = self.node_projection(torch.cat([
            clinical_batch.x,
            token_state,
        ], dim=-1))
        node_state = self.input_norm(
            node_state + self.node_type_embedding(clinical_batch.node_type)
        )
        edge_context = (
            self.relation_embedding(clinical_batch.edge_relation)
            + self.triple_embedding(clinical_batch.edge_triple)
            + self.edge_feature_projection(clinical_batch.edge_attr)
        )
        return F.relu(node_state), edge_context

    @staticmethod
    def _visit_states(node_state, visit_batch: _VisitBatch):
        membership = visit_batch.visit_membership_index
        if membership.size(1) == 0:
            visit_state = node_state.new_zeros((visit_batch.visit_count, node_state.size(1)))
            nonempty = torch.zeros(
                visit_batch.visit_count,
                dtype=torch.bool,
                device=node_state.device,
            )
            return visit_state, nonempty
        visit_index, node_index = membership
        visit_state = scatter(
            node_state[node_index],
            visit_index,
            dim=0,
            dim_size=visit_batch.visit_count,
            reduce="mean",
        )
        pair_counts = torch.bincount(
            visit_index,
            minlength=visit_batch.visit_count,
        )
        return visit_state, pair_counts > 0

    def _visit_attention(self, node_state, visit_state, nonempty, visit_batch, layer):
        membership = visit_batch.visit_membership_index
        beta = node_state.new_zeros((visit_batch.visit_count,))
        if nonempty.any():
            beta_logits = (
                self.beta_projection[layer](visit_state).squeeze(-1)
                - self.recency_decay * visit_batch.recency_rank.to(node_state.dtype)
            )
            beta[nonempty] = softmax(
                beta_logits[nonempty],
                visit_batch.visit_graph[nonempty],
                num_nodes=visit_batch.visit_counts.numel(),
            )
        if membership.size(1) == 0:
            alpha = node_state.new_empty((0,))
            visit_weight = node_state.new_empty((0,))
            node_context = torch.zeros_like(node_state)
            node_attention = node_state.new_zeros((node_state.size(0),))
            return alpha, beta, visit_weight, node_context, node_attention

        visit_index, node_index = membership
        pair_state = torch.cat([
            node_state[node_index],
            visit_state[visit_index],
        ], dim=-1)
        alpha_scores = self.alpha_projection[layer](pair_state).squeeze(-1)
        alpha = softmax(
            alpha_scores,
            node_index,
            num_nodes=node_state.size(0),
        )
        visit_weight = alpha * beta[visit_index]
        node_context = scatter(
            visit_weight.unsqueeze(-1) * visit_state[visit_index],
            node_index,
            dim=0,
            dim_size=node_state.size(0),
            reduce="sum",
        )
        node_attention = scatter(
            visit_weight,
            node_index,
            dim=0,
            dim_size=node_state.size(0),
            reduce="sum",
        )
        return alpha, beta, visit_weight, node_context, node_attention

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        epoch_index(epoch)
        clinical_batch = self._clinical_batch(batch)
        visit_batch = self._visit_batch(batch, clinical_batch)
        node_state, edge_context = self._encode_inputs(clinical_batch)
        non_global_mask = ~visit_batch.global_node_mask
        knowledge_index = NODE_KINDS.index("knowledge")
        direct_mask = non_global_mask & (clinical_batch.node_type != knowledge_index)
        direct_state = graph_mean(
            self.direct_projection(node_state),
            clinical_batch.batch_index,
            clinical_batch.graph_count,
            mask=direct_mask,
        )

        if clinical_batch.edge_index.numel():
            source, target = clinical_batch.edge_index
            message_edge_mask = non_global_mask[source] & non_global_mask[target]
            message_edge_index = clinical_batch.edge_index[:, message_edge_mask]
            message_edge_context = edge_context[message_edge_mask]
        else:
            message_edge_index = clinical_batch.edge_index
            message_edge_context = edge_context

        visit_state, nonempty = self._visit_states(node_state, visit_batch)
        alpha = node_state.new_empty((0,))
        beta = node_state.new_zeros((visit_batch.visit_count,))
        visit_weight = node_state.new_empty((0,))
        node_context = torch.zeros_like(node_state)
        for layer_index, layer in enumerate(self.layers):
            visit_state, nonempty = self._visit_states(node_state, visit_batch)
            alpha, beta, visit_weight, node_context, node_attention = self._visit_attention(
                node_state,
                visit_state,
                nonempty,
                visit_batch,
                layer_index,
            )
            node_state = layer(
                node_state,
                message_edge_index,
                message_edge_context,
                node_attention,
            )

        graph_state = graph_mean(
            node_state,
            clinical_batch.batch_index,
            clinical_batch.graph_count,
            mask=non_global_mask,
        )
        global_state = graph_mean(
            self.global_projection(node_state),
            clinical_batch.batch_index,
            clinical_batch.graph_count,
            mask=visit_batch.global_node_mask,
        )
        logits = self.classifier(self.readout_dropout(torch.cat([
            graph_state,
            direct_state,
            global_state,
        ], dim=-1)))
        auxiliary_loss = logits.sum() * 0.0

        self.last_alpha = alpha
        self.last_beta = beta
        self.last_visit_weight = visit_weight
        self.last_visit_state = visit_state
        self.last_node_context = node_context
        self.last_graph_state = graph_state
        self.last_direct_state = direct_state
        self.last_global_state = global_state
        empty_visits = int((~nonempty).sum().item())
        diagnostics = {
            "auxiliary_loss": 0.0,
            "mean_alpha": diagnostic_float(
                alpha.detach().mean() if alpha.numel() else logits.detach().sum() * 0.0,
                "GraphCare",
            ),
            "mean_beta": diagnostic_float(
                beta.detach().mean() if beta.numel() else logits.detach().sum() * 0.0,
                "GraphCare",
            ),
            "empty_visits": float(empty_visits),
            "global_nodes": float(visit_batch.global_node_mask.sum().item()),
        }
        return MethodOutput(logits, auxiliary_loss, diagnostics)

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        del train_loader
        epoch_index(epoch)

    def optimizer_groups(self, args) -> list[dict]:
        del args
        return [{"params": list(self.parameters())}]

    def run_config(self) -> dict:
        return {
            "method": "graphcare",
            "adaptation_version": self.adaptation_version,
            "native_defaults": dict(_NATIVE_DEFAULTS),
            "effective_settings": {
                "recency_decay": self.recency_decay,
                "message_dropout": self.message_dropout,
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
                "message_dropout": self.message_dropout,
                "token_dim": self.token_dim,
                "attention_heads": 1,
                "patient_mode": "joint_plus_global",
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
                "method_native_auxiliary": 0.0,
            },
            "mechanism_settings": {
                "bat_message": "relu(sender * visit_attention + relation_gate(edge_context) * edge_context)",
                "message_inputs": [
                    "sender_state",
                    "edge_relation",
                    "edge_triple",
                    "edge_attr",
                ],
                "edge_payload_width": PAYLOAD_WIDTH,
                "visit_membership": "source-derived sparse pairs",
                "visit_attention_projection": "hidden-width alpha/beta projections per layer",
                "empty_visit_state": "zero",
                "empty_visit_beta_mass": 0.0,
                "empty_visit_ordinal_retained": True,
                "recency_rule": "beta_logit - decay_rate * (newest_ordinal - visit_ordinal)",
                "recency_decay": self.recency_decay,
                "direct_feature_fusion": "mean of pre-message token/type/numeric/context state",
                "direct_feature_scope": "non-global non-knowledge clinical nodes",
                "message_scope": "edges whose endpoints are both non-global",
                "graph_readout_scope": "non-global nodes",
                "global_context": "global nodes consumed only by separate masked projection",
                "absent_membership_is_global": False,
                "upstream_checkpoint_compatible": False,
                "epoch_indexing": "zero_based",
            },
        }
