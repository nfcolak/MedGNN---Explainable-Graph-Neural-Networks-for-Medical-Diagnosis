"""CEI-GNN v3 adapter: v2 additive evidence + PLE + "no recorded result" block.

Stub (red step): loads the state file without validation and reports wrong values.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .. import NODE_KINDS
from ..cei_v3_absence import Universe
from ..cei_v3_ple import KnotTable
from .base import (ClinicalMethodAdapter, MethodOutput, diagnostic_float, method_option,
                   parameter_count, read_clinical_batch, relation_count)
from .cei_gnn_v3 import EvidenceNetworkV3
from .plugin_cei_gnn_v2 import _INDEX_FIELDS, _INTEGER_DTYPES

KNOWN_OPTIONS = frozenset(("arm", "v3_state", "k"))
STATE_VERSION = 'cei_v3_state_v1'
CONTROL_HIDDEN = 128


@dataclass(frozen=True)
class V3State:
    path: str
    sha256: str
    K: int
    knot_table: KnotTable
    universe: Universe
    vocabulary_tokens: tuple
    preprocessing_sha256: str
    node_feature_layout: tuple

    @property
    def num_tokens(self):
        return len(self.vocabulary_tokens) + 1

    def knot_row_of_token(self):
        return torch.full((self.num_tokens,), -1, dtype=torch.long)


def build_v3_state(*, K, knot_table, universe, vocabulary_tokens, vocabulary_min_count,
                   preprocessing_sha256, node_feature_layout):
    return {'version': STATE_VERSION, 'K': int(K)}


def load_v3_state(path):
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw.decode('utf-8'))
    return V3State(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), K=int(document['K']),
                   knot_table=KnotTable.load(document['knot_table']),
                   universe=Universe.load(document['universe']),
                   vocabulary_tokens=tuple(document['vocabulary']['tokens']),
                   preprocessing_sha256=str(document['preprocessing_sha256']),
                   node_feature_layout=tuple(document['node_feature_layout']))


class EvidenceAdapterV3(ClinicalMethodAdapter):
    """CEI-GNN v3 (stub)."""

    adaptation_version = "clinical_graph_v2_cei_gnn_v3_value_encoding"
    runner_defaults = dict(hidden=128, layers=1, dropout=0.3, lr=1.79e-3,
                           weight_decay=4.3e-5, batch_size=128, epochs=40,
                           patience=10, min_delta=0.005)
    grad_clip_value = 2.0

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        self.num_tokens, self.node_dim, self.edge_dim = map(int, (num_tokens, node_dim, edge_dim))
        self.num_classes, self.hidden = map(int, (num_classes, hidden))
        self.layers_count, self.dropout_rate = int(layers), float(dropout)
        self.token_dim, self.num_triples = int(token_dim), int(num_triples)
        self.num_relations = relation_count(args)
        self.edge_direction = 'forward'
        self.pair_rank = 16
        self.arm = 'A'
        self.k = method_option(args, "k", 0, int)
        self.encoder_depth, self.comorbid_block = 1, 0
        self.seed = int(getattr(args, 'seed', 0))
        self.state = load_v3_state(method_option(args, "v3_state", '', str))
        knots, active, _ = self.state.knot_table.tensor()
        self.network = EvidenceNetworkV3(
            num_tokens=self.num_tokens, node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_classes=self.num_classes, hidden=self.hidden, token_dim=self.token_dim,
            num_triples=self.num_triples, num_relations=self.num_relations,
            dropout=self.dropout_rate, pair_rank=self.pair_rank, num_node_types=len(NODE_KINDS),
            arm=self.arm, knots=knots, knot_active=active,
            slot_of_token=self.state.universe.slot_of_token(self.num_tokens),
            universe_size=len(self.state.universe), feature_layout=self.state.node_feature_layout,
            seed=self.seed, knot_row_of_token=self.state.knot_row_of_token())

    def _read(self, batch):
        for field in _INDEX_FIELDS:
            value = getattr(batch, field, None)
            if value is not None and (not torch.is_tensor(value) or value.dtype not in _INTEGER_DTYPES):
                raise ValueError(f"{field} must use an integer dtype")
        membership = getattr(batch, "visit_membership_index", None)
        if membership is None:
            raise ValueError("cei_gnn_v3 requires visit_membership_index")
        visit_counts = getattr(batch, "num_visits", None)
        if not torch.is_tensor(visit_counts) or visit_counts.dtype not in _INTEGER_DTYPES:
            raise ValueError("num_visits must be an integer tensor")
        clinical = read_clinical_batch(
            batch, method="cei_gnn_v3", node_dim=self.node_dim, edge_dim=self.edge_dim,
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
        return MethodOutput(logits=logits, auxiliary_loss=auxiliary_loss, diagnostics={
            "auxiliary_loss": diagnostic_float(auxiliary_loss, "cei_gnn_v3"),
            "pairs_per_graph": 0.0, "absent_items_per_graph": -1.0})

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        del train_loader
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")

    def optimizer_groups(self, args):
        del args
        return [{"params": list(self.parameters())}]

    def inactive_parameter_count(self) -> int:
        return 0

    def run_config(self) -> dict:
        total = parameter_count(self)
        return {"method": "cei_gnn_v3", "arm": self.arm, "k": self.k, "v3_state_sha256": '',
                "encoder_depth": 1, "comorbid_block": 0, "edge_direction": self.edge_direction,
                "hidden": self.hidden, "parameter_count": total, "active_parameter_count": total,
                "inactive_parameter_count": 0, "common_init_identical_to_c": False,
                "architecture": {"parameter_count": total, "active_parameter_count": total,
                                 "inactive_parameter_count": 0, "layers": 1},
                "parameter_inventory": {}, "effective_settings": {}}


REGISTER = {"cei_gnn_v3": EvidenceAdapterV3}
