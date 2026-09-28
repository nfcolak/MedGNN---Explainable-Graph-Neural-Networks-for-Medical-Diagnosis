"""Temporary public-interface stub for the CEI-GNN v2 plugin red step."""
from __future__ import annotations

from argparse import Namespace

import torch

from .. import NODE_KINDS
from .base import ClinicalMethodAdapter, MethodOutput, read_clinical_batch
from .cei_gnn_v2 import PAIR_MODES, PairEvidenceNetwork


class PairEvidenceAdapter(ClinicalMethodAdapter):
    adaptation_version = "clinical_graph_v2_cei_gnn_v2_pairs"
    runner_defaults = {}
    grad_clip_value = 2.0

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        options = getattr(args, "method_options", {}) or {}
        self.num_tokens, self.node_dim, self.edge_dim = num_tokens, node_dim, edge_dim
        self.num_classes, self.hidden = num_classes, hidden
        self.layers_count, self.dropout_rate = layers, dropout
        self.token_dim, self.num_triples = token_dim, num_triples
        self.num_relations = 15
        self.pair_rank = int(options.get("pair_rank", 16))
        self.pair_mode = options.get("pair_mode", "product")
        self.network = PairEvidenceNetwork(num_tokens=num_tokens, node_dim=node_dim,
            edge_dim=edge_dim, num_classes=num_classes, hidden=hidden, token_dim=token_dim,
            num_triples=num_triples, num_relations=15, dropout=dropout,
            pair_rank=self.pair_rank, pair_mode=self.pair_mode, num_node_types=len(NODE_KINDS))

    def _read(self, batch):
        membership = getattr(batch, "visit_membership_index", None)
        if membership is None:
            raise ValueError("cei_gnn_v2 requires visit_membership_index")
        if membership.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("visit_membership_index must use an integer dtype")
        return read_clinical_batch(batch, method="cei_gnn_v2", node_dim=self.node_dim,
            edge_dim=self.edge_dim, num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations), membership.long()

    def continuous_inputs(self, batch):
        clinical, _ = self._read(batch)
        return self.network.continuous_inputs(clinical)

    def forward_continuous(self, features, edge_index, metadata, *, return_parts=False):
        clinical, membership = self._read(metadata)
        return self.network.forward_continuous(features, clinical.edge_index, clinical,
                                               membership, return_parts=return_parts)

    def forward(self, batch, *, epoch):
        del epoch
        clinical, _ = self._read(batch)
        logits = self.network.bias.expand(clinical.graph_count, -1) * 0.0
        return MethodOutput(logits=logits, auxiliary_loss=logits.sum() * 0.0,
                            diagnostics={})

    def on_epoch_start(self, epoch, train_loader):
        del epoch, train_loader

    def optimizer_groups(self, args):
        del args
        return [{"params": list(self.parameters())}]

    def run_config(self):
        return {"effective_settings": {}, "architecture": {}}

    def inactive_parameter_count(self):
        return 0


REGISTER = {"cei_gnn_v2": PairEvidenceAdapter}
