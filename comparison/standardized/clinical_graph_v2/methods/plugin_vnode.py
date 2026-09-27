"""Clinical virtual-node (master-node) GNN plugin."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import NODE_KINDS
from ..tensorize import PAYLOAD_WIDTH
from .base import (
    ClinicalMethodAdapter,
    MethodOutput,
    diagnostic_float,
    graph_mean,
    method_option,
    read_clinical_batch,
    reject_unknown_options,
    relation_count,
)
from .protgnn import _RelationPayloadLayer


_RUNNER_DEFAULTS = {
    "hidden": 104,
    "layers": 3,
    "dropout": 0.3,
    "lr": 1.79e-3,
    "weight_decay": 4.3e-5,
    "batch_size": 128,
    "epochs": 40,
    "patience": 10,
    "min_delta": 0.005,
}
_KNOWN_OPTIONS = ("vn_pool",)


class VirtualNodeAdapter(ClinicalMethodAdapter):
    """ProtGNN clinical encoder with a graph-local, iteratively updated master node."""

    adaptation_version = "clinical_graph_v2_vnode_v1"
    runner_defaults = dict(_RUNNER_DEFAULTS)
    grad_clip_value = 2.0

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        if min(num_tokens, node_dim, edge_dim, num_classes, hidden, layers,
               token_dim, num_triples) < 1:
            raise ValueError("model dimensions must be positive")
        if edge_dim < PAYLOAD_WIDTH:
            raise ValueError("edge_dim is narrower than the clinical numeric payload")
        self.num_tokens, self.node_dim, self.edge_dim = int(num_tokens), int(node_dim), int(edge_dim)
        self.num_classes, self.hidden = int(num_classes), int(hidden)
        self.layers_count, self.token_dim = int(layers), int(token_dim)
        self.num_triples, self.num_relations = int(num_triples), relation_count(args)
        self.dropout_rate = float(dropout)
        if not math.isfinite(self.dropout_rate) or not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        reject_unknown_options(args, _KNOWN_OPTIONS)
        self.vn_pool = method_option(args, "vn_pool", "sum", str, choices=("sum", "mean"))

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
        self.virtual_node = nn.Parameter(torch.zeros(1, self.hidden))
        self.virtual_node_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden, 2 * self.hidden),
                nn.LayerNorm(2 * self.hidden),
                nn.GELU(),
                nn.Linear(2 * self.hidden, self.hidden),
                nn.LayerNorm(self.hidden),
                nn.GELU(),
            )
            for _ in range(max(0, self.layers_count - 1))
        ])
        self.virtual_node_dropouts = nn.ModuleList([
            nn.Dropout(self.dropout_rate) for _ in self.virtual_node_mlps
        ])
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.hidden, self.hidden),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden, self.num_classes),
        )

    def _encode(self, batch):
        clinical = read_clinical_batch(
            batch, method="vnode", node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations,
        )
        nodes = self.node_encoder(torch.cat(
            [clinical.x, self.token_embedding(clinical.token)], dim=-1))
        nodes = F.gelu(self.input_norm(
            nodes + self.node_type_embedding(clinical.node_type)))
        edge_context = (self.relation_embedding(clinical.edge_relation)
                        + self.triple_embedding(clinical.edge_triple)
                        + self.edge_feature_projection(clinical.edge_attr))
        virtual = self.virtual_node.expand(clinical.graph_count, -1)
        for layer_index, layer in enumerate(self.layers):
            nodes = nodes + virtual[clinical.batch_index]
            nodes = layer(nodes, clinical.edge_index, edge_context)
            if layer_index < len(self.virtual_node_mlps):
                pooled = graph_mean(nodes, clinical.batch_index, clinical.graph_count)
                if self.vn_pool == "sum":
                    counts = nodes.new_zeros((clinical.graph_count, 1))
                    counts.index_add_(0, clinical.batch_index, nodes.new_ones(
                        (clinical.batch_index.numel(), 1)))
                    pooled = pooled * counts
                virtual = virtual + self.virtual_node_dropouts[layer_index](
                    self.virtual_node_mlps[layer_index](pooled + virtual))
        means = graph_mean(nodes, clinical.batch_index, clinical.graph_count)
        return nodes, virtual, means, clinical.batch_index, clinical.graph_count

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        nodes, virtual, means, _, _ = self._encode(batch)
        logits = self.classifier(torch.cat([means, virtual], dim=-1))
        auxiliary_loss = logits.sum() * 0.0
        return MethodOutput(
            logits=logits,
            auxiliary_loss=auxiliary_loss,
            diagnostics={
                "virtual_node_norm": diagnostic_float(virtual.norm(dim=-1).mean(), "vnode"),
                "node_state_norm": diagnostic_float(nodes.norm(dim=-1).mean(), "vnode"),
            },
        )

    def explain(self, batch) -> torch.Tensor:
        """Return nonnegative cosine alignment of each node with its graph master node."""
        if self.training:
            raise RuntimeError("vnode explain() requires eval mode")
        with torch.no_grad():
            nodes, virtual, _, graph_ids, _ = self._encode(batch)
            return F.cosine_similarity(nodes, virtual[graph_ids], dim=-1).clamp_min(0.0)

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
            "method": "vnode",
            "adaptation_version": self.adaptation_version,
            "native_defaults": {**self.runner_defaults, "vn_pool": "sum"},
            "effective_settings": {"vn_pool": self.vn_pool},
            "architecture": {
                "node_dim": self.node_dim,
                "edge_dim": self.edge_dim,
                "num_tokens": self.num_tokens,
                "num_triples": self.num_triples,
                "num_relations": self.num_relations,
                "num_node_types": len(NODE_KINDS),
                "num_classes": self.num_classes,
                "parameter_count": sum(p.numel() for p in self.parameters()),
                "hidden": self.hidden,
                "layers": self.layers_count,
                "dropout": self.dropout_rate,
                "token_dim": self.token_dim,
            },
            "mechanism_settings": {
                "base_paper": "Neural Message Passing for Quantum Chemistry (Gilmer et al., ICML 2017), master node; Open Graph Benchmark (Hu et al., NeurIPS 2020), virtual-node GNN",
                "virtual_node_initialization": "learnable zero-initialized shared embedding expanded per graph",
                "virtual_node_pool": self.vn_pool,
                "virtual_node_update": "residual MLP after each layer except the final layer",
                "readout": "concat(graph_mean(node_state), final_virtual_node)",
                "explanation": "clamped nonnegative cosine similarity of final node and graph virtual-node states",
            },
        }


REGISTER = {"vnode": VirtualNodeAdapter}
