"""ProtoNode-WD: node-patch prototypes with a sparse clinical-token wide path."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import NODE_KINDS
from ..core.tensorize import PAYLOAD_WIDTH
from .base import (ClinicalMethodAdapter, MethodOutput, diagnostic_float, graph_mean,
                   method_setting, read_clinical_batch, relation_count)
from .protgnn import _RelationPayloadLayer

_NATIVE_DEFAULTS = {"max_epochs": 300, "lr": 1.79e-3, "batch_size": 128,
    "weight_decay": 4.3e-5, "patience": 10, "min_delta": 0.005,
    "warm_epochs": 10, "proj_epochs": 20, "proj_interval": 25,
    "nearest_graphs": 200, "prototypes_per_class": 3, "cluster_weight": 0.1,
    "separation_weight": 0.1, "margin": 1.0, "readout": "both", "wide": True,
    "wide_l1": 1e-4}


class ProtoNodeAdapter(ClinicalMethodAdapter):
    adaptation_version = "clinical_graph_v2_protonode_v1"
    epsilon = 1e-4
    cross_class_l1_coefficient = 5e-4

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        if min(num_tokens, node_dim, edge_dim, num_triples, num_classes, hidden, layers, token_dim) < 1:
            raise ValueError("model dimensions must be positive")
        if edge_dim < PAYLOAD_WIDTH:
            raise ValueError("edge_dim is narrower than the clinical numeric payload")
        self.num_tokens, self.node_dim, self.edge_dim = int(num_tokens), int(node_dim), int(edge_dim)
        self.num_classes, self.hidden = int(num_classes), int(hidden)
        self.layers_count, self.dropout_rate = int(layers), float(dropout)
        self.token_dim, self.num_triples = int(token_dim), int(num_triples)
        if not math.isfinite(self.dropout_rate) or not 0 <= self.dropout_rate < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.num_relations = relation_count(args)
        def setting(name, default, cast, minimum=None):
            return method_setting(args, ("protonode_" + name,), default, cast, minimum=minimum)
        self.warm_epochs = setting("warm_epochs", 10, int, 0)
        self.proj_epochs = setting("proj_epochs", 20, int, 0)
        if self.proj_epochs < self.warm_epochs:
            raise ValueError("projection cannot begin before the warm-up phase ends")
        self.proj_interval = setting("proj_interval", 25, int, 1)
        self.nearest_graphs = setting("nearest_graphs", 200, int, 1)
        self.prototypes_per_class = setting("prototypes_per_class", 3, int, 1)
        self.cluster_weight = setting("cluster_weight", 0.1, float, 0.0)
        self.separation_weight = setting("separation_weight", 0.1, float, 0.0)
        self.margin = setting("margin", 1.0, float, 0.0)
        self.readout = setting("readout", "both", str)
        if self.readout not in ("both", "node_max", "graph_mean"):
            raise ValueError("protonode_readout must be both, node_max, or graph_mean")
        self.wide = not bool(args.get("protonode_no_wide", False) if isinstance(args, Mapping)
                             else getattr(args, "protonode_no_wide", False))
        self.wide_l1 = setting("wide_l1", 1e-4, float, 0.0)
        self.num_prototypes = self.num_classes * self.prototypes_per_class
        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(len(NODE_KINDS), self.hidden)
        self.node_encoder = nn.Sequential(nn.Linear(self.node_dim + self.token_dim, self.hidden),
                                          nn.GELU(), nn.Linear(self.hidden, self.hidden))
        self.input_norm = nn.LayerNorm(self.hidden)
        self.relation_embedding = nn.Embedding(self.num_relations, self.hidden)
        self.triple_embedding = nn.Embedding(self.num_triples, self.hidden)
        self.edge_feature_projection = nn.Linear(self.edge_dim, self.hidden, bias=False)
        self.layers = nn.ModuleList([_RelationPayloadLayer(self.hidden, self.dropout_rate)
                                     for _ in range(self.layers_count)])
        self.prototype_vectors = nn.Parameter(torch.empty(self.num_prototypes, self.hidden))
        nn.init.normal_(self.prototype_vectors, mean=0.0, std=0.1)
        classes = torch.arange(self.num_prototypes) // self.prototypes_per_class
        identity = F.one_hot(classes, num_classes=self.num_classes).float()
        self.register_buffer("prototype_class_ids", classes)
        self.register_buffer("prototype_class_identity", identity)
        nfeatures = self.num_prototypes * (2 if self.readout == "both" else 1)
        self.prototype_classifier = nn.Linear(nfeatures, self.num_classes, bias=False)
        with torch.no_grad():
            correct = torch.cat([identity.t(), identity.t()], 1) if self.readout == "both" else identity.t()
            self.prototype_classifier.weight.copy_(correct + (-0.5) * (1.0 - correct))
        if self.wide:
            self.wide_token_votes = nn.Embedding(self.num_tokens, self.num_classes, padding_idx=0)
            nn.init.zeros_(self.wide_token_votes.weight)
        self.projection_summary = {"epoch": None, "projected_prototypes": 0,
            "unmatched_prototypes": self.num_prototypes, "candidate_graphs_by_class": {},
            "projected_node_type_ids": []}

    def _encode(self, batch):
        b = read_clinical_batch(batch, method="ProtoNode", node_dim=self.node_dim,
            edge_dim=self.edge_dim, num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations)
        nodes = self.node_encoder(torch.cat([b.x, self.token_embedding(b.token)], dim=-1))
        nodes = F.gelu(self.input_norm(nodes + self.node_type_embedding(b.node_type)))
        edge = (self.relation_embedding(b.edge_relation) + self.triple_embedding(b.edge_triple)
                + self.edge_feature_projection(b.edge_attr))
        for layer in self.layers:
            nodes = layer(nodes, b.edge_index, edge)
        means = graph_mean(nodes, b.batch_index, b.graph_count)
        return nodes, means, b.batch_index, b.graph_count, b.node_type, b.token

    def _distances(self, vectors):
        return (vectors.unsqueeze(1) - self.prototype_vectors.unsqueeze(0)).square().sum(-1).clamp_min(0)

    def _node_min_distances(self, nodes, graph_ids, graph_count):
        distances = self._distances(nodes)
        p = self.num_prototypes
        proto_ids = torch.arange(p, device=graph_ids.device)
        indices = (graph_ids[:, None] * p + proto_ids[None, :]).reshape(-1)
        flat = distances.new_full((graph_count * p,), float("inf"))
        flat.scatter_reduce_(0, indices, distances.reshape(-1), reduce="amin", include_self=True)
        result = flat.view(graph_count, p)
        return torch.where(torch.isfinite(result), result, result.new_full((), 1e6)), distances

    def _similarity(self, d):
        return torch.log((d + 1.0) / (d + self.epsilon))

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        nodes, means, graph_ids, graph_count, node_type, token = self._encode(batch)
        node_d, all_node_d = self._node_min_distances(nodes, graph_ids, graph_count)
        graph_d = self._distances(means)
        node_a, graph_a = self._similarity(node_d), self._similarity(graph_d)
        if self.readout == "both":
            features = torch.cat([node_a, graph_a], 1)
            active_distances = [node_d, graph_d]
        elif self.readout == "node_max":
            features, active_distances = node_a, [node_d]
        else:
            features, active_distances = graph_a, [graph_d]
        logits = self.prototype_classifier(features)
        if self.wide:
            votes = graph_mean(self.wide_token_votes(token), graph_ids, graph_count)
            logits = logits + votes
            wide_abs = self.wide_token_votes.weight.abs().mean()
        else:
            wide_abs = logits.sum() * 0.0
        labels = getattr(batch, "y", None)
        zero = logits.sum() * 0.0
        if labels is None:
            cluster = separation = zero
        else:
            labels = labels.view(-1).to(device=logits.device, dtype=torch.long)
            if labels.numel() != graph_count:
                raise ValueError("one target label is required per graph")
            if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
                raise ValueError("target label is outside the adapter's class range")
            own = self.prototype_class_ids[None, :] == labels[:, None]
            clusters, separations = [], []
            for distance in active_distances:
                clusters.append(distance.masked_fill(~own, torch.inf).min(1).values.mean())
                if self.num_classes > 1:
                    wrong = distance.masked_fill(own, torch.inf).min(1).values
                    separations.append(torch.relu(self.margin - wrong).mean())
                else:
                    separations.append(distance.sum() * 0.0)
            cluster, separation = torch.stack(clusters).mean(), torch.stack(separations).mean()
        wrong_weights = ~self.prototype_class_identity.t().bool()
        if self.readout == "both":
            wrong_weights = torch.cat([wrong_weights, wrong_weights], 1)
        cross_l1 = (self.prototype_classifier.weight * wrong_weights).abs().sum()
        auxiliary = self.cluster_weight * cluster + self.separation_weight * separation
        auxiliary = auxiliary + self.cross_class_l1_coefficient * cross_l1
        if self.wide:
            auxiliary = auxiliary + self.wide_l1 * wide_abs
        diagnostics = {"cluster_loss": diagnostic_float(cluster, "ProtoNode"),
            "separation_loss": diagnostic_float(separation, "ProtoNode"),
            "cross_class_l1": diagnostic_float(cross_l1, "ProtoNode"),
            "auxiliary_loss": diagnostic_float(auxiliary, "ProtoNode"),
            "mean_node_min_distance": diagnostic_float(node_d.mean(), "ProtoNode"),
            "wide_vote_abs_mean": diagnostic_float(wide_abs, "ProtoNode")}
        return MethodOutput(logits, auxiliary, diagnostics)

    def optimizer_groups(self, args):
        del args
        return [{"params": list(self.parameters())}]

    def _project_from_training_loader(self, loader, epoch):
        if loader is None:
            raise ValueError("ProtoNode projection requires the training loader")
        by_class = [[] for _ in range(self.num_classes)]
        device = next(self.parameters()).device
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                labels = getattr(batch, "y", None)
                if labels is None:
                    raise ValueError("training projection batches must contain labels")
                nodes, _, graph_ids, count, types, _ = self._encode(batch)
                labels = labels.view(-1).to(device=device, dtype=torch.long)
                if labels.numel() != count:
                    raise ValueError("training projection requires one label per graph")
                for gid, label in enumerate(labels.tolist()):
                    if label < 0 or label >= self.num_classes:
                        raise ValueError("training projection label is outside the class range")
                    if len(by_class[label]) < self.nearest_graphs:
                        mask = graph_ids == gid
                        by_class[label].append((nodes[mask].clone(), types[mask].clone()))
        types_projected, projected = [], 0
        with torch.no_grad():
            for class_id, graphs in enumerate(by_class):
                for pid in range(class_id * self.prototypes_per_class, (class_id + 1) * self.prototypes_per_class):
                    if not graphs:
                        continue
                    candidates = torch.cat([v for v, _ in graphs])
                    node_types = torch.cat([t for _, t in graphs])
                    distances = (candidates - self.prototype_vectors[pid]).square().sum(1)
                    index = int(distances.argmin().item())
                    self.prototype_vectors[pid].copy_(candidates[index])
                    types_projected.append(int(node_types[index].item()))
                    projected += 1
        self.projection_summary = {"epoch": int(epoch), "projected_prototypes": projected,
            "unmatched_prototypes": self.num_prototypes - projected,
            "candidate_graphs_by_class": {str(i): len(v) for i, v in enumerate(by_class)},
            "projected_node_type_ids": types_projected}

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        for p in self.parameters():
            p.requires_grad_(True)
        if epoch < self.warm_epochs:
            self.prototype_classifier.weight.requires_grad_(False)
        if epoch >= self.proj_epochs and (epoch - self.proj_epochs) % self.proj_interval == 0:
            was_training = self.training
            self.eval()
            try:
                self._project_from_training_loader(train_loader, epoch)
            finally:
                self.train(was_training)

    def run_config(self):
        total = sum(p.numel() for p in self.parameters())
        classifier = self.prototype_classifier.weight.numel()
        return {"method": "protonode", "adaptation_version": self.adaptation_version,
            "native_defaults": dict(_NATIVE_DEFAULTS),
            "effective_settings": {"warm_epochs": self.warm_epochs, "proj_epochs": self.proj_epochs,
                "proj_interval": self.proj_interval, "nearest_graphs": self.nearest_graphs,
                "prototypes_per_class": self.prototypes_per_class, "cluster_weight": self.cluster_weight,
                "separation_weight": self.separation_weight, "margin": self.margin,
                "readout": self.readout, "wide": self.wide, "wide_l1": self.wide_l1},
            "architecture": {"node_dim": self.node_dim, "edge_dim": self.edge_dim,
                "num_tokens": self.num_tokens, "num_triples": self.num_triples,
                "num_relations": self.num_relations, "num_node_types": len(NODE_KINDS),
                "num_classes": self.num_classes, "hidden": self.hidden, "layers": self.layers_count,
                "dropout": self.dropout_rate, "token_dim": self.token_dim,
                "num_prototypes": self.num_prototypes, "prototypes_per_class": self.prototypes_per_class,
                "prototype_classifier_bias": False, "parameter_count": total,
                "warmup_active_parameter_count": total - classifier, "joint_active_parameter_count": total},
            "native_schedule": {"optimizer": "Adam", "max_epochs": 300, "warm_epochs": self.warm_epochs,
                "proj_epochs": self.proj_epochs, "proj_interval": self.proj_interval,
                "nearest_graphs": self.nearest_graphs, "learning_rate": 1.79e-3,
                "batch_size": 128, "weight_decay": 4.3e-5, "patience": 10, "min_delta": 0.005},
            "objective_coefficients": {"cluster": self.cluster_weight, "separation": self.separation_weight,
                "cross_class_l1": self.cross_class_l1_coefficient, "wide_l1": self.wide_l1, "margin": self.margin},
            "mechanism_settings": {"readout": self.readout, "wide": self.wide,
                "node_prototype_distance": "squared_l2_min_over_nodes",
                "node_prototype_activation": "log((distance + 1) / (distance + epsilon))",
                "graph_readout": "mean", "projection_source": "training_loader_only",
                "base_papers": ["ProtoPNet (Chen et al., NeurIPS 2019)", "ProtGNN (Zhang et al., AAAI 2022)", "Wide & Deep (Cheng et al., 2016)"]}}
