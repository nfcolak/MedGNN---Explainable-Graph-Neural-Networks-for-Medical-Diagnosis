"""Clinical relation/payload-aware ProtGNN adaptation.

This keeps class-specific prototypes, distance activations, the bias-free
prototype classifier, bounded cluster/separation terms, warm-up, and scheduled
projection. The legacy GCN is replaced with a shared clinical encoder that consumes
node features/types, relation identity, meta-relation identity, and numeric edge
payload. Projection searches connected subsets in that encoder's latent space and
receives only the shared runner's training loader.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from core import NODE_KINDS
from core.tensorize import PAYLOAD_WIDTH
from core.method_base import ClinicalMethodAdapter, MethodOutput, relation_count


_NATIVE_DEFAULTS = {
    "max_epochs": 300,
    "warm_epochs": 10,
    "proj_epochs": 20,
    "proj_interval": 25,
    "nearest_graphs": 10,
    "prototypes_per_class": 3,
    "lr": 0.00179,
    "batch_size": 128,
    "weight_decay": 4.3e-5,
    "patience": 10,
    "min_delta": 0.005,
    "cluster_weight": 0.1,
    "separation_weight": 0.1,
    "margin": 1.0,
    "rollout": 3,
    "min_atoms": 3,
    "max_atoms": 6,
    "expand_atoms": 8,
    "c_puct": 5.0,
}


def _arg_value(args, names, default):
    """Read the first supplied, non-None override from a Namespace or mapping."""
    for name in names:
        if isinstance(args, Mapping):
            value = args.get(name)
        else:
            value = getattr(args, name, None) if args is not None else None
        if value is not None:
            return value
    return default


def _setting(args, names, default, cast, *, minimum=None):
    value = cast(_arg_value(args, names, default))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{names[0]} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{names[0]} must be >= {minimum}")
    return value


class _RelationPayloadLayer(nn.Module):
    """Residual mean-message layer with typed relation and continuous edge input."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_state, edge_index, edge_context, edge_mask=None):
        node_count = node_state.size(0)
        if edge_index.numel() == 0:
            aggregate = torch.zeros_like(node_state)
        else:
            source, target = edge_index
            messages = self.message_mlp(torch.cat(
                [node_state[source], node_state[target], edge_context], dim=-1))
            # `edge_mask` (e.g. GraphXAI's GNNExplainer, via the `edge_mask`
            # hook on the owning adapter's own forward) scales both the
            # message numerator AND its degree-normalisation weight, exactly
            # once -- not just the numerator -- so a masked-to-zero edge is
            # truly removed from the mean rather than merely diluting it, and
            # `edge_mask=None`/all-ones reproduces the original unweighted
            # mean exactly (the default keeps every existing caller of this
            # layer unchanged).
            if edge_mask is not None:
                if edge_mask.ndim != 1 or edge_mask.numel() != target.numel():
                    raise ValueError("edge_mask must contain one value per edge")
                weight = edge_mask.unsqueeze(-1)
                messages = messages * weight
            else:
                weight = node_state.new_ones((target.numel(), 1))
            aggregate = torch.zeros_like(node_state)
            aggregate.index_add_(0, target, messages)
            degree = node_state.new_zeros((node_count, 1))
            degree.index_add_(0, target, weight)
            aggregate = aggregate / degree.clamp_min_(1.0)
        update = self.update_mlp(torch.cat([node_state, aggregate], dim=-1))
        return self.norm(node_state + self.dropout(update))


class ProtGNNAdapter(ClinicalMethodAdapter):
    """Method-faithful prototype classifier adapted to clinical graph tensors."""

    adaptation_version = "clinical_graph_v2_protgnn_v1"
    epsilon = 1e-4
    cross_class_l1_coefficient = 5e-4

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        if num_tokens < 1 or node_dim < 1 or edge_dim < 1 or num_triples < 1:
            raise ValueError("token, node, edge, and triple dimensions must be positive")
        if hidden < 1 or layers < 1 or token_dim < 1:
            raise ValueError("hidden, layers, and token_dim must be positive")

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
        if self.edge_dim < PAYLOAD_WIDTH:
            raise ValueError("edge_dim is narrower than the clinical numeric payload")

        self.max_epochs = _setting(args, ("max_epochs", "epochs"),
                                   _NATIVE_DEFAULTS["max_epochs"], int, minimum=1)
        self.warm_epochs = _setting(args, ("warm_epochs",),
                                    _NATIVE_DEFAULTS["warm_epochs"], int, minimum=0)
        self.proj_epochs = _setting(args, ("proj_epochs",),
                                    _NATIVE_DEFAULTS["proj_epochs"], int, minimum=0)
        if self.proj_epochs < self.warm_epochs:
            raise ValueError("projection cannot begin before the warm-up phase ends")
        self.proj_interval = _setting(args, ("proj_interval",),
                                      _NATIVE_DEFAULTS["proj_interval"], int, minimum=1)
        self.nearest_graphs = _setting(args, ("nearest_graphs",),
                                       _NATIVE_DEFAULTS["nearest_graphs"], int, minimum=1)
        self.prototypes_per_class = _setting(
            args, ("prototypes_per_class", "num_prototypes_per_class"),
            _NATIVE_DEFAULTS["prototypes_per_class"], int, minimum=1)
        self.learning_rate = _setting(args, ("lr", "learning_rate"),
                                      _NATIVE_DEFAULTS["lr"], float, minimum=0.0)
        self.batch_size = _setting(args, ("batch_size",),
                                   _NATIVE_DEFAULTS["batch_size"], int, minimum=1)
        self.weight_decay = _setting(args, ("weight_decay",),
                                     _NATIVE_DEFAULTS["weight_decay"], float, minimum=0.0)
        self.patience = _setting(args, ("patience", "early_stopping"),
                                 _NATIVE_DEFAULTS["patience"], int, minimum=1)
        self.min_delta = _setting(args, ("min_delta", "early_stop_min_delta"),
                                  _NATIVE_DEFAULTS["min_delta"], float, minimum=0.0)
        self.cluster_weight = _setting(
            args, ("cluster_weight", "clst"), _NATIVE_DEFAULTS["cluster_weight"],
            float, minimum=0.0)
        self.separation_weight = _setting(
            args, ("separation_weight", "sep"), _NATIVE_DEFAULTS["separation_weight"],
            float, minimum=0.0)
        self.margin = _setting(args, ("margin",), _NATIVE_DEFAULTS["margin"],
                               float, minimum=0.0)
        self.rollout = _setting(args, ("rollout",), _NATIVE_DEFAULTS["rollout"],
                                int, minimum=1)
        self.min_atoms = _setting(args, ("min_atoms",), _NATIVE_DEFAULTS["min_atoms"],
                                  int, minimum=1)
        self.max_atoms = _setting(args, ("max_atoms",), _NATIVE_DEFAULTS["max_atoms"],
                                  int, minimum=1)
        self.expand_atoms = _setting(args, ("expand_atoms",),
                                     _NATIVE_DEFAULTS["expand_atoms"], int, minimum=1)
        self.c_puct = _setting(args, ("c_puct",), _NATIVE_DEFAULTS["c_puct"],
                               float, minimum=0.0)
        if self.min_atoms > self.max_atoms:
            raise ValueError("min_atoms cannot exceed max_atoms")

        self.num_prototypes = self.num_classes * self.prototypes_per_class
        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim,
                                            padding_idx=0)
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

        self.prototype_vectors = nn.Parameter(torch.empty(self.num_prototypes, self.hidden))
        nn.init.normal_(self.prototype_vectors, mean=0.0, std=0.1)
        prototype_class_ids = torch.arange(self.num_prototypes) // self.prototypes_per_class
        prototype_identity = F.one_hot(prototype_class_ids,
                                       num_classes=self.num_classes).to(torch.float32)
        self.register_buffer("prototype_class_ids", prototype_class_ids)
        self.register_buffer("prototype_class_identity", prototype_identity)
        self.prototype_classifier = nn.Linear(self.num_prototypes, self.num_classes,
                                              bias=False)
        with torch.no_grad():
            correct = prototype_identity.t()
            self.prototype_classifier.weight.copy_(correct + (-0.5) * (1.0 - correct))
        self.projection_summary = {
            "epoch": None,
            "projected_prototypes": 0,
            "unmatched_prototypes": self.num_prototypes,
            "candidate_graphs_by_class": {},
        }

    def _node_and_graph_embeddings(self, batch, edge_mask=None):
        x = getattr(batch, "x", None)
        token = getattr(batch, "token", None)
        node_type = getattr(batch, "node_type", None)
        edge_index = getattr(batch, "edge_index", None)
        edge_attr = getattr(batch, "edge_attr", None)
        edge_relation = getattr(batch, "edge_relation", None)
        edge_triple = getattr(batch, "edge_triple", None)
        if x is None:
            raise ValueError("ProtGNN requires x")
        if token is None:
            raise ValueError("ProtGNN requires token")
        if node_type is None:
            raise ValueError("ProtGNN requires node_type")
        if edge_index is None:
            raise ValueError("ProtGNN requires edge_index")
        if edge_attr is None:
            raise ValueError("ProtGNN requires edge_attr")
        if edge_relation is None:
            raise ValueError("ProtGNN requires edge_relation")
        if edge_triple is None:
            raise ValueError("ProtGNN requires edge_triple")
        if x.ndim != 2 or x.size(1) != self.node_dim:
            raise ValueError("clinical node feature width differs from adapter configuration")
        if token.ndim != 1 or token.numel() != x.size(0):
            raise ValueError("token indices must align with graph nodes")
        if node_type.ndim != 1 or node_type.numel() != x.size(0):
            raise ValueError("node_type indices must align with graph nodes")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        edge_count = edge_index.size(1)
        if edge_attr.ndim != 2 or edge_attr.shape != (edge_count, self.edge_dim):
            raise ValueError("edge_attr width differs from adapter configuration")
        if edge_relation.ndim != 1 or edge_relation.numel() != edge_count:
            raise ValueError("edge_relation indices must align with graph edges")
        if edge_triple.ndim != 1 or edge_triple.numel() != edge_count:
            raise ValueError("edge_triple indices must align with graph edges")
        if x.size(0) == 0:
            raise ValueError("ProtGNN does not accept a graph batch with no nodes")
        if node_type.numel() and (node_type.min() < 0 or node_type.max() >= len(NODE_KINDS)):
            raise ValueError("node_type index is outside the clinical node-kind vocabulary")
        if edge_relation.numel() and (edge_relation.min() < 0 or
                                      edge_relation.max() >= self.num_relations):
            raise ValueError("edge_relation index is outside the clinical relation vocabulary")
        if edge_triple.numel() and (edge_triple.min() < 0 or
                                    edge_triple.max() >= self.num_triples):
            raise ValueError("edge_triple index is outside the fitted meta-relation vocabulary")

        node_state = self.node_encoder(torch.cat(
            [x, self.token_embedding(token.long())], dim=-1))
        node_state = self.input_norm(node_state + self.node_type_embedding(node_type.long()))
        node_state = F.gelu(node_state)
        edge_context = (self.relation_embedding(edge_relation.long())
                        + self.triple_embedding(edge_triple.long())
                        + self.edge_feature_projection(edge_attr))
        for layer in self.layers:
            node_state = layer(node_state, edge_index.long(), edge_context, edge_mask=edge_mask)

        batch_index = getattr(batch, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            graph_count = 1
        else:
            batch_index = batch_index.to(device=x.device, dtype=torch.long)
            graph_count = int(batch_index.max().item()) + 1
            declared_count = getattr(batch, "num_graphs", None)
            if declared_count is not None:
                graph_count = max(graph_count, int(declared_count))
        graph_sum = node_state.new_zeros((graph_count, self.hidden))
        graph_sum.index_add_(0, batch_index, node_state)
        graph_count_per_graph = node_state.new_zeros((graph_count, 1))
        graph_count_per_graph.index_add_(
            0, batch_index, node_state.new_ones((batch_index.numel(), 1)))
        graph_embedding = graph_sum / graph_count_per_graph.clamp_min_(1.0)
        return node_state, graph_embedding, batch_index

    def _prototype_activations(self, graph_embedding):
        distances = torch.cdist(graph_embedding, self.prototype_vectors, p=2).square()
        distances = distances.clamp_min(0.0)
        activations = torch.log((distances + 1.0) / (distances + self.epsilon))
        return activations, distances

    @staticmethod
    def _diagnostic_float(value):
        result = float(value.detach().cpu().item())
        if not math.isfinite(result):
            raise FloatingPointError("ProtGNN produced a non-finite diagnostic")
        return result

    def forward(self, batch, *, epoch: int, external_edge_mask=None) -> MethodOutput:
        del epoch  # The shared epoch hook owns phase changes and projection.
        _, graph_embedding, _ = self._node_and_graph_embeddings(
            batch, edge_mask=external_edge_mask
        )
        activations, distances = self._prototype_activations(graph_embedding)
        logits = self.prototype_classifier(activations)

        labels = getattr(batch, "y", None)
        if labels is None:
            zero = logits.sum() * 0.0
            cluster_loss = separation_loss = zero
        else:
            labels = labels.view(-1).to(device=distances.device, dtype=torch.long)
            if labels.numel() != graph_embedding.size(0):
                raise ValueError("one target label is required per graph")
            if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
                raise ValueError("target label is outside the adapter's class range")
            correct = self.prototype_class_ids.unsqueeze(0) == labels.unsqueeze(1)
            cluster_loss = distances.masked_fill(~correct, torch.inf).min(dim=1).values.mean()
            if self.num_classes > 1:
                wrong_min = distances.masked_fill(correct, torch.inf).min(dim=1).values
                separation_loss = torch.relu(self.margin - wrong_min).mean()
            else:
                separation_loss = distances.sum() * 0.0
        incorrect_connections = ~self.prototype_class_identity.t().bool()
        cross_class_l1 = (self.prototype_classifier.weight * incorrect_connections).abs().sum()
        auxiliary_loss = (self.cluster_weight * cluster_loss
                          + self.separation_weight * separation_loss
                          + self.cross_class_l1_coefficient * cross_class_l1)
        diagnostics = {
            "cluster_loss": self._diagnostic_float(cluster_loss),
            "separation_loss": self._diagnostic_float(separation_loss),
            "cross_class_l1": self._diagnostic_float(cross_class_l1),
            "auxiliary_loss": self._diagnostic_float(auxiliary_loss),
            "mean_prototype_distance": self._diagnostic_float(distances.detach().mean()),
        }
        return MethodOutput(logits, auxiliary_loss, diagnostics)

    def optimizer_groups(self, args) -> list[dict]:
        """Return Adam-compatible parameters; the shared runner supplies lr/decay."""
        del args
        return [{"params": list(self.parameters())}]

    def _connected(self, coalition, adjacency):
        if len(coalition) <= 1:
            return True
        allowed = set(coalition)
        visited = {coalition[0]}
        stack = [coalition[0]]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in allowed and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return len(visited) == len(allowed)

    def _initial_coalition(self, node_state, edge_index, prototype):
        node_count = node_state.size(0)
        distances = (node_state - prototype.unsqueeze(0)).square().sum(dim=1)
        adjacency = [set() for _ in range(node_count)]
        for source, target in edge_index.detach().cpu().t().tolist():
            if source != target:
                adjacency[source].add(target)
                adjacency[target].add(source)

        components = []
        unseen = set(range(node_count))
        while unseen:
            start = min(unseen)
            component = {start}
            stack = [start]
            unseen.remove(start)
            while stack:
                current = stack.pop()
                for neighbor in adjacency[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)
            components.append(component)
        eligible = [component for component in components
                    if len(component) >= self.min_atoms]
        if not eligible:
            return (), distances
        component = min(
            eligible,
            key=lambda nodes: min((float(distances[node].item()), node) for node in nodes),
        )
        if len(component) <= self.max_atoms:
            return tuple(sorted(component)), distances
        seed = min(component, key=lambda node: (float(distances[node].item()), node))
        selected = [seed]
        selected_set = {seed}
        while len(selected) < self.max_atoms:
            frontier = {neighbor for node in selected for neighbor in adjacency[node]
                        if neighbor in component and neighbor not in selected_set}
            if not frontier:
                break
            next_node = min(frontier, key=lambda node: (float(distances[node].item()), node))
            selected.append(next_node)
            selected_set.add(next_node)
        return tuple(sorted(selected)), distances

    def _projection_candidate(self, batch, graph_id, node_state, graph_index):
        """Copy one graph's raw clinical tensors for induced-subgraph projection."""
        node_ids = torch.nonzero(graph_index == graph_id, as_tuple=False).view(-1)
        if node_ids.numel() == 0:
            return None
        node_to_local = torch.full(
            (node_state.size(0),), -1, dtype=torch.long, device=node_state.device)
        node_to_local[node_ids] = torch.arange(node_ids.numel(), device=node_state.device)
        edge_index = batch.edge_index.long()
        edge_keep = ((graph_index[edge_index[0]] == graph_id)
                     & (graph_index[edge_index[1]] == graph_id))
        return {
            "node_state": node_state.index_select(0, node_ids).detach().clone(),
            "x": batch.x.index_select(0, node_ids).detach().clone(),
            "token": batch.token.index_select(0, node_ids).detach().clone(),
            "node_type": batch.node_type.index_select(0, node_ids).detach().clone(),
            "edge_index": node_to_local[edge_index[:, edge_keep]].detach().clone(),
            "edge_attr": batch.edge_attr[edge_keep].detach().clone(),
            "edge_relation": batch.edge_relation[edge_keep].detach().clone(),
            "edge_triple": batch.edge_triple[edge_keep].detach().clone(),
        }

    def _induced_graph_embedding(self, candidate, coalition):
        """Re-encode an induced coalition so removed nodes cannot leak messages."""
        indices = torch.as_tensor(
            coalition, dtype=torch.long, device=candidate["x"].device)
        local = torch.full(
            (candidate["x"].size(0),), -1, dtype=torch.long,
            device=candidate["x"].device)
        local[indices] = torch.arange(indices.numel(), device=indices.device)
        source, target = candidate["edge_index"]
        edge_keep = (local[source] >= 0) & (local[target] >= 0)
        data = Data(
            x=candidate["x"].index_select(0, indices),
            token=candidate["token"].index_select(0, indices),
            node_type=candidate["node_type"].index_select(0, indices),
            edge_index=local[candidate["edge_index"][:, edge_keep]],
            edge_attr=candidate["edge_attr"][edge_keep],
            edge_relation=candidate["edge_relation"][edge_keep],
            edge_triple=candidate["edge_triple"][edge_keep],
        )
        _, graph_embedding, _ = self._node_and_graph_embeddings(data)
        return graph_embedding[0]

    def _mcts_project(self, candidate, prototype):
        """Search connected induced clinical subgraphs; never reads other splits."""
        node_state = candidate["node_state"]
        edge_index = candidate["edge_index"]
        root, _ = self._initial_coalition(node_state, edge_index, prototype)
        if not root:
            return None, float("-inf"), ()
        node_count = node_state.size(0)
        adjacency = [set() for _ in range(node_count)]
        for source, target in edge_index.detach().cpu().t().tolist():
            if source != target:
                adjacency[source].add(target)
                adjacency[target].add(source)

        def score(coalition):
            embedding = self._induced_graph_embedding(candidate, coalition)
            distance = (embedding - prototype).square().sum()
            similarity = torch.log((distance + 1.0) / (distance + self.epsilon))
            return float(similarity.item()), embedding

        root_score, root_embedding = score(root)
        # state -> [visit count, accumulated reward, prior reward]
        stats = {root: [1, root_score, max(root_score, 0.0)]}
        embeddings = {root: root_embedding}
        rewards = {root: root_score}
        best_state, best_score = root, root_score
        for _ in range(self.rollout):
            coalition = root
            while len(coalition) > self.min_atoms:
                active = set(coalition)
                removable = []
                for node in coalition:
                    child = tuple(index for index in coalition if index != node)
                    if len(child) >= self.min_atoms and self._connected(child, adjacency):
                        degree = len(adjacency[node] & active)
                        removable.append((degree, node, child))
                removable.sort(key=lambda item: (item[0], item[1]))
                removable = removable[:self.expand_atoms]
                if not removable:
                    break
                children = []
                for _, _, child in removable:
                    if child not in stats:
                        child_score, child_embedding = score(child)
                        stats[child] = [0, 0.0, max(child_score, 0.0)]
                        embeddings[child] = child_embedding
                        rewards[child] = child_score
                    children.append(child)
                parent_visits = max(1, stats[coalition][0])
                prior_values = [stats[child][2] for child in children]
                max_prior = max(prior_values)
                prior_exp = [math.exp(value - max_prior) for value in prior_values]
                prior_total = sum(prior_exp) or 1.0
                priors = [value / prior_total for value in prior_exp]
                selected = max(
                    zip(children, priors),
                    key=lambda item: (
                        (stats[item[0]][1] / stats[item[0]][0]
                         if stats[item[0]][0] else 0.0)
                        + self.c_puct * item[1] * math.sqrt(parent_visits)
                        / (1 + stats[item[0]][0])
                    ),
                )[0]
                stats[coalition][0] += 1
                stats[selected][0] += 1
                stats[selected][1] += rewards[selected]
                coalition = selected
                if rewards[coalition] > best_score:
                    best_state, best_score = coalition, rewards[coalition]
        return embeddings[best_state], best_score, best_state

    def _project_from_training_loader(self, train_loader, epoch):
        if train_loader is None:
            raise ValueError("ProtGNN projection requires the training loader")
        candidates = [[] for _ in range(self.num_classes)]
        device = next(self.parameters()).device
        with torch.no_grad():
            for source_batch in train_loader:
                batch = source_batch.to(device)
                if getattr(batch, "y", None) is None:
                    raise ValueError("training projection batches must contain labels")
                node_state, graph_embedding, graph_index = self._node_and_graph_embeddings(batch)
                labels = batch.y.view(-1).to(device=device, dtype=torch.long)
                if labels.numel() != graph_embedding.size(0):
                    raise ValueError("training projection requires one label per graph")
                if labels.numel() and (labels.min() < 0 or labels.max() >= self.num_classes):
                    raise ValueError("training projection label is outside the class range")
                for graph_id, label in enumerate(labels.tolist()):
                    if len(candidates[label]) >= self.nearest_graphs:
                        continue
                    candidate = self._projection_candidate(
                        batch, graph_id, node_state, graph_index)
                    if candidate is not None:
                        candidates[label].append(candidate)

        projected = 0
        with torch.no_grad():
            for class_id, class_candidates in enumerate(candidates):
                if not class_candidates:
                    continue
                start = class_id * self.prototypes_per_class
                stop = start + self.prototypes_per_class
                for prototype_id in range(start, stop):
                    prototype = self.prototype_vectors[prototype_id]
                    best_vector, best_similarity = None, float("-inf")
                    for candidate in class_candidates:
                        vector, similarity, _ = self._mcts_project(candidate, prototype)
                        if vector is not None and similarity > best_similarity:
                            best_vector, best_similarity = vector, similarity
                    if best_vector is not None:
                        self.prototype_vectors[prototype_id].copy_(best_vector)
                        projected += 1
        counts = {str(class_id): len(graphs)
                  for class_id, graphs in enumerate(candidates)}
        self.projection_summary = {
            "epoch": int(epoch),
            "projected_prototypes": int(projected),
            "unmatched_prototypes": int(self.num_prototypes - projected),
            "candidate_graphs_by_class": counts,
        }

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        if epoch < self.warm_epochs:
            self.prototype_classifier.weight.requires_grad_(False)
        if (epoch >= self.proj_epochs
                and (epoch - self.proj_epochs) % self.proj_interval == 0):
            was_training = self.training
            self.eval()
            try:
                self._project_from_training_loader(train_loader, epoch)
            finally:
                self.train(was_training)

    def run_config(self) -> dict:
        """Return only JSON-serializable effective architecture and schedule values."""
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        classifier_parameters = sum(parameter.numel()
                                    for parameter in self.prototype_classifier.parameters())
        return {
            "method": "protgnn",
            "adaptation_version": self.adaptation_version,
            "native_defaults": dict(_NATIVE_DEFAULTS),
            "effective_settings": {
                "warm_epochs": self.warm_epochs,
                "proj_epochs": self.proj_epochs,
                "proj_interval": self.proj_interval,
                "nearest_graphs": self.nearest_graphs,
                "prototypes_per_class": self.prototypes_per_class,
                "cluster_weight": self.cluster_weight,
                "separation_weight": self.separation_weight,
                "margin": self.margin,
                "rollout": self.rollout,
                "min_atoms": self.min_atoms,
                "max_atoms": self.max_atoms,
                "expand_atoms": self.expand_atoms,
                "c_puct": self.c_puct,
            },
            "architecture": {
                "node_dim": int(self.node_dim),
                "edge_dim": int(self.edge_dim),
                "num_tokens": int(self.num_tokens),
                "num_triples": int(self.num_triples),
                "num_relations": int(self.num_relations),
                "num_node_types": int(len(NODE_KINDS)),
                "num_classes": int(self.num_classes),
                "hidden": int(self.hidden),
                "layers": int(self.layers_count),
                "dropout": float(self.dropout_rate),
                "token_dim": int(self.token_dim),
                "num_prototypes": int(self.num_prototypes),
                "prototypes_per_class": int(self.prototypes_per_class),
                "prototype_classifier_bias": False,
                "parameter_count": int(total_parameters),
                "warmup_active_parameter_count": int(total_parameters - classifier_parameters),
                "joint_active_parameter_count": int(total_parameters),
            },
            "native_schedule": {
                "optimizer": "Adam",
                "max_epochs": int(self.max_epochs),
                "warm_epochs": int(self.warm_epochs),
                "proj_epochs": int(self.proj_epochs),
                "proj_interval": int(self.proj_interval),
                "nearest_graphs": int(self.nearest_graphs),
                "learning_rate": float(self.learning_rate),
                "batch_size": int(self.batch_size),
                "weight_decay": float(self.weight_decay),
                "patience": int(self.patience),
                "min_delta": float(self.min_delta),
            },
            "objective_coefficients": {
                "cluster": float(self.cluster_weight),
                "separation": float(self.separation_weight),
                "cross_class_l1": float(self.cross_class_l1_coefficient),
                "margin": float(self.margin),
            },
            "mechanism_settings": {
                "backbone_adaptation": "relation/payload-conditioned residual mean-message layers replace legacy GCN",
                "upstream_checkpoint_compatible": False,
                "node_inputs": ["x", "token", "node_type"],
                "edge_inputs": ["edge_index", "edge_attr", "edge_relation", "edge_triple"],
                "edge_payload_width": int(PAYLOAD_WIDTH),
                "graph_readout": "mean",
                "prototype_distance": "squared_l2",
                "prototype_activation": "log((distance + 1) / (distance + epsilon))",
                "epsilon": float(self.epsilon),
                "projection": {
                    "candidate_source": "training_loader_only",
                    "candidate_policy": "first class-matched training graphs, capped per class",
                    "latent_search": "connected-subgraph MCTS over relation/payload-aware node states",
                    "rollout": int(self.rollout),
                    "min_atoms": int(self.min_atoms),
                    "max_atoms": int(self.max_atoms),
                    "expand_atoms": int(self.expand_atoms),
                    "c_puct": float(self.c_puct),
                },
            },
        }
