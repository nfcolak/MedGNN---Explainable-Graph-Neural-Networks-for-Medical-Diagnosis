"""Relational, temporal message passing over event_graph_v1 tensors.

Unlike categorical patient-ID embeddings, all embeddings here refer only to the
training-fitted clinical token/unit and relation vocabularies or fixed node kinds.
Outputs are logits, consistent with neighboring classification integrations.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.nn.aggr import DegreeScalerAggregation

from .tensorize import EDGE_FEATURES, KIND_TO_ID, NODE_FEATURES, NODE_KINDS


class TemporalRelationalLayer(MessagePassing):
    """Sum messages conditioned jointly on sender, receiver, relation and time.

    A residual update and per-node LayerNorm avoid graph-to-graph normalization
    coupling. No implicit self loops are added; the residual carries self state.
    ``edge_index[0]`` denotes senders, matching the tensor adapter and PyG.
    """

    def __init__(self, hidden_dim: int, relation_dim: int, dropout: float):
        super().__init__(aggr="add", node_dim=0)
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + relation_dim + len(EDGE_FEATURES), hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, relation: Tensor,
                edge_attr: Tensor) -> Tensor:
        aggregate = self.propagate(
            edge_index, x=x, relation=relation, edge_attr=edge_attr,
            size=(x.size(0), x.size(0)),
        )
        update = self.update_mlp(torch.cat((x, aggregate), dim=-1))
        return self.norm(x + self.dropout(update))

    def message(self, x_j: Tensor, x_i: Tensor, relation: Tensor,
                edge_attr: Tensor) -> Tensor:
        return self.message_mlp(torch.cat((x_j, x_i, relation, edge_attr), dim=-1))


class EventGraphGNN(nn.Module):
    """Classify one PyG Data or a PyG Batch, returning logits [B, num_classes].

    Construct after fitting the adapter::

        model = EventGraphGNN(
            num_tokens=adapter.num_tokens,
            num_relations=adapter.num_relations,
            num_classes=number_of_training_classes,
        )

    Embedding sizes include unknown ID 0. Unknown rows are zero/frozen padding
    embeddings, so unseen tokens/relations do not receive untrained random vectors.
    Numeric/kind inputs still describe unknown nodes. Three default propagation
    layers allow knowledge -> concept -> event -> visit messages; the event mean
    readout also exposes knowledge after two hops. Actual receptive field depends
    on builder topology and layer count, not the semantic layer count alone.

    Readout concatenates the sum of patient states (exactly one per graph) with
    the event-state mean. An event-empty graph contributes an exact zero event
    vector. Graph size, sample/node identifiers, labels and provenance are not
    direct inputs. ``forward`` never reads ``y``. No softmax is applied; use
    cross-entropy for multiclass or num_classes=1 with BCEWithLogitsLoss and
    caller-converted floating labels. Move model and Data to the same device.

    Save constructor configuration alongside ``state_dict`` and the adapter's
    serialized state. Dimensions alone cannot detect a mismatched vocabulary.
    """

    def __init__(self, num_tokens: int, num_relations: int, num_classes: int,
                 *, hidden_dim: int = 128, relation_dim: int = 32,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        dimensions = {"num_tokens": num_tokens, "num_relations": num_relations,
                      "num_classes": num_classes, "hidden_dim": hidden_dim,
                      "relation_dim": relation_dim, "num_layers": num_layers}
        if any(type(value) is not int or value < 1 for value in dimensions.values()):
            raise ValueError("Embedding sizes, classes, widths and layer count must be positive integers")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.config = {**dimensions, "dropout": float(dropout)}
        self.token_embedding = nn.Embedding(num_tokens, hidden_dim, padding_idx=0)
        self.kind_embedding = nn.Embedding(len(NODE_KINDS), hidden_dim)
        self.numeric_projection = nn.Linear(len(NODE_FEATURES), hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, relation_dim, padding_idx=0)
        self.layers = nn.ModuleList([
            TemporalRelationalLayer(hidden_dim, relation_dim, float(dropout))
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data: Data) -> Tensor:
        """Consume only adapter covariates; reject invalid patient/batch structure.

        Standard PyG batching offsets edge_index but concatenates categorical
        IDs unchanged. Empty edge sets are supported. Inputs should be produced
        by EventGraphTensorizer, which validates graph-local endpoints and finite
        raw values; this method additionally rejects cross-graph edges.
        """
        if data.x is None or data.x.ndim != 2 or data.x.size(1) != len(NODE_FEATURES):
            raise ValueError("Expected x of shape [N, 6]")
        n = data.x.size(0)
        if n == 0:
            raise ValueError("Empty node tensors are not supported")
        for name in ("token_id", "kind_id", "patient_mask"):
            value = getattr(data, name, None)
            if value is None or value.ndim != 1 or value.size(0) != n:
                raise ValueError(f"Expected {name} of shape [N]")
        if data.token_id.dtype != torch.long or data.kind_id.dtype != torch.long:
            raise ValueError("token_id and kind_id must be int64")
        if data.patient_mask.dtype != torch.bool:
            raise ValueError("patient_mask must be boolean")
        if not torch.equal(data.patient_mask, data.kind_id == KIND_TO_ID["patient"]):
            raise ValueError("patient_mask must match patient node kinds")
        if data.edge_index is None or data.edge_index.ndim != 2 or data.edge_index.size(0) != 2:
            raise ValueError("Expected edge_index of shape [2, E]")
        if data.edge_index.dtype != torch.long:
            raise ValueError("edge_index must be int64")
        e = data.edge_index.size(1)
        if data.edge_type.ndim != 1 or data.edge_type.size(0) != e or data.edge_type.dtype != torch.long:
            raise ValueError("Expected int64 edge_type of shape [E]")
        if data.edge_attr.ndim != 2 or tuple(data.edge_attr.shape) != (e, len(EDGE_FEATURES)):
            raise ValueError("Expected edge_attr of shape [E, 2]")
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(n, dtype=torch.long, device=data.x.device)
            num_graphs = 1
        else:
            if batch.ndim != 1 or batch.size(0) != n or batch.dtype != torch.long:
                raise ValueError("Expected int64 batch vector of shape [N]")
            if bool((batch < 0).any()):
                raise ValueError("Negative batch index")
            num_graphs = int(batch.max().item()) + 1
            declared = getattr(data, "num_graphs", num_graphs)
            if declared != num_graphs:
                raise ValueError("Batch includes an empty or invalid graph")
        patient_counts = global_add_pool(
            data.patient_mask.to(data.x.dtype).unsqueeze(-1), batch, size=num_graphs,
        )
        if not bool((patient_counts == 1).all()):
            raise ValueError("Each graph must contain exactly one patient")
        if e:
            if bool((data.edge_index < 0).any()) or bool((data.edge_index >= n).any()):
                raise ValueError("Out-of-range edge endpoint")
            if not torch.equal(batch[data.edge_index[0]], batch[data.edge_index[1]]):
                raise ValueError("Cross-graph edges are not permitted")
        x = self.input_norm(
            self.token_embedding(data.token_id) + self.kind_embedding(data.kind_id)
            + self.numeric_projection(data.x),
        )
        relation = self.relation_embedding(data.edge_type)
        for layer in self.layers:
            x = layer(x, data.edge_index, relation, data.edge_attr)
        patient_state = global_add_pool(
            x * data.patient_mask.unsqueeze(-1), batch, size=num_graphs,
        )
        event_mask = (data.kind_id == KIND_TO_ID["event"]).to(x.dtype).unsqueeze(-1)
        event_sum = global_add_pool(x * event_mask, batch, size=num_graphs)
        event_count = global_add_pool(event_mask, batch, size=num_graphs)
        event_mean = event_sum / event_count.clamp_min(1)
        return self.classifier(torch.cat((patient_state, event_mean), dim=-1))


class EventGatedRelationalLayer(MessagePassing):
    """GCHM-style receiver-conditioned modulation for typed event graphs.

    The native GCHM assumes a single patient hub and concept spokes. Event graphs
    retain that patient hub but add visit, event and knowledge nodes, so the same
    measured mechanism is applied to every typed edge while preserving relation and
    availability-time inputs. The receiver state gates the sender/context message;
    no patient or stay identifier is embedded.
    """

    def __init__(self, hidden_dim: int, relation_dim: int, degree_histogram: Tensor,
                 dropout: float):
        if degree_histogram.ndim != 1 or degree_histogram.numel() < 1:
            raise ValueError("degree_histogram must be a nonempty vector")
        super().__init__(
            aggr=DegreeScalerAggregation(
                aggr=["mean", "min", "max", "std"],
                scaler=["identity", "amplification", "attenuation"],
                deg=degree_histogram.detach().clone(),
                train_norm=False,
            ),
            node_dim=0,
        )
        self.aggr_module.init_avg_deg_log = max(self.aggr_module.init_avg_deg_log, 1e-6)
        self.aggr_module.avg_deg_log.clamp_(min=1e-6)
        message_input = 2 * hidden_dim + relation_dim + len(EDGE_FEATURES)
        self.message_mlp = nn.Sequential(
            nn.Linear(message_input, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.receiver_gate = nn.Linear(hidden_dim, hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(13 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, relation: Tensor,
                edge_attr: Tensor) -> Tensor:
        aggregate = self.propagate(
            edge_index, x=x, relation=relation, edge_attr=edge_attr,
            size=(x.size(0), x.size(0)),
        )
        update = self.update_mlp(torch.cat((x, aggregate), dim=-1))
        return self.norm(x + self.dropout(update))

    def message(self, x_j: Tensor, x_i: Tensor, relation: Tensor,
                edge_attr: Tensor) -> Tensor:
        content = self.message_mlp(torch.cat((x_j, x_i, relation, edge_attr), dim=-1))
        # This is the event-graph analogue of GCHM's FiLM gate: destination
        # state modulates the incoming clinical event/context content.
        return content * torch.sigmoid(self.receiver_gate(x_i))


class EventGCHM(nn.Module):
    """GCHM adaptation for ``event_graph_v1`` with a patient-state readout.

    It is intentionally a separate class from the legacy native GCHM. The legacy
    model requires one hub per graph, x width 331, native node IDs and a 132-column
    hub payload; event_graph_v1 instead supplies typed nodes, train-fitted token and
    relation IDs, six numeric node covariates and temporal edge attributes.
    """

    def __init__(self, num_tokens: int, num_relations: int, num_classes: int,
                 *, degree_histogram: Tensor, hidden_dim: int = 128,
                 relation_dim: int = 32, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        dimensions = {
            "num_tokens": num_tokens, "num_relations": num_relations,
            "num_classes": num_classes, "hidden_dim": hidden_dim,
            "relation_dim": relation_dim, "num_layers": num_layers,
        }
        if any(type(value) is not int or value < 1 for value in dimensions.values()):
            raise ValueError("Embedding sizes, classes, widths and layer count must be positive integers")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if (degree_histogram.ndim != 1 or degree_histogram.numel() < 1
                or degree_histogram.dtype not in (torch.float16, torch.float32, torch.float64,
                                                  torch.int32, torch.int64)
                or bool((degree_histogram < 0).any())
                or not bool(torch.isfinite(degree_histogram.float()).all())):
            raise ValueError("degree_histogram must be a finite nonnegative vector")
        degree_histogram = degree_histogram.float()
        self.register_buffer("degree_histogram", degree_histogram.detach().clone())
        self.config = {**dimensions, "degree_histogram": degree_histogram.tolist(),
                       "dropout": float(dropout), "mechanism": "receiver_conditioned_multiplication_pna"}
        self.token_embedding = nn.Embedding(num_tokens, hidden_dim, padding_idx=0)
        self.kind_embedding = nn.Embedding(len(NODE_KINDS), hidden_dim)
        self.numeric_projection = nn.Linear(len(NODE_FEATURES), hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.relation_embedding = nn.Embedding(num_relations, relation_dim, padding_idx=0)
        self.layers = nn.ModuleList([
            EventGatedRelationalLayer(hidden_dim, relation_dim, self.degree_histogram,
                                      float(dropout))
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(float(dropout)), nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data: Data) -> Tensor:
        if data.x is None or data.x.ndim != 2 or data.x.size(1) != len(NODE_FEATURES):
            raise ValueError("Expected x of shape [N, 6]")
        n = data.x.size(0)
        if n == 0:
            raise ValueError("Empty node tensors are not supported")
        for name in ("token_id", "kind_id", "patient_mask"):
            value = getattr(data, name, None)
            if value is None or value.ndim != 1 or value.size(0) != n:
                raise ValueError(f"Expected {name} of shape [N]")
        if data.token_id.dtype != torch.long or data.kind_id.dtype != torch.long:
            raise ValueError("token_id and kind_id must be int64")
        if data.patient_mask.dtype != torch.bool:
            raise ValueError("patient_mask must be boolean")
        if not torch.equal(data.patient_mask, data.kind_id == KIND_TO_ID["patient"]):
            raise ValueError("patient_mask must match patient node kinds")
        if data.edge_index is None or data.edge_index.ndim != 2 or data.edge_index.size(0) != 2:
            raise ValueError("Expected edge_index of shape [2, E]")
        if data.edge_index.dtype != torch.long:
            raise ValueError("edge_index must be int64")
        e = data.edge_index.size(1)
        if data.edge_type.ndim != 1 or data.edge_type.size(0) != e or data.edge_type.dtype != torch.long:
            raise ValueError("Expected int64 edge_type of shape [E]")
        if data.edge_attr.ndim != 2 or tuple(data.edge_attr.shape) != (e, len(EDGE_FEATURES)):
            raise ValueError("Expected edge_attr of shape [E, 2]")
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(n, dtype=torch.long, device=data.x.device)
            num_graphs = 1
        else:
            if batch.ndim != 1 or batch.size(0) != n or batch.dtype != torch.long:
                raise ValueError("Expected int64 batch vector of shape [N]")
            if bool((batch < 0).any()):
                raise ValueError("Negative batch index")
            num_graphs = int(batch.max().item()) + 1
            if getattr(data, "num_graphs", num_graphs) != num_graphs:
                raise ValueError("Batch includes an empty or invalid graph")
        patient_counts = global_add_pool(
            data.patient_mask.to(data.x.dtype).unsqueeze(-1), batch, size=num_graphs,
        )
        if not bool((patient_counts == 1).all()):
            raise ValueError("Each graph must contain exactly one patient")
        if e:
            if bool((data.edge_index < 0).any()) or bool((data.edge_index >= n).any()):
                raise ValueError("Out-of-range edge endpoint")
            if not torch.equal(batch[data.edge_index[0]], batch[data.edge_index[1]]):
                raise ValueError("Cross-graph edges are not permitted")
        x = self.input_norm(
            self.token_embedding(data.token_id) + self.kind_embedding(data.kind_id)
            + self.numeric_projection(data.x),
        )
        relation = self.relation_embedding(data.edge_type)
        for layer in self.layers:
            x = layer(x, data.edge_index, relation, data.edge_attr)
        patient_state = global_add_pool(
            x * data.patient_mask.unsqueeze(-1), batch, size=num_graphs,
        )
        event_mask = (data.kind_id == KIND_TO_ID["event"]).to(x.dtype).unsqueeze(-1)
        event_sum = global_add_pool(x * event_mask, batch, size=num_graphs)
        event_count = global_add_pool(event_mask, batch, size=num_graphs)
        event_mean = event_sum / event_count.clamp_min(1)
        return self.classifier(torch.cat((patient_state, event_mean), dim=-1))
