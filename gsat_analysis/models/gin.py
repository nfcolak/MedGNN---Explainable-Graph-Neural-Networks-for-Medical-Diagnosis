"""GIN backbone for GSAT.

Self-contained (protgnn_analysis/models/GIN.py is fused with ProtGNN's prototype
layer and cannot be reused). This is the GSAT paper's default encoder — a plain
GIN — ported to modern PyG and extended so message passing can be weighted by a
per-edge attention tensor ``edge_atten`` in [0, 1] (this is how GSAT's extracted
subgraph G_S is realised: A_S = alpha ⊙ A, Miao et al. 2022 Sec. 4.2).

Sized to match ProtGNN's capacity (hidden 128 x 3 layers). ``mean`` readout to
match both other methods' graph-level pooling — the paper uses add-pool, but the
readout is held constant across methods here so the explanation mechanism is the
only moving part (see docs/PROJECT_CONTEXT.md).

Contract mirrors the upstream reference (external/GSAT/src/models/gin.py):
``forward(x, edge_index, batch, edge_atten=None) -> logits`` and
``get_emb(x, edge_index, batch, edge_atten=None) -> node embeddings``.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GINConv as _BaseGINConv
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.typing import OptTensor

_READOUTS = {"mean": global_mean_pool, "sum": global_add_pool}


class GINConv(_BaseGINConv):
    """GINConv whose messages can be scaled by a per-edge attention weight.

    ``edge_atten`` is a [num_edges, 1] tensor; when given, message m_uv becomes
    ``x_j * edge_atten`` before aggregation. Identical to
    external/GSAT/src/models/conv_layers.py::GINConv.
    """

    def forward(self, x: Tensor, edge_index, edge_atten: OptTensor = None, size=None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=size)
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        return x_j if edge_atten is None else x_j * edge_atten


class GIN(nn.Module):
    def __init__(self, x_dim, num_class, hidden_dim=128, num_layers=3,
                 dropout=0.3, readout="mean"):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.node_encoder = nn.Linear(x_dim, hidden_dim)
        self.convs = nn.ModuleList(
            GINConv(self._mlp(hidden_dim, hidden_dim), train_eps=True)
            for _ in range(num_layers)
        )
        self.relu = nn.ReLU()
        self.pool = _READOUTS[readout]
        self.fc_out = nn.Linear(hidden_dim, num_class)

    @staticmethod
    def _mlp(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def get_emb(self, x, edge_index, batch, edge_atten=None):
        x = self.node_encoder(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))

    def forward(self, x, edge_index, batch, edge_atten=None):
        emb = self.get_emb(x, edge_index, batch, edge_atten=edge_atten)
        return self.get_pred_from_emb(emb, batch)
