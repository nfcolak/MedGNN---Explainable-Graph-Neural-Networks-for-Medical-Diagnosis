import torch
import torch.nn as nn
from torch_geometric.data import Data


class ProtGNNWrapper(nn.Module):
    """
    GraphXAI-compatible wrapper for ProtGNN (GnnNets).
    Returns only logits from the full forward pass (including prototype layer).
    Used for inference and for explainers when prototypes are DISABLED.
    """

    def __init__(self, gnn_nets: nn.Module):
        super().__init__()
        self._gnn_nets = gnn_nets

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        logits, _probs, _node_emb, _graph_emb, _min_dist = self._gnn_nets(data)
        return logits


class BackboneWrapper(nn.Module):
    """
    GraphXAI-compatible wrapper that bypasses the prototype distance layer.

    When prototype layers are enabled, the forward path goes:
        GCN layers → readout → prototype_distances() → last_layer → logits
    The prototype_distances() step uses a log formula that crushes gradients,
    making GradExplainer useless.

    This wrapper replaces that path with the MLP branch:
        GCN layers → readout → mlps → logits
    which preserves clean gradient flow for the explainers.

    Only used during the explanation pass when enable_prot=True.
    The model predictions may differ slightly from the prototype path,
    but the gradient signal correctly reflects which nodes matter.
    """

    def __init__(self, gnn_nets: nn.Module):
        super().__init__()
        self._gnn_nets = gnn_nets

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        model = self._gnn_nets.model  # the GCNNet / GINNet / GATNet instance
        device = x.device

        # ── Step 1: GCN message-passing layers ──────────────────────────
        edge_weight = None  # BackboneWrapper always works without edge weights
        for i in range(model.num_gnn_layers):
            if edge_weight is not None:
                x = model.gnn_layers[i](x, edge_index, edge_weight)
            elif not model.gnn_layers[i].normalize:
                unit_w = torch.ones(edge_index.shape[1], device=device)
                x = model.gnn_layers[i](x, edge_index, unit_w)
            else:
                x = model.gnn_layers[i](x, edge_index)
            if model.emb_normlize:
                import torch.nn.functional as F
                x = F.normalize(x, p=2, dim=-1)
            x = model.gnn_non_linear(x)

        # ── Step 2: Graph-level readout pooling ──────────────────────────
        pooled = [readout(x, batch) for readout in model.readout_layers]
        x = torch.cat(pooled, dim=-1)

        # ── Step 3: MLP classifier (no prototype distances) ──────────────
        for i in range(model.num_mlp_layers - 1):
            x = model.mlps[i](x)
            x = model.mlp_non_linear(x)
            x = model.dropout(x)
        logits = model.mlps[-1](x)
        return logits
