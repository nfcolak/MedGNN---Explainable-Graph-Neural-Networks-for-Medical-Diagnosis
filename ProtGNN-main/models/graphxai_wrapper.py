import torch
import torch.nn as nn
from torch_geometric.data import Data


class ProtGNNWrapper(nn.Module):
    """
    GraphXAI-compatible wrapper for ProtGNN (GnnNets).

    GraphXAI explainers call model(x, edge_index, **forward_kwargs) and expect a
    single logit tensor.  ProtGNN's GnnNets.forward(data) takes a Data/Batch
    object and returns (logits, probs, node_emb, graph_emb, min_distances).

    This wrapper bridges the gap:
      - Accepts (x, edge_index, batch=None) — GraphXAI's calling convention.
      - Builds a Data object internally and delegates to the wrapped GnnNets.
      - Returns only logits so GraphXAI loss/gradient computations work normally.

    All sub-modules (including the MessagePassing layers that GraphXAI counts
    to determine self.L) are reachable via self.modules() because _gnn_nets is
    registered as a proper nn.Module child.

    Usage::

        from models import GnnNets
        from models.graphxai_wrapper import ProtGNNWrapper
        from graphxai.explainers import GradExplainer

        gnn_nets = GnnNets(input_dim, output_dim, model_args)
        # ... load checkpoint into gnn_nets ...

        wrapper = ProtGNNWrapper(gnn_nets)
        explainer = GradExplainer(wrapper, criterion=torch.nn.CrossEntropyLoss())
    """

    def __init__(self, gnn_nets: nn.Module):
        super().__init__()
        self._gnn_nets = gnn_nets

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor, [N x F]): Node feature matrix.
            edge_index (torch.Tensor, [2 x E]): Edge indices.
            batch (torch.Tensor, [N], optional): Batch vector assigning each
                node to a graph.  If None, all nodes are assumed to belong to
                a single graph.

        Returns:
            logits (torch.Tensor, [B x C]): Unnormalised class scores.
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        data = Data(x=x, edge_index=edge_index, batch=batch)
        logits, _probs, _node_emb, _graph_emb, _min_dist = self._gnn_nets(data)
        return logits
