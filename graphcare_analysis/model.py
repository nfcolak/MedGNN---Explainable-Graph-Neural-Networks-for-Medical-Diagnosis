"""First-party GraphCare integration with a clinical-empty EHR policy.

Inherit upstream parameter construction (checkpoint keys remain unchanged), but
own the forward integration so training and functional-call explanations share
one pooling rule. The forward below follows external/GraphCare/graphcare_/model.py;
only the direct-EHR denominator differs. Regression tests compare nonempty logits
and every gradient exactly against upstream, with matched dropout RNG.

A canonical record with no fitted concepts retains its patient hub and zero
clinical membership. Its raw EHR mean is exactly zero (0 / max(count, 1)); the
subsequent learned affine projection still has its normal bias. No clinical
concept or embedding is fabricated. Hub GNN computation remains unchanged.
"""
import random

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from graphcare_.model import GraphCare as UpstreamGraphCare


class GraphCare(UpstreamGraphCare):
    """Checkpoint-compatible BAT-GNN used by MedGNN training and explanations."""

    def forward(self, node_ids, rel_ids, edge_index, batch, visit_node, ehr_nodes, store_attn=False, in_drop=False):

        if in_drop and self.drop_rate > 0:
            edge_count = edge_index.size(1)
            edges_to_remove = int(edge_count * self.drop_rate)
            indices_to_remove = set(random.sample(range(edge_count), edges_to_remove))
            edge_index = edge_index[:, [i for i in range(edge_count) if i not in indices_to_remove]].to(edge_index.device)
            rel_ids = torch.tensor([rel_id for i, rel_id in enumerate(rel_ids) if i not in indices_to_remove], device=rel_ids.device)

        x = self.node_emb(node_ids).float()
        edge_attr = self.rel_emb(rel_ids).float()

        # we found that batch normalization is not helpful
        # x = self.bn1(self.lin(x))
        # edge_attr = self.bn1(self.lin(edge_attr))

        x = self.lin(x)
        edge_attr = self.lin(edge_attr)


        if store_attn:
            self.alpha_weights = []
            self.beta_weights = []
            self.attention_weights = []
            self.edge_weights = []

        for layer in range(1, self.layers+1):
            if self.use_alpha:
                # alpha = masked_softmax((self.leakyrelu(self.alpha_attn[str(layer)](visit_node.float()))), mask=visit_node>1, dim=1)
                alpha = torch.softmax((self.alpha_attn[str(layer)](visit_node.float())), dim=1)  # (batch, max_visit, num_nodes)

            if self.use_beta:
                # beta = masked_softmax((self.leakyrelu(self.beta_attn[str(layer)](visit_node.float()))), mask=visit_node>1, dim=0) * self.lambda_j
                beta = torch.tanh((self.beta_attn[str(layer)](visit_node.float()))) * self.lambda_j

            if self.use_alpha and self.use_beta:
                attn = alpha * beta
            elif self.use_alpha:
                attn = alpha * torch.ones((batch.max().item() + 1, self.max_visit, 1)).to(edge_index.device)
            elif self.use_beta:
                attn = beta * torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes)).to(edge_index.device)
            else:
                attn = torch.ones((batch.max().item() + 1, self.max_visit, self.num_nodes)).to(edge_index.device)

            attn = torch.sum(attn, dim=1)

            xj_node_ids = node_ids[edge_index[0]]
            xj_batch = batch[edge_index[0]]
            attn = attn[xj_batch, xj_node_ids].reshape(-1, 1)

            if self.gnn == "BAT":
                x, w_rel = self.conv[str(layer)](x, edge_index, edge_attr, attn=attn)

            else:
                x = self.conv[str(layer)](x, edge_index)

            # x = self.bn_gnn[str(layer)](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            if store_attn:
                self.alpha_weights.append(alpha)
                self.beta_weights.append(beta)
                self.attention_weights.append(attn)
                self.edge_weights.append(w_rel)

        if self.patient_mode == "joint" or self.patient_mode == "graph":
            # patient graph embedding through global mean pooling
            x_graph = global_mean_pool(x, batch)
            x_graph = F.dropout(x_graph, p=self.dropout, training=self.training)


        if self.patient_mode == "joint" or self.patient_mode == "node":
            # patient node embedding through local (direct EHR) mean pooling
            x_node = torch.stack([ehr_nodes[i].view(1, -1) @ self.node_emb.weight / torch.sum(ehr_nodes[i]).clamp_min(1) for i in range(batch.max().item() + 1)])
            x_node = self.lin(x_node).squeeze(1)
            x_node = F.dropout(x_node, p=self.dropout, training=self.training)

        if self.patient_mode == "joint":
            # concatenate patient graph embedding and patient node embedding
            x_concat = torch.cat((x_graph, x_node), dim=1)
            x_concat = F.dropout(x_concat, p=self.dropout, training=self.training)
            # MLP for prediction
            logits = self.MLP(x_concat)

        elif self.patient_mode == "graph":
            # MLP for prediction
            logits = self.MLP(x_graph)

        elif self.patient_mode == "node":
            # MLP for prediction
            logits = self.MLP(x_node)

        if store_attn:
            return logits, self.alpha_weights, self.beta_weights,  self.attention_weights, self.edge_weights
        else:
            return logits
