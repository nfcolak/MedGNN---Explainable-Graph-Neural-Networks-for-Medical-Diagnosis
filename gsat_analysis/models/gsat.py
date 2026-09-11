"""Graph Stochastic Attention (Miao, Liu & Li, ICML 2022).

Reimplemented from the paper + the upstream reference
(external/GSAT/example/gsat.py) against this project's per-patient PyG graphs.
See docs/PROJECT_CONTEXT.md for why it is reimplemented rather than vendored.

Mechanism (paper Sec. 4.2)
-------------------------
1. A shared GIN encoder g_phi embeds the input graph G -> node embeddings h.
2. An extractor MLP maps h to a logit per node; a stochastic attention
   alpha_v ~ Bern(p_v) is sampled via the Gumbel-sigmoid (binary concrete)
   reparameterisation so gradients flow to p_v. Edge attention is lifted as
   alpha_uv = alpha_u * alpha_v (node-level attention, paper App. C.3).
3. The SAME GIN encoder (unified model) re-embeds G with messages scaled by
   alpha_uv -> mean-pool -> linear -> logits. This is the predictor f_theta.
4. Loss = CE(logits, y)  +  beta * KL( Bern(p) || Bern(r) ), the graph
   information bottleneck (Eq. 8 / Eq. 9). r follows a curriculum: start at
   ``init_r`` and step down by ``decay_r`` every ``decay_interval`` epochs to
   ``final_r``.

Interpretation: the per-node probability ``p_v`` (deterministic sigmoid, no
noise, at eval) IS the explanation — faithful by construction, same status as
GraphCare's attention. Higher p_v = node kept in the label-relevant subgraph.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtractorMLP(nn.Module):
    """h_v -> attention logit. Node-level: [hidden -> 2*hidden -> hidden -> 1].
    Edge-level: input is concat(h_u, h_v) so first dim is 2*hidden.
    (Matches external/GSAT/src/utils/get_model.py::ExtractorMLP.)"""

    def __init__(self, hidden_dim, attention_level="node", dropout=0.5):
        super().__init__()
        self.attention_level = attention_level
        in_dim = hidden_dim * (2 if attention_level == "edge" else 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, emb, edge_index):
        if self.attention_level == "edge":
            src, dst = edge_index
            return self.mlp(torch.cat([emb[src], emb[dst]], dim=-1))
        return self.mlp(emb)


class GSAT(nn.Module):
    def __init__(self, clf, extractor, *, attention_level="node", temperature=1.0,
                 info_loss_coef=1.0, init_r=0.9, final_r=0.7, decay_interval=10,
                 decay_r=0.1):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.attention_level = attention_level
        self.temperature = temperature
        self.info_loss_coef = info_loss_coef
        self.init_r = init_r
        self.final_r = final_r
        self.decay_interval = decay_interval
        self.decay_r = decay_r

    # -- r curriculum (paper App. C.2.2) -----------------------------------
    def get_r(self, epoch):
        r = self.init_r - (epoch // self.decay_interval) * self.decay_r
        return max(r, self.final_r)

    # -- stochastic attention -------------------------------------------------
    def _sample(self, att_log_logits, training):
        """Binary concrete / Gumbel-sigmoid. Deterministic sigmoid at eval."""
        if training:
            noise = torch.empty_like(att_log_logits).uniform_(1e-10, 1 - 1e-10)
            noise = torch.log(noise) - torch.log(1.0 - noise)
            return ((att_log_logits + noise) / self.temperature).sigmoid()
        return att_log_logits.sigmoid()

    @staticmethod
    def _lift(node_att, edge_index):
        src, dst = edge_index
        return node_att[src] * node_att[dst]

    def _info_loss(self, att, r):
        """KL( Bern(att) || Bern(r) ), mean over nodes (or edges). Eq. (9)."""
        return (att * torch.log(att / r + 1e-6)
                + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()

    # -- forward ------------------------------------------------------------
    def forward(self, data, epoch=0, training=True):
        """Returns a dict: logits, node_att (p_v, [N] or None), edge_att ([E]),
        loss, pred_loss, info_loss."""
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch)
        att_log_logits = self.extractor(emb, data.edge_index)      # [N,1] or [E,1]
        att = self._sample(att_log_logits, training)               # same shape

        if self.attention_level == "edge":
            edge_att = att.squeeze(-1)
            node_att = None
        else:
            node_att = att.squeeze(-1)                             # [N]
            edge_att = self._lift(node_att, data.edge_index)       # [E]

        logits = self.clf(data.x, data.edge_index, batch=data.batch,
                          edge_atten=edge_att.unsqueeze(-1))

        r = self.get_r(epoch)
        pred_loss = F.cross_entropy(logits, data.y.view(-1).long())
        info_loss = self._info_loss(att.squeeze(-1), r) * self.info_loss_coef
        return {
            "logits": logits,
            "node_att": node_att,
            "edge_att": edge_att,
            "loss": pred_loss + info_loss,
            "pred_loss": pred_loss,
            "info_loss": info_loss,
            "r": r,
        }

    def predict(self, x, edge_index, batch):
        """Deterministic joint forward (no Gumbel noise): the function a post-hoc
        explainer differentiates / perturbs. Returns logits [B, C]. Grad-safe
        w.r.t. x. Mirrors forward() with training=False and no loss."""
        emb = self.clf.get_emb(x, edge_index, batch=batch)
        p = self.extractor(emb, edge_index).sigmoid()             # [N,1] or [E,1]
        if self.attention_level == "edge":
            edge_att = p.squeeze(-1)
        else:
            edge_att = self._lift(p.squeeze(-1), edge_index)
        return self.clf(x, edge_index, batch=batch, edge_atten=edge_att.unsqueeze(-1))

    @torch.no_grad()
    def node_importance(self, data):
        """Deterministic per-node explanation p_v (no Gumbel noise). [N] tensor.
        For edge-level attention, returns per-node max of incident edge p."""
        self.eval()
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch)
        logits = self.extractor(emb, data.edge_index).squeeze(-1)
        p = logits.sigmoid()
        if self.attention_level == "node":
            return p
        # edge-level: scatter-max incident edge probs back to nodes
        n = data.x.size(0)
        out = torch.zeros(n, device=p.device)
        src, dst = data.edge_index
        for e in range(p.size(0)):
            out[src[e]] = torch.maximum(out[src[e]], p[e])
            out[dst[e]] = torch.maximum(out[dst[e]], p[e])
        return out
