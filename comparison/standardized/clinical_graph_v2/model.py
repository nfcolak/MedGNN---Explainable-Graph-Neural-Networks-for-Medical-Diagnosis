"""A small edge-conditioned GNN for the clinical decision-point graph.

Deliberately small (~100k parameters at default width) because the point of this run
is to compare INPUTS and EDGE SETS, not to win a benchmark. Every arm shares this
class and differs by constructor flags. Payload ablations keep parameter count
identical; architecture choices can change it, and NOMP bypasses the conv layers.
Report both total and active counts rather than equating total count with capacity.

Message function, the part that matters:

    m(i<-j) = MLP([x_j, x_i, e_ji])

It takes the sender, the RECEIVER, and the edge payload. This is not decoration: an
additively separable message m(x_j) + f(e) can never express "this creatinine is high
*relative to this patient's own baseline*", which is exactly what `baseline_of`
carries. `test_message_not_separable` in the mechanism check verifies the residual

    m(x_i, x_j, e) - [m(x_i, 0, 0) + m(0, x_j, 0) + m(0, 0, e) - 2*m(0, 0, 0)]

is far above float32 roundoff, i.e. the layer really does mix its inputs.

Readout is sum + mean concatenated: sum keeps multiplicity (how many abnormal results
support a class), mean keeps intensity, and graphs here vary hugely in size.

`HeteroGraphTransformerLayer` below is the HGT (WWW'20) arm. The baseline's
relation one-hot shifts the first preactivation, but the following ReLU and Linear
can make the full-message effect depend on node content. The mechanism diagnostic
constructs such a witness. HGT adds explicit per-meta-relation matrices and
normalised attention; the baseline is not claimed equivalent to HGT_noHeter or
architecturally incapable of relation-content interactions.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import DegreeScalerAggregation
from torch_geometric.utils import scatter, softmax

from .tensorize import PAYLOAD_WIDTH


class EdgeConditionedLayer(MessagePassing):
    """One round of edge-conditioned message passing with a residual update."""

    def __init__(self, dim, edge_dim, dropout=0.1, use_edge_payload=True):
        super().__init__(aggr='add', node_dim=0)
        self.use_edge_payload = use_edge_payload
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        if self.use_edge_payload:
            payload = edge_attr
        else:
            # Remove numeric values AND their presence flags, not relation identity.
            # Clone so another arm can still consume the original graph unchanged.
            payload = edge_attr.clone()
            payload[:, -PAYLOAD_WIDTH:] = 0
        out = self.propagate(edge_index, x=x, edge_attr=payload)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x + self.dropout(out))

    def message(self, x_j, x_i, edge_attr):
        return self.message_mlp(torch.cat([x_j, x_i, edge_attr], dim=-1))


class HeteroGraphTransformerLayer(MessagePassing):
    """HGT (Hu et al., WWW'20) over a flat typed graph.

    One round of typed attention. Every edge is decomposed into its meta-relation
    <source kind, relation, target kind>, and the attention weight is split across
    those components exactly as the paper prescribes:

        att(i<-j) = softmax_i( (K(x_j) W^ATT_phi(e)) Q(x_i)^T * mu_phi(e) / sqrt(d) )
        msg(i<-j) = V(x_j) W^MSG_phi(e)
        h_i       = a_i * out(sum_j att * msg) + (1 - a_i) * x_i

    Three things here are NOT in `EdgeConditionedLayer`, which is why this arm can
    be different rather than just wider:

    1. `W^ATT_phi` / `W^MSG_phi` are explicit per-meta-relation *matrices*, unlike
       the shared nonlinear MLP's implicit relation-content interactions.
    2. `mu_phi` is a learned per-meta-relation scalar (the paper's relation prior):
       it lets the model down-weight a whole relation without touching its content.
    3. Attention is softmax-normalised per receiver, so a node with 400 `instance_of`
       neighbours and 2 `baseline_of` neighbours no longer drowns the latter in a sum.

    **BG-HGNN's warning (2024) is the reason for the shapes chosen here.** The naive
    "one MLP per relation" (RGCN-style) multiplies parameters by the relation count:
    measured on this graph, 15 relations take a 27k-parameter layer to 415k. So the
    per-relation weights act on the *per-head* dimension (`dim // heads`), not on the
    full hidden width, and only K and V are typed -- Q, the output projection and the
    update path stay shared. Cost is `2 * T * H * (d/H)^2`, which is linear in the
    number of meta-relations with a small constant, instead of `T * d^2`.

    `relation_collapse` is BG-HGNN's second failure mode, so it is *measurable* here
    rather than assumed absent: `relation_separation()` reports how far the typed
    projections stay apart, and the mechanism check fails if they collapse to one.

    The edge payload (delta / interval / recency) is not dropped: it is projected to
    a per-head additive bias on the attention logit, which is where a continuous
    quantity belongs in an attention model. Set `use_edge_payload=False` to blank it
    with the parameter count unchanged, matching the other arms' ablation contract.
    """

    def __init__(self, dim, num_triples, payload_dim, heads=4, dropout=0.1,
                 use_edge_payload=True):
        super().__init__(aggr='add', node_dim=0)
        if dim % heads:
            raise ValueError(f'hidden dim {dim} must be divisible by heads {heads}')
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.num_triples = num_triples
        self.use_edge_payload = use_edge_payload

        self.k_lin = nn.Linear(dim, dim)
        self.q_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        self.out_lin = nn.Linear(dim, dim)

        # Per-meta-relation projections, per head, on the HEAD dimension. This is the
        # BG-HGNN-safe shape: 15 relations cost 15*4*24*24*2 = 69k, not 15 * a full
        # hidden-width MLP.
        self.k_rel = nn.Parameter(torch.empty(num_triples, heads, self.head_dim,
                                              self.head_dim))
        self.v_rel = nn.Parameter(torch.empty(num_triples, heads, self.head_dim,
                                              self.head_dim))
        # The paper's mu: a learned prior scaling each meta-relation's attention.
        self.rel_prior = nn.Parameter(torch.ones(num_triples, heads))
        # Continuous edge payload -> additive attention bias, one scalar per head.
        self.payload_lin = nn.Linear(payload_dim, heads)
        # Per-node-type skip gate (paper's alpha); sigmoid keeps it in (0, 1).
        self.skip = nn.Parameter(torch.zeros(dim))

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in (self.k_lin, self.q_lin, self.v_lin, self.out_lin, self.payload_lin):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        # Identity-centred init: an untrained typed projection should start as a
        # no-op plus noise, so training has to EARN the type specialisation rather
        # than start from random per-relation scrambling.
        eye = torch.eye(self.head_dim)
        with torch.no_grad():
            self.k_rel.copy_(eye.expand_as(self.k_rel).clone())
            self.v_rel.copy_(eye.expand_as(self.v_rel).clone())
            self.k_rel.add_(torch.randn_like(self.k_rel) * 0.02)
            self.v_rel.add_(torch.randn_like(self.v_rel) * 0.02)
            self.rel_prior.fill_(1.0)
            self.skip.zero_()

    def forward(self, x, edge_index, edge_triple, edge_payload):
        n = x.size(0)
        k = self.k_lin(x).view(n, self.heads, self.head_dim)
        q = self.q_lin(x).view(n, self.heads, self.head_dim)
        v = self.v_lin(x).view(n, self.heads, self.head_dim)
        if not self.use_edge_payload:
            edge_payload = torch.zeros_like(edge_payload)
        out = self.propagate(edge_index, k=k, q=q, v=v, edge_triple=edge_triple,
                             edge_payload=edge_payload, size=(n, n))
        out = self.out_lin(torch.nn.functional.gelu(out.view(n, self.dim)))
        alpha = torch.sigmoid(self.skip)
        return self.norm(alpha * self.dropout(out) + (1 - alpha) * x)

    def message(self, k_j, q_i, v_j, edge_triple, edge_payload, index, ptr, size_i):
        # Per-edge typed projection: [E, H, D] x [E, H, D, D] -> [E, H, D].
        k_w = self.k_rel[edge_triple]
        v_w = self.v_rel[edge_triple]
        k = torch.einsum('ehd,ehdf->ehf', k_j, k_w)
        v = torch.einsum('ehd,ehdf->ehf', v_j, v_w)
        logit = (q_i * k).sum(dim=-1) / math.sqrt(self.head_dim)
        logit = logit * self.rel_prior[edge_triple]
        logit = logit + self.payload_lin(edge_payload)
        att = self.attn_dropout(softmax(logit, index, ptr, size_i))
        return v * att.unsqueeze(-1)

    @torch.no_grad()
    def relation_separation(self):
        """Mean pairwise distance between meta-relation projections, K and V.

        BG-HGNN calls the failure mode `relation collapse`: typed weights drift to
        the same matrix, and the model silently degrades into its own no-hetero
        ablation while still reporting a typed architecture. This number makes that
        observable. It is normalised by the weights' own scale, so it is comparable
        across widths, and it is 0 exactly when every relation shares one projection.
        """
        report = {}
        for name, w in (('k_rel', self.k_rel), ('v_rel', self.v_rel)):
            flat = w.reshape(self.num_triples, -1)
            centre = flat.mean(dim=0, keepdim=True)
            spread = (flat - centre).norm(dim=1)
            scale = flat.norm(dim=1).mean().clamp_min(1e-12)
            report[name] = float((spread.mean() / scale))
        return report


class GatedConceptHubLayer(MessagePassing):
    """GCHM's mechanism (gchm_analysis/model.py) ported to the typed v3 graph.

    The native `GCHM` class cannot read this artifact: it requires `node_ids` into a
    193-concept embedding, `node_type in {0,1,2}` with exactly one HUB_TYPE node per
    graph, and a 132-wide numeric payload at slots 199: of a 331-wide vector. v3
    supplies eight node kinds, a fitted token vocabulary, 14 node features and 22 edge
    features. Only the MECHANISM transfers, so it is reimplemented here rather than
    adapted, and the two distinctive ingredients are kept exactly:

    1. **Multiplicative receiver gate (V1).** `content * sigmoid(W_g . x_i)`. The
       receiver's state SCALES the incoming message instead of shifting it, which is
       the second-order concept x state conjunction the native depth sweep isolated
       (depth1 0.5581 -> depth2 0.5827, flat after). `modulation='additive'` is the
       capacity-matched control: identical parameters, but the gate can only shift.
    2. **PNA degree-scaled aggregation.** Four aggregators (mean/min/max/std) times
       three scalers (identity/amplification/attenuation) = 12 x width, versus the
       baseline's single sum. This is the part that distinguishes GCHM from the
       plain edge-conditioned layer independently of the gate.

    Unlike the native star-graph version the receiver here is not always a patient
    hub -- v3 has measurement, analyte, diagnosis and knowledge receivers too -- so
    the gate is applied on every typed edge, which is the same choice already made
    for `EventGCHM` in event_graph_analysis/model.py.

    The edge payload enters the message MLP as in the baseline, so the payload
    ablation contract (`use_edge_payload=False`, parameter count unchanged) holds.
    """

    def __init__(self, dim, edge_dim, degree_histogram, dropout=0.1,
                 use_edge_payload=True, modulation='multiplicative'):
        if modulation not in ('multiplicative', 'additive'):
            raise ValueError("modulation must be 'multiplicative' or 'additive'")
        if degree_histogram.ndim != 1 or degree_histogram.numel() < 1:
            raise ValueError('degree_histogram must be a nonempty vector')
        super().__init__(
            aggr=DegreeScalerAggregation(
                aggr=['mean', 'min', 'max', 'std'],
                scaler=['identity', 'amplification', 'attenuation'],
                deg=degree_histogram.detach().clone(),
                train_norm=False,
            ),
            node_dim=0,
        )
        # A fold whose training graphs are genuinely edgeless has average log degree
        # zero; keep the histogram exact and only floor the numerical denominator.
        self.aggr_module.init_avg_deg_log = max(self.aggr_module.init_avg_deg_log, 1e-6)
        self.aggr_module.avg_deg_log.clamp_(min=1e-6)
        self.use_edge_payload = use_edge_payload
        self.modulation = modulation
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * dim + edge_dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.gate = nn.Linear(dim, dim)
        # 12 aggregator/scaler combinations plus the node's own state.
        self.update_mlp = nn.Sequential(
            nn.Linear(13 * dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        payload = edge_attr
        if not self.use_edge_payload:
            payload = edge_attr.clone()
            payload[:, -PAYLOAD_WIDTH:] = 0
        out = self.propagate(edge_index, x=x, edge_attr=payload)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x + self.dropout(out))

    def message(self, x_j, x_i, edge_attr):
        content = self.message_mlp(torch.cat([x_j, x_i, edge_attr], dim=-1))
        gate = torch.sigmoid(self.gate(x_i))
        if self.modulation == 'multiplicative':
            return content * gate
        return content + gate


class ClinicalGNN(nn.Module):
    def __init__(self, num_tokens, node_dim, edge_dim, num_classes=30,
                 hidden=96, layers=3, dropout=0.1, token_dim=32,
                 use_edge_payload=True, use_message_passing=True,
                 conv='edge_conditioned', num_triples=None, payload_dim=None,
                 heads=4, degree_histogram=None, modulation='multiplicative'):
        super().__init__()
        if conv not in ('edge_conditioned', 'hgt', 'gchm'):
            raise ValueError(f'unknown conv {conv!r}')
        self.use_message_passing = use_message_passing
        self.conv = conv
        self.token = nn.Embedding(num_tokens, token_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(node_dim + token_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden))
        if conv == 'hgt':
            if num_triples is None or payload_dim is None:
                raise ValueError('hgt requires num_triples and payload_dim')
            self.layers = nn.ModuleList([
                HeteroGraphTransformerLayer(hidden, num_triples, payload_dim,
                                            heads, dropout, use_edge_payload)
                for _ in range(layers)])
        elif conv == 'gchm':
            if degree_histogram is None:
                raise ValueError('gchm requires a train-fitted degree_histogram')
            self.register_buffer('degree_histogram',
                                 torch.as_tensor(degree_histogram).float())
            self.layers = nn.ModuleList([
                GatedConceptHubLayer(hidden, edge_dim, self.degree_histogram,
                                     dropout, use_edge_payload, modulation)
                for _ in range(layers)])
        else:
            self.layers = nn.ModuleList([
                EdgeConditionedLayer(hidden, edge_dim, dropout, use_edge_payload)
                for _ in range(layers)])
        # Sum pooling scales with graph size, and these graphs range from 1 to
        # thousands of nodes. Without normalising the pooled vector the logit scale
        # tracks node count (measured |z|max ~80, std ~9 at init), the softmax
        # saturates and gradients vanish -- the run then sits at the majority class
        # with a frozen loss. Normalising here makes the readout size-independent
        # while keeping sum's multiplicity information in the direction of the vector.
        self.pool_norm = nn.LayerNorm(2 * hidden)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))

    def forward(self, data):
        h = self.encoder(torch.cat([data.x, self.token(data.token)], dim=-1))
        if self.use_message_passing:
            for layer in self.layers:
                if self.conv == 'hgt':
                    h = layer(h, data.edge_index, data.edge_triple, data.edge_payload)
                else:
                    h = layer(h, data.edge_index, data.edge_attr)
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        size = int(batch.max()) + 1
        pooled = torch.cat([scatter(h, batch, dim=0, dim_size=size, reduce='sum'),
                            scatter(h, batch, dim=0, dim_size=size, reduce='mean')], dim=-1)
        return self.head(self.pool_norm(pooled))

    def parameter_count(self):
        """All allocated parameters, including bypassed NOMP conv layers."""
        return sum(p.numel() for p in self.parameters())

    def active_parameter_count(self):
        """Parameters in enabled forward blocks, not minibatch nonzero gradients.

        NOMP keeps the conv state for checkpoint compatibility but never executes
        it. Count it only with message passing enabled. This is a block-level
        count: unused token rows or numerically masked input columns are not removed.
        """
        if self.use_message_passing:
            return self.parameter_count()
        bypassed = {id(p) for p in self.layers.parameters()}
        return sum(p.numel() for p in self.parameters() if id(p) not in bypassed)

    def relation_separation(self):
        """Per-layer BG-HGNN relation-collapse report; empty for untyped arms."""
        return [layer.relation_separation() for layer in self.layers
                if isinstance(layer, HeteroGraphTransformerLayer)]
