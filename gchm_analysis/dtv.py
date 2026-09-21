"""DTV-GNN: Direct Threshold Voting graph network for patient star graphs.

Designed from four measurements on this exact artifact, not from a prior about
what tabular architectures should look like.

Measurement 1 — the interaction order that matters is exactly two.
    An XGBoost depth sweep on the pinned input gives depth1 0.5581, depth2 0.5827,
    depth3 0.5791, depth4 0.5835 (validation macro-F1). Everything the tree
    ensemble earns over a purely additive model, it earns at order two, and
    deeper interaction adds nothing. A model therefore needs an additive term and
    one pairwise term; depth beyond that is wasted capacity on this dataset.

Measurement 2 — the incumbent GNN is operating at additive level.
    A plain linear model over threshold codes plus concept indicators reaches
    0.5508 (CE) / 0.5458 (sqrt) validation macro-F1. The incumbent GCHM, with
    179,486 parameters and two rounds of gated message passing, reaches 0.5593.
    The entire machinery is worth about +0.009 over a linear model, and still
    sits below the depth-1 tree at 0.5581. Its nominal second-order mechanism
    (a FiLM gate) is not delivering second-order value.

Measurement 3 — the hub bottleneck destroys the thresholds.
    Probing the trained incumbent: a linear read-out recovers threshold codes
    from its 64-dimensional hub state at R^2 = 0.200, but from the raw 132 fields
    at R^2 = 0.495. The 132->64 projection discards 59.6% of the recoverable
    threshold information *before* any interaction or classification happens.
    This is why two earlier fixes failed: piecewise-linear encoding (0.5539) and
    a hub skip path (0.5518) both improved what enters the projection while
    leaving the projection itself in the path to the classifier.

Measurement 4 — the per-class split follows single-feature separability.
    On classes where XGBoost is uniquely correct most often, the best single
    feature-threshold rule already reaches mean validation F1 0.268 (Hypokalemia
    0.336 on sodium, Diabetes 0.488 on antidiabetic medication). On the classes
    where the GNN is uniquely correct, the same rule reaches only 0.125 (Sepsis
    0.155, URI 0.025). The tree model wins precisely where one threshold decides
    the label, and loses where the answer is a combination.

Design consequence
------------------
Every term votes on the 30 classes *directly*. Nothing is compressed into a
shared latent width before reaching the classifier, because measurement 3 shows
that is where the signal dies.

    logit = concept_votes + threshold_votes + interaction_votes

- ``concept_votes``  one learned 30-vector per concept identity, aggregated over
  the real star edges with PNA degree scaling. This stays a genuine GNN: the
  votes travel along edges and honour edge masks.
- ``threshold_votes`` one learned 30-vector per (continuous feature, bin) pair,
  read off the train-fitted piecewise-linear quantile code, plus one 30-vector
  per binary hub field. A single feature crossing a single threshold can move a
  class score on its own — the capability measurement 4 says decides Hypokalemia
  and Diabetes.
- ``interaction_votes`` a factorization-machine style second-order term: each
  concept carries a rank-k vector, the hub code produces one rank-k vector, and
  their inner product scales that concept's class vote. This is the concept x
  hub conjunction, at exactly order two (measurement 1), computed while feature
  identity is still intact (measurement 3).

Regularization comes from structure rather than from a narrow bottleneck: the
interaction is rank-limited, and each additive term is confined to its own
feature. The incumbent overfits from epoch 7 with 179K parameters on 59.6K
training graphs; here the parameter count is dominated by sparse per-feature
vote tables that each see only their own feature's signal.

Topology, artifact, split, label order and loss are unchanged.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import DegreeScalerAggregation

HUB_TYPE = 2


class ConceptVoteConv(MessagePassing):
    """Carry each concept's class vote to its patient hub over the real edges.

    Aggregation is a SUM, not PNA's degree-scaled mean/min/max/std, and that
    choice is measured rather than inherited. Fitting the same additive model
    with different aggregation semantics for the concept term gives:

        sum of present concepts   0.5469
        degree^-0.5 scaling       0.5314
        mean over concepts        0.4940

    Vote counting is the whole point of this term: a patient on four
    cardiac-related medications should push the cardiac classes harder than a
    patient on one. Every PNA aggregator is normalized, so it discards exactly
    that evidence — which is why the incumbent's PNA readout could not exploit
    concept multiplicity. Sum aggregation is also the standard expressiveness
    argument for GIN over mean/max GNNs, so this is not an unusual choice.

    Degree is not thrown away: the count of contributing edges is available to
    the model through the hub's own ``n_medications`` field and through the
    threshold-vote term, where it can be used as a feature rather than forced
    into the aggregator as a normalizer.

    Edge masks are applied once, in ``message``; ``explain_message`` is the
    identity so PyG cannot square them.
    """

    def __init__(self, classes: int):
        super().__init__(aggr="sum", node_dim=0)
        self.classes = classes
        self._edge_gate = None

    def forward(self, votes, edge_index, *, edge_mask=None):
        if self.explain and self._edge_mask is not None:
            if edge_mask is not None:
                raise ValueError("Use either explicit edge_mask or the PyG mask, not both")
            edge_mask = self._edge_mask.sigmoid() if self._apply_sigmoid else self._edge_mask
        gate = votes.new_ones(edge_index.size(1)) if edge_mask is None else edge_mask
        if gate.ndim != 1 or gate.numel() != edge_index.size(1):
            raise ValueError("One mask value per real directed edge required")
        self._edge_gate = gate
        try:
            return self.propagate(edge_index, x=votes)
        finally:
            self._edge_gate = None

    def message(self, x_j):
        return x_j * self._edge_gate[:, None]

    def explain_message(self, inputs, dim_size):
        return inputs


class DTVGNN(nn.Module):
    """Direct Threshold Voting GNN.

    Args:
        hub_encoder: fitted ``PiecewiseLinearQuantileHub``. Required — the whole
            design rests on threshold codes, so there is no raw-value fallback.
        rank: width of the second-order interaction. Small by construction; the
            depth sweep says order two is all this dataset supports.
    """

    def __init__(self, classes: int = 30, num_concepts: int = 193,
                 hub_dim: int = 132, hub_encoder: nn.Module | None = None,
                 degree_histogram: torch.Tensor | None = None,
                 rank: int = 24, dropout: float = 0.1,
                 interaction_scale: float = 1.0):
        super().__init__()
        if hub_encoder is None:
            raise ValueError("DTV-GNN requires a fitted threshold encoder")
        if rank < 1:
            raise ValueError("rank must be positive")
        if getattr(hub_encoder, "hub_dim", hub_dim) != hub_dim:
            raise ValueError("Hub encoder width must match the native payload")
        self.classes = classes
        self.hub_dim = hub_dim
        self.rank = rank
        self.interaction_scale = float(interaction_scale)
        self.hub_encoder = hub_encoder
        self.n_cont = len(hub_encoder.continuous)
        self.n_bin = len(hub_encoder.binary)
        self.bins = hub_encoder.bins

        # Term 1: one class vote per concept identity, delivered over star edges.
        self.concept_vote = nn.Embedding(num_concepts, classes)
        nn.init.zeros_(self.concept_vote.weight)
        self.conv = ConceptVoteConv(classes)

        # Term 2: one class vote per (continuous feature, bin), plus per binary
        # field. Block-structured on purpose: a feature's vote depends only on
        # that feature, so a single threshold can decide a class by itself.
        self.threshold_vote = nn.Parameter(torch.zeros(self.n_cont, self.bins, classes))
        self.missing_vote = nn.Parameter(torch.zeros(self.n_cont, classes))
        self.binary_vote = nn.Parameter(torch.zeros(self.n_bin, classes))
        self.bias = nn.Parameter(torch.zeros(classes))

        # Term 3: order-two concept x hub conjunction, rank-limited.
        self.concept_factor = nn.Embedding(num_concepts, rank)
        nn.init.normal_(self.concept_factor.weight, std=0.05)
        self.hub_factor = nn.Linear(hub_encoder.out_dim, rank)
        self.interaction_vote = nn.Embedding(num_concepts, classes)
        nn.init.zeros_(self.interaction_vote.weight)
        # Dropout belongs on the interaction factor, not on the summed votes.
        # Dropping whole class-vote totals injects noise directly into the logit
        # scale and destabilises the additive terms; dropping factor dimensions
        # regularises only the second-order path, which is the part with the
        # capacity to overfit.
        self.drop = nn.Dropout(dropout)

    def _hub_rows(self, data, types, batch):
        hub = (types == HUB_TYPE).nonzero().flatten()
        hub = hub[batch[hub].argsort()]
        if not torch.equal(batch[hub], torch.arange(int(batch.max()) + 1, device=types.device)):
            raise ValueError("Exactly one hub per graph required")
        return hub

    def forward(self, data, *, edge_mask=None, no_messages=False):
        x, edges, types = data.x, data.edge_index, data.node_type
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if types.shape != (x.size(0),) or not torch.all((types >= 0) & (types <= 2)):
            raise ValueError("Stable node_type metadata required")
        if not torch.equal(batch[edges[0]], batch[edges[1]]):
            raise ValueError("cross-graph edges are forbidden")
        hub_rows = self._hub_rows(data, types, batch)
        if no_messages:
            if edge_mask is not None:
                raise ValueError("no_messages and edge_mask are mutually exclusive")
            edge_mask = x.new_zeros(edges.size(1))

        # ---- hub threshold code, feature identity intact ----
        payload = x[hub_rows, 199:199 + self.hub_dim]
        code = self.hub_encoder(payload)
        n = code.size(0)
        cont = code[:, :self.n_cont * self.bins].view(n, self.n_cont, self.bins)
        missing = code[:, self.n_cont * self.bins:self.n_cont * self.bins + self.n_cont]
        binary = code[:, self.n_cont * self.bins + self.n_cont:]

        # ---- Term 2: direct per-feature votes, no shared projection ----
        threshold_logits = (torch.einsum("nfb,fbc->nc", cont, self.threshold_vote)
                            + missing @ self.missing_vote
                            + binary @ self.binary_vote)

        # ---- Term 1: concept votes carried over the real edges ----
        votes = self.concept_vote(data.node_ids)
        votes = votes * (types != HUB_TYPE).unsqueeze(-1)  # the hub casts no identity vote
        concept_logits = self.conv(votes, edges, edge_mask=edge_mask)[hub_rows]

        # ---- Term 3: order-two concept x hub conjunction ----
        z = self.drop(self.hub_factor(code))                        # [graphs, rank]
        factors = self.concept_factor(data.node_ids)                # [nodes, rank]
        pair_votes = self.interaction_vote(data.node_ids)           # [nodes, classes]
        strength = (factors * z[batch]).sum(-1, keepdim=True)       # <e_j, z_patient>
        modulated = pair_votes * strength
        modulated = modulated * (types != HUB_TYPE).unsqueeze(-1)
        # The conjunction is edge-derived content: silence it exactly as message
        # passing is silenced, so an edge ablation cannot understate dependence.
        interaction_logits = self.conv(modulated, edges, edge_mask=edge_mask)[hub_rows]

        return (self.bias + threshold_logits
                + concept_logits
                + self.interaction_scale * interaction_logits)
