"""GCHM: Gated Concept-Hub Modulation — a message-passing GNN for patient star graphs.

Motivation is measured, not assumed. Diagnostic probes on the pinned native artifact
established three facts about the incumbent GNN family:

  1. The PyG PNAConv message function with pre_layers=1 is EXACTLY additively
     separable: m(x_i, x_j) = W_i·x_i + W_j·x_j (verified residual 4.8e-07, float32
     roundoff). Patient hub state can therefore never multiply concept identity.
  2. A depth sweep showed the entire tabular advantage is SECOND ORDER: depth1 (purely
     additive) 0.5581 -> depth2 0.5827 (+0.025), while depth 3/4/6 add nothing. The
     missing ingredient is pairwise concept x hub conjunction, not deep interaction.
  3. Concepts alone score 0.4727 and hub alone 0.3545, but jointly 0.5825 — the signal
     lives in the intersection of the two blocks.

GCHM closes exactly that gap while remaining a genuine GNN: messages still flow over
the real star edges, aggregation is still PNA degree-scaled, and edge masks remain
honoured for GraphXAI.

Design, each element traced to a measured vulnerability:

  V1 (additive messages)   -> FiLM gate: every message is modulated multiplicatively by
                              the RECEIVER's state, sigma(W_g · h_i). At the hub this is
                              literally concept x patient-state, the second-order term.
  V3 (331->64 shared bottleneck, hub and concept forced through one subspace)
                           -> separate encoders: concept identity via embedding, hub
                              numeric payload via its own MLP. No shared compression.
  V6 (no activation inside conv; pre_nn and post_nn are single Linear layers)
                           -> explicit ReLU inside both the message MLP and the update.

Topology, artifact, split and label order are unchanged. This is an architecture change
only; it introduces no new input information and no concept-concept edges.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.aggr import DegreeScalerAggregation

HUB_TYPE = 2


class GatedConceptHubConv(MessagePassing):
    """Receiver-conditioned FiLM message passing with PNA degree-scaled aggregation.

    The gate is computed from the destination node. On a patient star graph the
    destination of a concept->hub edge IS the hub, so the gate is the patient's
    clinical state and the product gate * message is the concept x hub conjunction
    the depth sweep identified as the missing second-order term.

    Edge masks are applied once, to the final message, and ``explain_message`` is the
    identity so PyG's automatic masking cannot square them.
    """

    def __init__(self, width: int, degree_histogram: torch.Tensor, dropout: float = 0.1,
                 modulation: str = "multiplicative"):
        if width < 1:
            raise ValueError("width must be positive")
        if modulation not in ("multiplicative", "additive"):
            raise ValueError("modulation must be 'multiplicative' or 'additive'")
        aggr = DegreeScalerAggregation(
            aggr=["mean", "min", "max", "std"],
            scaler=["identity", "amplification", "attenuation"],
            deg=degree_histogram.detach().clone(),
            train_norm=False,
        )
        super().__init__(aggr=aggr, node_dim=0)
        self.modulation = modulation
        # A genuinely edgeless training set has average log degree zero; keep the
        # histogram exact and only floor the numerical denominator.
        self.aggr_module.init_avg_deg_log = max(self.aggr_module.init_avg_deg_log, 1e-6)
        self.aggr_module.avg_deg_log.clamp_(min=1e-6)

        # V6: genuine nonlinearity inside the message function.
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, width)
        )
        # V1: multiplicative FiLM gate from the receiver's state.
        self.gate = nn.Linear(width, width)
        self.update_mlp = nn.Sequential(
            nn.Linear(12 * width + width, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self._edge_gate = None

    def forward(self, x, edge_index, *, edge_mask=None):
        # Consume the exact real-edge PyG explanation mask before building messages.
        if self.explain and self._edge_mask is not None:
            if edge_mask is not None:
                raise ValueError("Use either explicit edge_mask or the PyG mask, not both")
            edge_mask = self._edge_mask.sigmoid() if self._apply_sigmoid else self._edge_mask
        gate = x.new_ones(edge_index.size(1)) if edge_mask is None else edge_mask
        if gate.ndim != 1 or gate.numel() != edge_index.size(1):
            raise ValueError("One mask value per real directed edge required")
        self._edge_gate = gate
        try:
            aggregated = self.propagate(edge_index, x=x)
        finally:
            self._edge_gate = None
        return self.update_mlp(torch.cat([aggregated, x], dim=-1))

    def message(self, x_i, x_j):
        content = self.message_mlp(torch.cat([x_i, x_j], dim=-1))
        gate = torch.sigmoid(self.gate(x_i))
        if self.modulation == "multiplicative":
            # Second-order term: receiver state multiplies sender content.
            modulated = content * gate
        else:
            # Ablation control: identical parameters and compute, but the receiver
            # state can only SHIFT the message, never scale it. Any advantage of the
            # multiplicative arm over this one is attributable to interaction order.
            modulated = content + gate
        return modulated * self._edge_gate[:, None]

    def explain_message(self, inputs, dim_size):
        """Mask already applied in `message`; no automatic second application."""
        return inputs


class ConceptPairMixer(nn.Module):
    """Let concepts of the SAME patient interact directly, without the hub bottleneck.

    Measured motivation: on a star graph every concept is adjacent only to the hub, so
    two concepts can only meet after hub aggregation has already averaged them. A depth
    sweep showed the tabular advantage is second order (depth1 0.5581 -> depth2 0.5827,
    flat thereafter) and gain attribution put the signal on chief-complaint x hub-lab
    conjunctions. The `cooccur` topology cannot supply this: it adds only 5% more edges
    and leaves 83.5% of patients with a byte-identical graph, and its extra edges are
    med<->med, not the complaint axis that carries the gain.

    This module therefore adds the missing pairwise path in the MODEL, leaving the
    artifact, split, label order and star topology untouched. Each concept receives a
    low-rank bilinear summary of its same-patient peers. Self-pairs are excluded and
    the mean uses the true peer count, so a patient's own concept cannot inflate it.
    """

    def __init__(self, width: int, rank: int = 16):
        super().__init__()
        if width < 1 or rank < 1:
            raise ValueError("width and rank must be positive")
        self.left = nn.Linear(width, rank, bias=False)
        self.right = nn.Linear(width, rank, bias=False)
        self.out = nn.Linear(rank, width, bias=False)

    def forward(self, h, batch, is_concept):
        delta = torch.zeros_like(h)
        idx = is_concept.nonzero().flatten()
        if idx.numel() == 0:
            return delta
        left = self.left(h[idx])
        right = self.right(h[idx])
        graph = batch[idx]
        n_graphs = int(batch.max()) + 1
        # Sum of peer projections per patient, then subtract self to exclude self-pairs.
        totals = right.new_zeros((n_graphs, right.size(-1)))
        totals.index_add_(0, graph, right)
        counts = right.new_zeros(n_graphs)
        counts.index_add_(0, graph, torch.ones_like(graph, dtype=right.dtype))
        peers = (totals[graph] - right)
        n_peers = (counts[graph] - 1).clamp(min=1).unsqueeze(-1)
        # Bilinear interaction against the averaged peers; a lone concept gets zero.
        interaction = torch.tanh(left * (peers / n_peers))
        alone = (counts[graph] <= 1).unsqueeze(-1)
        interaction = torch.where(alone, torch.zeros_like(interaction), interaction)
        delta = delta.index_copy(0, idx, self.out(interaction))
        return delta


def fit_hub_quantiles(hub, train_mask, bins=8, missing_value=0.0, min_unique=3):
    """Fit per-column quantile bin edges on the TRAINING fold only.

    Returns ``(edges, continuous, binary, degenerate)`` where ``edges`` has shape
    ``[C, bins + 1]`` for the ``C`` continuous columns.

    Leakage guard: every quantile is computed from rows where ``train_mask`` is
    true. Validation and test rows never enter the fitted statistics, exactly as
    the native z-scaler is train-only.

    Columns with at most ``min_unique - 1`` distinct training values are treated as
    binary/categorical and bypass the encoder unchanged.

    Quantiles are fitted on the NON-missing values only. The native contract states
    ``pnum NaN -> z=0``, so the mass sitting exactly at 0.0 is the imputed-missing
    mass; including it would collapse a third of the bin edges onto one point and
    waste the encoder's resolution on a value that the missingness channel already
    reports. A genuine observation landing exactly on the float32 training mean is
    indistinguishable from missing here, which is the same ambiguity the tabular
    baseline works with.
    """
    import numpy as np

    hub = np.asarray(hub)
    train_mask = np.asarray(train_mask, dtype=bool)
    if hub.ndim != 2:
        raise ValueError("hub must be a 2D array")
    if train_mask.shape != (hub.shape[0],):
        raise ValueError("train_mask must cover every hub row")
    if not train_mask.any():
        raise ValueError("Empty training fold; quantiles cannot be fitted")
    if bins < 1:
        raise ValueError("bins must be positive")
    fit = hub[train_mask]
    uniques = np.array([len(np.unique(fit[:, j])) for j in range(fit.shape[1])])
    continuous = np.flatnonzero(uniques >= min_unique)
    binary = np.flatnonzero(uniques < min_unique)
    probs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.empty((len(continuous), bins + 1), dtype=np.float32)
    degenerate = 0
    for row, j in enumerate(continuous):
        column = fit[:, j]
        observed = column[column != missing_value]
        if observed.size < bins + 1:
            observed = column
        q = np.quantile(observed.astype(np.float64), probs).astype(np.float32)
        # Enforce strict increase in float32, the dtype the encoder divides in.
        # Repairing in float64 and casting afterwards silently collapses the nudge
        # back onto the same float32 value, giving a zero-width bin and NaN codes.
        # A coincident quantile is legitimate (mass piled on one value); the
        # one-ulp bin then encodes an exact "x > value" step, which is the correct
        # reading, not an error to smooth away.
        for t in range(1, len(q)):
            if not q[t] > q[t - 1]:
                q[t] = np.nextafter(q[t - 1], np.float32(np.inf), dtype=np.float32)
                degenerate += 1
        if not np.all(np.diff(q) > 0):
            raise ValueError(f"Could not build strictly increasing edges for column {j}")
        edges[row] = q
    return edges, continuous.astype(np.int64), binary.astype(np.int64), int(degenerate)


class PiecewiseLinearQuantileHub(nn.Module):
    """Give the hub encoder the threshold capability a linear projection lacks.

    Measured motivation, from the selected XGBoost-sqrt seed-1234 booster on this
    exact artifact: 80.1% of all 307,916 tree splits land on hub columns, using
    **4,952 distinct thresholds** across 131 hub features (median 2, max 208 per
    feature), and those splits carry 48.7% of total gain. GCHM feeds the same 132
    numbers through ``Linear(132, 64)``. A linear map cannot represent a step, so
    the incumbent has to spend MLP capacity approximating every threshold the
    boosted trees obtain for free from greedy histogram search. This is the classic
    axis-aligned-split deficit of tabular deep learning.

    Two re-encodings, no new information — every output is a deterministic function
    of the same 132 stored floats:

    1. **Piecewise-linear quantile code** (Gorishniy et al., NeurIPS 2022). For bin
       edges ``b_0 < ... < b_B`` fitted on the training fold,

           ``code_t(x) = clip((x - b_t) / (b_{t+1} - b_t), 0, 1)``

       is monotone and piecewise linear, so a single linear layer on top of it can
       place an independent knot at every edge, i.e. learn a step-like response.
       Below ``b_0`` the code is all zeros and above ``b_B`` all ones, so the
       encoding extrapolates monotonically rather than clipping information away.

    2. **Missingness channel.** 53 of the 74 continuous hub columns carry more than
       30% of their training mass at exactly 0.0 (median 33.8%). Because the native
       pipeline imputes missing numerics to z=0, a linear encoder reads a missing
       lab as "average patient", which is a strong and wrong claim. The indicator
       separates "not measured" from "measured, near the mean".

    Binary/categorical hub columns bypass the encoder unchanged.
    """

    def __init__(self, edges, continuous, binary, hub_dim):
        super().__init__()
        if edges.ndim != 2 or edges.shape[0] != len(continuous):
            raise ValueError("One edge row per continuous column required")
        if edges.shape[1] < 2:
            raise ValueError("At least one bin (two edges) required")
        if len(continuous) + len(binary) != hub_dim:
            raise ValueError("Continuous and binary columns must partition the hub")
        self.hub_dim = int(hub_dim)
        self.bins = int(edges.shape[1] - 1)
        self.register_buffer("edges", torch.as_tensor(edges, dtype=torch.float32))
        self.register_buffer("continuous", torch.as_tensor(continuous, dtype=torch.long))
        self.register_buffer("binary", torch.as_tensor(binary, dtype=torch.long))
        # Fitted constants travel inside the checkpoint, so a replay cannot silently
        # re-fit them against a different fold.
        self.out_dim = len(continuous) * self.bins + len(continuous) + len(binary)

    def forward(self, values):
        if values.size(-1) != self.hub_dim:
            raise ValueError("Hub payload width mismatch")
        x = values[:, self.continuous].unsqueeze(-1)
        lo = self.edges[:, :-1]
        width = self.edges[:, 1:] - lo
        code = ((x - lo) / width).clamp(0.0, 1.0)
        missing = (values[:, self.continuous] == 0.0).to(values.dtype)
        # A missing entry gets an all-zero code plus its own flag, so it is not
        # read as "smallest observed value".
        code = code * (1.0 - missing).unsqueeze(-1)
        return torch.cat([code.flatten(1), missing, values[:, self.binary]], dim=-1)


class GCHM(nn.Module):
    """Separate modality encoders + gated message passing + hub readout.

    Concept nodes carry a one-hot identity; the hub carries the 132 native numeric
    fields. Encoding them separately (V3) prevents one 331->width projection from
    forcing both through a single shared subspace.
    """

    def __init__(self, classes: int = 30, width: int = 64, layers: int = 2,
                 num_concepts: int = 193, hub_dim: int = 132, dropout: float = 0.1,
                 degree_histogram: torch.Tensor | None = None,
                 modulation: str = "multiplicative", concept_pairs: bool = False,
                 rank: int = 16, concept_dropout: float = 0.0,
                 hub_encoder: nn.Module | None = None, hub_skip: bool = False):
        super().__init__()
        if layers < 1:
            raise ValueError("At least one message-passing layer required")
        if degree_histogram is None:
            raise ValueError("PNA degree histogram is required")
        if not 0.0 <= concept_dropout < 1.0:
            raise ValueError("concept_dropout must be in [0, 1)")
        self.concept_dropout = float(concept_dropout)
        self.width = width
        self.hub_dim = hub_dim
        self.modulation = modulation
        self.concept_pairs = concept_pairs
        # V3: identity and clinical state get their own parameter blocks.
        self.concept = nn.Embedding(num_concepts, width)
        # V7 (measured): a linear hub projection cannot express an axis-aligned
        # threshold, while 80.1% of the tuned XGBoost splits are exactly that.
        # When a fitted encoder is supplied the hub MLP reads its code instead of
        # the raw z-scores; the default stays byte-identical to the incumbent.
        self.hub_encoder = hub_encoder
        if hub_encoder is not None:
            if getattr(hub_encoder, "hub_dim", hub_dim) != hub_dim:
                raise ValueError("Hub encoder width must match the native payload")
            hub_in = hub_encoder.out_dim
        else:
            hub_in = hub_dim
        self.hub = nn.Sequential(
            nn.Linear(hub_in, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width),
        )
        self.convs = nn.ModuleList(
            [GatedConceptHubConv(width, degree_histogram, dropout, modulation)
             for _ in range(layers)]
        )
        self.mixers = nn.ModuleList(
            [ConceptPairMixer(width, rank) for _ in range(layers)]
        ) if concept_pairs else None
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        # V8 (measured): the readout sees only the width-64 hub state, so all 132
        # numeric fields must survive a 132->64 bottleneck before they can reach a
        # class score. XGBoost has no such bottleneck: every one of the 4,952 hub
        # thresholds addresses the classifier directly. The skip path restores that
        # direct route by letting the encoded hub code reach the head alongside the
        # message-passing state. It adds no new information (the same encoder output
        # already feeds self.hub) and no new edges; it only removes a width
        # constraint the incumbent imposes on information that is already present.
        self.hub_skip = bool(hub_skip)
        if self.hub_skip and hub_encoder is None:
            raise ValueError("hub_skip requires a fitted hub encoder")
        head_in = width + (hub_in if self.hub_skip else 0)
        self.head = nn.Sequential(
            nn.Linear(head_in, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, classes)
        )

    def encode(self, data):
        x, ids, types = data.x, data.node_ids, data.node_type
        h = self.concept(ids)
        hub = types == HUB_TYPE
        code = None
        if hub.any():
            # Native layout: hub numeric payload occupies slots 199: of the 331 vector.
            h = h.clone()
            payload = x[hub, 199:199 + self.hub_dim]
            if self.hub_encoder is not None:
                payload = self.hub_encoder(payload)
            code = payload
            h[hub] = self.hub(payload)
        if self.training and self.concept_dropout > 0.0:
            # One inverted-dropout decision per concept occurrence, shared across
            # its embedding channels. Unlike elementwise dropout this prevents
            # relying on a concept's remaining channels to recover its identity.
            # Only input content is perturbed: keep hub values, nodes, edges and
            # PNA degrees intact. This is NOT an edge-removal intervention; a
            # zeroed concept can still receive hub messages in later layers.
            concept = ~hub
            keep = nn.functional.dropout(
                torch.ones_like(h[concept, :1]), p=self.concept_dropout, training=True
            )
            h = h.clone()
            h[concept] = h[concept] * keep
        return h, code

    def forward(self, data, *, edge_mask=None, no_messages=False):
        x, edges, types = data.x, data.edge_index, data.node_type
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if types.shape != (x.size(0),) or not torch.all((types >= 0) & (types <= 2)):
            raise ValueError("Stable node_type metadata required")
        if not torch.equal(batch[edges[0]], batch[edges[1]]):
            raise ValueError("cross-graph edges are forbidden")
        hub = (types == HUB_TYPE).nonzero().flatten()
        hub = hub[batch[hub].argsort()]
        if not torch.equal(batch[hub], torch.arange(int(batch.max()) + 1, device=x.device)):
            raise ValueError("Exactly one hub per graph required")
        if no_messages:
            if edge_mask is not None:
                raise ValueError("no_messages and edge_mask are mutually exclusive")
            edge_mask = x.new_zeros(edges.size(1))
        h, code = self.encode(data)
        is_concept = types != HUB_TYPE
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            delta = torch.relu(conv(h, edges, edge_mask=edge_mask))
            if self.mixers is not None:
                # The pair path is an edge-derived content route: it must obey the same
                # silencing as message passing, or an edge ablation would understate
                # the model's true dependence on graph content.
                if no_messages:
                    scale = h.new_zeros(1)
                elif edge_mask is not None:
                    scale = edge_mask.mean().clamp(0., 1.)
                else:
                    scale = h.new_ones(1)
                delta = delta + scale * self.mixers[i](h, batch, is_concept)
            h = norm(h + delta)
        state = h[hub]
        if self.hub_skip:
            # The skip carries the patient's OWN observed numerics, which are node
            # content, not edge-derived content. Unlike the concept-pair mixer it is
            # therefore NOT silenced by an edge ablation: zeroing every edge should
            # leave a patient's own labs readable, exactly as the hub row itself
            # survives `no_messages`. Silencing it would overstate the model's
            # dependence on message passing.
            if code is None:
                raise ValueError("hub_skip requires an encoded hub payload")
            state = torch.cat([state, code], dim=-1)
        return self.head(state)


class GraphXAIWrapper(nn.Module):
    """Bind original node order/ids/types while explainers perturb continuous features."""

    def __init__(self, model, node_ids, node_type):
        super().__init__()
        self.model = model
        self.register_buffer("node_ids", node_ids.detach().clone())
        self.register_buffer("node_type", node_type.detach().clone())

    def forward(self, x, edge_index, batch=None):
        from torch_geometric.data import Data
        return self.model(Data(x=x, edge_index=edge_index, batch=batch,
                               node_ids=self.node_ids, node_type=self.node_type))
