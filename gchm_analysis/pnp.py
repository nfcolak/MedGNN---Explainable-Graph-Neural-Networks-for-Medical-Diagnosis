"""PNP: Patient Neighbourhood Propagation for transductive diagnosis prediction.

This is a different model from the star-graph family, because it answers a
different question. The star graph carries no information beyond the feature
vector (concept set determines the edge list; XGBoost's 192 presence columns
reconstruct every graph exactly), so message passing there can only re-deliver
what the hub already holds. The patient-patient neighbourhood does carry extra
information, and it is information a row-independent tree model cannot consume:
other patients' labels are not features of this patient.

Measured before writing this model:

    neighbour label homophily (k=25)       0.412  vs 0.063 random-pair baseline
    true label present in 25-neighbourhood 0.870
    recovered by plain majority vote       0.525
    XGBoost alone                          0.639
    XGBoost + majority-vote blend          +0.0082 on a held-out half
    the same blend with RANDOM neighbours  -0.0010

Two numbers set the design. The random-neighbour control at -0.0010 says the
signal is structural rather than smoothing, so propagation is worth building.
The gap between 0.525 and 0.870 says a fixed aggregator wastes most of what the
neighbourhood knows, so the aggregator must be learned — which is precisely what
a GNN provides over a kNN vote.

Design
------
Three terms, all voting directly on the 30 classes, following the finding that
compressing through a narrow shared width destroys threshold information (a
probe on the trained incumbent recovered threshold codes at R^2 0.200 from its
64-dim hub state versus 0.495 from the raw 132 fields):

    logit = own_evidence + neighbour_label_evidence + neighbour_feature_evidence

- ``own_evidence``: the patient's own threshold and concept votes. Without this
  the model would be a pure label propagator and could not beat the feature-based
  baseline on patients whose neighbourhood is uninformative.
- ``neighbour_label_evidence``: attention-weighted aggregation of neighbours'
  one-hot training labels. Attention is computed from the pair (distance,
  own state, neighbour state), so the model can learn *which* neighbours to
  trust instead of counting all of them equally. This is the term that closes
  the 0.525 -> 0.870 gap if anything does.
- ``neighbour_feature_evidence``: attention-weighted aggregation of neighbour
  feature states, letting the model use a neighbour's record even when its label
  is misleading.

Leakage
-------
Neighbour labels come only from the training fold, and a training row never sees
itself (enforced in the artifact and re-asserted at load). During training the
model sees its own label as a target only, never as an input feature.
"""
from __future__ import annotations

import torch
from torch import nn


class NeighbourAttention(nn.Module):
    """Learned, distance-aware attention over a fixed-size neighbourhood.

    Scores are computed from the query patient's state, the neighbour's state and
    the pair distance. Softmax is taken over the k neighbours of each patient, so
    the weights are per-patient and sum to one — a patient in a dense, coherent
    region can concentrate on a few neighbours, while a patient in a sparse region
    spreads out rather than being forced into a hard choice.
    """

    def __init__(self, width: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if width % heads:
            raise ValueError("width must divide evenly across heads")
        self.heads = heads
        self.dim = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        # Distance enters the score directly: the artifact's distances are
        # meaningful (euclidean in the train-fitted scaled space), so discarding
        # them would throw away the ranking the graph was built from.
        self.distance = nn.Linear(1, heads)
        self.drop = nn.Dropout(dropout)

    def forward(self, own, neighbour, distance):
        n, k, _ = neighbour.shape
        q = self.query(own).view(n, self.heads, 1, self.dim)
        kk = self.key(neighbour).view(n, k, self.heads, self.dim).transpose(1, 2)
        score = (q * kk).sum(-1) / self.dim ** 0.5              # [n, heads, k]
        score = score + self.distance(distance.unsqueeze(-1)).permute(0, 2, 1)
        weight = self.drop(score.softmax(-1))
        return weight


class PNP(nn.Module):
    """Patient Neighbourhood Propagation.

    Args:
        feature_dim: width of the per-patient encoded feature vector.
        neighbour_labels: LongTensor of training labels, indexed by neighbour id.
        classes: number of diagnosis classes.
        label_smoothing: mass reserved off the observed neighbour label. A
            neighbour's label is evidence, not ground truth for this patient, so
            propagating a hard one-hot overstates it.
    """

    def __init__(self, feature_dim: int, neighbour_labels: torch.Tensor,
                 classes: int = 30, width: int = 128, heads: int = 4,
                 dropout: float = 0.1, label_smoothing: float = 0.05,
                 neighbour_scale: float = 1.0):
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.classes = classes
        self.neighbour_scale = float(neighbour_scale)
        self.register_buffer("neighbour_labels", neighbour_labels.detach().clone().long())

        self.encode = nn.Sequential(
            nn.Linear(feature_dim, width), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.attention = NeighbourAttention(width, heads, dropout)
        self.heads = heads

        # Own evidence votes straight onto the classes.
        self.own_vote = nn.Linear(width, classes)
        # Neighbour-label evidence: one learned gain per head, so the model can
        # decide how much a head's label consensus is worth.
        self.label_gain = nn.Parameter(torch.zeros(heads, classes))
        # Neighbour-feature evidence.
        self.neighbour_vote = nn.Linear(width, classes)
        self.smoothing = float(label_smoothing)
        self.drop = nn.Dropout(dropout)

    def label_matrix(self, ids):
        """One-hot neighbour labels, smoothed off the observed class."""
        onehot = torch.zeros(*ids.shape, self.classes, device=ids.device)
        onehot.scatter_(-1, ids.unsqueeze(-1), 1.0)
        if self.smoothing > 0.0:
            onehot = onehot * (1.0 - self.smoothing) + self.smoothing / self.classes
        return onehot

    def forward(self, features, neighbour_ids, neighbour_features, distances,
                *, no_neighbours=False):
        """features: [n, F]; neighbour_ids: [n, k]; neighbour_features: [n, k, F]."""
        if features.ndim != 2 or neighbour_ids.ndim != 2:
            raise ValueError("features [n, F] and neighbour_ids [n, k] required")
        if neighbour_features.shape[:2] != neighbour_ids.shape:
            raise ValueError("neighbour_features must align with neighbour_ids")
        if distances.shape != neighbour_ids.shape:
            raise ValueError("one distance per neighbour required")
        own = self.encode(features)
        logits = self.own_vote(own)
        if no_neighbours:
            # Ablation: the patient's own evidence only. Used to measure exactly
            # what propagation contributes, against the same trained weights.
            return logits

        n, k = neighbour_ids.shape
        nb = self.encode(neighbour_features.reshape(n * k, -1)).view(n, k, -1)
        weight = self.attention(own, nb, distances)               # [n, heads, k]

        labels = self.label_matrix(self.neighbour_labels[neighbour_ids])  # [n,k,C]
        # Per head: attention-weighted neighbour label distribution, then a
        # learned per-head gain onto the classes.
        label_mix = torch.einsum("nhk,nkc->nhc", weight, labels)
        label_logits = (label_mix * self.label_gain).sum(1)

        feature_mix = torch.einsum("nhk,nkd->nhd", weight, nb).reshape(n, -1)
        feature_logits = self.neighbour_vote(feature_mix.view(n, self.heads, -1).mean(1))

        return logits + self.neighbour_scale * self.drop(label_logits + feature_logits)
