"""Explicit medication–complaint content residuals before PyG PNA aggregation.

Types are immutable graph metadata: medication=0, complaint=1, hub=2.
No concept–concept edges or patient identifiers are introduced.
"""
import torch
from torch import nn
from torch_geometric.nn import PNAConv


class PairContent(nn.Module):
    """Low-rank bilinear content, averaged over real same-hub opposite-type edges."""
    def __init__(self, width, rank):
        super().__init__()
        self.med = nn.Linear(width, rank, bias=False)
        self.complaint = nn.Linear(width, rank, bias=False)
        self.to_med = nn.Linear(rank, width, bias=False)
        self.to_complaint = nn.Linear(rank, width, bias=False)

    def forward(self, x, edge_index, node_type, gate):
        src, dst = edge_index
        delta = x.new_zeros((src.numel(), x.size(-1)))
        incoming = node_type[dst] == 2
        for hub in dst[incoming].unique():
            med = ((dst == hub) & incoming & (node_type[src] == 0)).nonzero().flatten()
            cc = ((dst == hub) & incoming & (node_type[src] == 1)).nonzero().flatten()
            if med.numel() == 0 or cc.numel() == 0:
                continue
            pair = torch.tanh(self.med(x[src[med]])[:, None, :] *
                              self.complaint(x[src[cc]])[None, :, :])
            # Fixed original neighbor counts: zero masks must not renormalize away.
            med_context = (pair * gate[cc][None, :, None]).mean(1)
            cc_context = (pair * gate[med][:, None, None]).mean(0)
            delta = delta.index_add(0, med, self.to_med(med_context))
            delta = delta.index_add(0, cc, self.to_complaint(cc_context))
        return delta


class InteractionPNAConv(PNAConv):
    """Standard single-tower PNA with an optional incoming content residual."""
    def __init__(self, width, degree_histogram, rank=16, interactions=True):
        if width < 1 or rank < 1:
            raise ValueError('width and rank must be positive')
        super().__init__(width, width, ['mean', 'min', 'max', 'std'],
                         ['identity', 'amplification', 'attenuation'],
                         degree_histogram.detach().clone(), towers=1,
                         pre_layers=1, post_layers=1, train_norm=False)
        # A genuinely all-edgeless training set has avg log degree zero.
        # Keep its histogram exact; use only a numerical denominator floor.
        self.aggr_module.init_avg_deg_log = max(self.aggr_module.init_avg_deg_log, 1e-6)
        self.aggr_module.avg_deg_log.clamp_(min=1e-6)
        self.pair = PairContent(width, rank) if interactions else None

    def forward(self, x, edge_index, *, node_type, cross_pairs=True, edge_mask=None):
        # Consume the exact real-edge PyG mask BEFORE constructing pair context.
        # explain_message below is identity: applying it again would square masks.
        if self.explain and self._edge_mask is not None:
            if edge_mask is not None:
                raise ValueError('Use either explicit edge_mask or PyG mask, not both')
            edge_mask = self._edge_mask.sigmoid() if self._apply_sigmoid else self._edge_mask
        gate = x.new_ones(edge_index.size(1)) if edge_mask is None else edge_mask
        if gate.ndim != 1 or gate.numel() != edge_index.size(1):
            raise ValueError('One mask value per real directed edge required')
        self._gate = gate
        self._content = (self.pair(x, edge_index, node_type, gate)
                         if self.pair is not None and cross_pairs else None)
        try:
            return super().forward(x, edge_index)
        finally:
            self._content = None
            self._gate = None

    def message(self, x_i, x_j, edge_attr):
        standard = super().message(x_i, x_j, edge_attr)
        content = standard if self._content is None else standard + self._content[:, None, :]
        return content * self._gate[:, None, None]

    def explain_message(self, inputs, dim_size):
        """Mask already applied to context and messages; no automatic second mask."""
        return inputs


class PNAPredictor(nn.Module):
    """Residual PNA layers and hub-only readout (no unmasked concept pool)."""
    def __init__(self, input_dim, classes, degree_histogram, width=64, layers=2,
                 rank=16, interactions=True):
        super().__init__()
        if layers < 1:
            raise ValueError('At least one message-passing layer required')
        self.encoder = nn.Linear(input_dim, width, bias=False)
        self.convs = nn.ModuleList([InteractionPNAConv(width, degree_histogram, rank, interactions)
                                    for _ in range(layers)])
        self.head = nn.Linear(width, classes)

    def forward(self, data, *, cross_pairs=True, edge_mask=None, no_messages=False):
        x, edges, types = data.x, data.edge_index, data.node_type
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if types.shape != (x.size(0),) or not torch.all((types >= 0) & (types <= 2)):
            raise ValueError('Stable node_type metadata required')
        if not torch.equal(batch[edges[0]], batch[edges[1]]):
            raise ValueError('cross-graph edges are forbidden')
        hub = (types == 2).nonzero().flatten()
        order = batch[hub].argsort()
        hub = hub[order]
        if not torch.equal(batch[hub], torch.arange(int(batch.max()) + 1, device=x.device)):
            raise ValueError('Exactly one hub per graph required')
        if no_messages:
            if edge_mask is not None:
                raise ValueError('no_messages and edge_mask are mutually exclusive')
            edge_mask = x.new_zeros(edges.size(1))
        hidden = self.encoder(x)
        for conv in self.convs:
            hidden = hidden + torch.relu(conv(hidden, edges, node_type=types,
                                               cross_pairs=cross_pairs, edge_mask=edge_mask))
        return self.head(hidden[hub])


def capacity_report(input_dim, classes, width=64, layers=2, rank=16):
    """Nearest integer wider plain control; analytic search, actual parameter counts.

    Two-percent tolerance is a declared comparison flag, not exact capacity parity.
    Count probes preserve the caller's RNG; no validation/test metrics are consulted.
    """
    def plain_count(w):
        return input_dim * w + layers * (16 * w * w + 3 * w) + classes * w + classes
    target = plain_count(width) + layers * 4 * width * rank
    wide = min(range(width + 1, width + 4 * rank + 2), key=lambda w: abs(plain_count(w) - target))
    report = {}
    with torch.random.fork_rng(devices=[]):
        for name, w, pairs in [('plain', width, False), ('interaction', width, True), ('plain_wide', wide, False)]:
            model = PNAPredictor(input_dim, classes, torch.tensor([1, 8, 2]), w, layers, rank, pairs)
            report[name] = {'width': w, 'parameters': sum(p.numel() for p in model.parameters())}
    report['relative_gap'] = abs(report['plain_wide']['parameters'] - report['interaction']['parameters']) / report['interaction']['parameters']
    report['tolerance'] = .02
    report['within_tolerance'] = report['relative_gap'] <= report['tolerance']
    return report


class GraphXAIWrapper(nn.Module):
    """Bind original node order/types while explainers perturb continuous features.

    Create a new wrapper for each graph/node order. Edge subsets are supported;
    feature-masking keeps nodes and immutable metadata in their original order.
    """
    def __init__(self, model, node_type):
        super().__init__()
        self.model = model
        self.register_buffer('node_type', node_type.detach().clone())

    def forward(self, x, edge_index, batch=None):
        from torch_geometric.data import Data
        return self.model(Data(x=x, edge_index=edge_index, batch=batch, node_type=self.node_type))
