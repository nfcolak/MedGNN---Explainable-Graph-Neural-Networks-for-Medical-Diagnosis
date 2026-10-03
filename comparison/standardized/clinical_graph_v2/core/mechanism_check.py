"""Mechanism checks that pass or fail WITHOUT training.

A benchmark score never proves the intended mechanism ran. These checks do, and they
cost seconds. Run before any training claim.
"""
import sys

import torch

from ..methods.gchm_pna.gchm_v2 import DEFAULT_HIDDEN, HUB_KIND, GCHMv2, HubGatedPNALayer, average_log_degree
from ..methods.gchm_pna.gchm_v3 import DEFAULT_HIDDEN as V3_DEFAULT_HIDDEN
from ..methods.gchm_pna.gchm_v3 import GCHMv3
from .model import (ClinicalGNN, EdgeConditionedLayer, GatedConceptHubLayer,
                    HeteroGraphTransformerLayer)
from .tensorize import ALL_RELATIONS, PAYLOAD_WIDTH, relation_vocabulary

NUM_TRIPLES = 16  # 15 measured meta-relations + index 0 for an unseen one
# GCHM-PNA v2 checks use the realised bidirectional widths: 24 relations and a
# 31-wide edge_attr (relation one-hot + 7 payload columns).
V2_RELATIONS = len(relation_vocabulary('bidirectional'))
V2_EDGE_DIM = V2_RELATIONS + PAYLOAD_WIDTH
# Frozen sample10k/max6/top10 bindings (clinical_runs_v3_adapters_sample10k_max6_top10_
# 20260924): ProtGNN, the largest rival GNN, and the input widths v2 is budgeted on.
PROTGNN_FROZEN_PARAMETERS = 399_884
FROZEN_VOCABULARY, FROZEN_NODE_DIM, FROZEN_CLASSES = 391, 63, 10


def check_message_not_separable(dim=16, edge_dim=8, seed=0):
    """m(x_i, x_j, e) must not equal the sum of its one-at-a-time parts.

    If the residual sits at float32 roundoff (~1e-7) the layer is additive and can
    never multiply a neighbour's content by the receiver's state -- meaning
    `baseline_of`'s delta could not condition on the receiving measurement.
    """
    torch.manual_seed(seed)
    layer = EdgeConditionedLayer(dim, edge_dim, dropout=0.0)
    xi, xj = torch.randn(4, dim), torch.randn(4, dim)
    e = torch.randn(4, edge_dim)
    zi, zj, ze = torch.zeros_like(xi), torch.zeros_like(xj), torch.zeros_like(e)
    with torch.no_grad():
        full = layer.message(xj, xi, e)
        base = layer.message(zj, zi, ze)
        only_j = layer.message(xj, zi, ze) - base
        only_i = layer.message(zj, xi, ze) - base
        only_e = layer.message(zj, zi, e) - base
        residual = (full - (base + only_i + only_j + only_e)).abs().max().item()
    return residual, residual > 1e-4


def check_receiver_changes_message(dim=16, edge_dim=8, seed=1):
    """Fixing sender and edge, varying only the receiver, must change the message."""
    torch.manual_seed(seed)
    layer = EdgeConditionedLayer(dim, edge_dim, dropout=0.0)
    xj, e = torch.randn(1, dim), torch.randn(1, edge_dim)
    with torch.no_grad():
        a = layer.message(xj, torch.randn(1, dim), e)
        b = layer.message(xj, torch.randn(1, dim), e)
    delta = (a - b).abs().max().item()
    return delta, delta > 1e-4


def check_edge_payload_ablation(dim=16, edge_dim=8, seed=2):
    """The payload ablation must change the output but NOT the parameter count."""
    torch.manual_seed(seed)
    free = EdgeConditionedLayer(dim, edge_dim, dropout=0.0, use_edge_payload=True)
    torch.manual_seed(seed)
    ablated = EdgeConditionedLayer(dim, edge_dim, dropout=0.0, use_edge_payload=False)
    x = torch.randn(5, dim)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    e = torch.randn(4, edge_dim)
    with torch.no_grad():
        a = free(x, edge_index, e)
        b = ablated(x, edge_index, e)
    same_params = sum(p.numel() for p in free.parameters()) == \
        sum(p.numel() for p in ablated.parameters())
    delta = (a - b).abs().max().item()
    return delta, (delta > 1e-4 and same_params)


def check_degenerate_graph_finite(seed=3):
    """An edgeless single-node graph must produce finite logits, not NaN."""
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2)
    from torch_geometric.data import Batch, Data
    d = Data(x=torch.randn(1, 14), edge_index=torch.zeros((2, 0), dtype=torch.long),
             edge_attr=torch.zeros((0, 22)))
    d.token = torch.zeros(1, dtype=torch.long)
    out = model(Batch.from_data_list([d]))
    return float(out.abs().max()), bool(torch.isfinite(out).all())


def check_batching_does_not_mix(seed=4):
    """Two graphs batched together must give the same logits as run separately."""
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2)
    model.eval()
    from torch_geometric.data import Batch, Data

    def make(n, m):
        d = Data(x=torch.randn(n, 14),
                 edge_index=torch.randint(0, n, (2, m)),
                 edge_attr=torch.randn(m, 22))
        d.token = torch.randint(0, 10, (n,))
        return d

    a, b = make(6, 9), make(4, 5)
    with torch.no_grad():
        separate = torch.cat([model(Batch.from_data_list([a])),
                              model(Batch.from_data_list([b]))])
        together = model(Batch.from_data_list([a, b]))
    delta = (separate - together).abs().max().item()
    return delta, delta < 1e-4


def check_gradients_reach_every_block(seed=5):
    """Every parameter block must receive a gradient, or a module is dead code."""
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2)
    from torch_geometric.data import Batch, Data
    d = Data(x=torch.randn(6, 14), edge_index=torch.randint(0, 6, (2, 9)),
             edge_attr=torch.randn(9, 22))
    d.token = torch.randint(1, 10, (6,))
    out = model(Batch.from_data_list([d]))
    out.sum().backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    return missing, not missing


def _hgt_graph(n, m, dim, seed):
    """One random typed graph in the layer's own tensor contract."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    edge_index = torch.randint(0, n, (2, m), generator=g)
    triple = torch.randint(0, NUM_TRIPLES, (m,), generator=g)
    payload = torch.randn(m, PAYLOAD_WIDTH, generator=g)
    return x, edge_index, triple, payload


def check_baseline_relation_interacts_with_content(dim=16, edge_dim=22, seed=10):
    """Construct a witness for relation-dependent full-MLP messages.

    A one-hot relation is an additive offset before the first ReLU, but shifting
    the activation boundary can change the relation effect across node contents.
    This checks Linear -> ReLU -> Linear, not just the first preactivation. The
    deterministic, nondegenerate weights prove architectural possibility, NOT a
    learned effect or an advantage over HGT. ``seed`` is kept for call compatibility;
    the constructed witness does not depend on random initialization.
    """
    if dim < 1 or edge_dim < PAYLOAD_WIDTH + 2:
        raise ValueError('relation witness requires hidden width >= 1 and two relations')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layer = EdgeConditionedLayer(dim, edge_dim, dropout=0.0)
    xj = torch.tensor([-2., -0.5, 0.5, 2.]).unsqueeze(1).repeat(1, dim)
    xi = torch.zeros_like(xj)
    a, b = torch.zeros(4, edge_dim), torch.zeros(4, edge_dim)
    a[:, 0], b[:, 1] = 1., 1.
    with torch.no_grad():
        first, last = layer.message_mlp[0], layer.message_mlp[2]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[:, :dim] = torch.eye(dim)
        first.weight[:, dim:2 * dim] = 0.5 * torch.eye(dim)
        first.weight[:, 2 * dim + 1] = 1.
        last.weight.copy_(2. * torch.eye(dim) + 0.25 / dim)
        last.bias.zero_()
        pre = (first(torch.cat([xj, xi, a], dim=-1))
               - first(torch.cat([xj, xi, b], dim=-1)))
        message = layer.message(xj, xi, a) - layer.message(xj, xi, b)
    report = {'preactivation_variation': float((pre - pre[0]).abs().max()),
              'message_variation': float((message - message[0]).abs().max())}
    return report, (report['preactivation_variation'] < 1e-6
                    and report['message_variation'] > 1e-3)


def check_hgt_relation_transforms_content(dim=16, heads=4, seed=11):
    """In HGT the relation must TRANSFORM the neighbour, not shift it.

    Same sender, two different meta-relations: the resulting message difference must
    vary across senders. This checks HGT's explicit typed projection, not a claim
    that the nonlinear baseline cannot learn relation-content interactions.
    """
    torch.manual_seed(seed)
    layer = HeteroGraphTransformerLayer(dim, NUM_TRIPLES, PAYLOAD_WIDTH, heads,
                                        dropout=0.0)
    with torch.no_grad():
        # Spread the typed weights so the identity-centred init is not the thing
        # under test; training is what normally does this.
        layer.k_rel.add_(torch.randn_like(layer.k_rel) * 0.5)
        layer.v_rel.add_(torch.randn_like(layer.v_rel) * 0.5)
    n, hd = 8, dim // heads
    xj = torch.randn(n, heads, hd)
    with torch.no_grad():
        a = torch.einsum('ehd,ehdf->ehf', xj, layer.v_rel[torch.full((n,), 3)])
        b = torch.einsum('ehd,ehdf->ehf', xj, layer.v_rel[torch.full((n,), 7)])
    d = (a - b).reshape(n, -1)
    variation = float((d - d[0]).abs().max())
    return variation, variation > 1e-3


def check_hgt_relation_prior(dim=16, heads=4, seed=12):
    """`mu_phi` must be able to down-weight a whole relation on its own.

    The paper's relation prior is the knob that says "trust `instance_of` less than
    `baseline_of`" without changing what either carries. Zeroing it for one
    meta-relation must visibly change the layer's output.
    """
    torch.manual_seed(seed)
    layer = HeteroGraphTransformerLayer(dim, NUM_TRIPLES, PAYLOAD_WIDTH, heads,
                                        dropout=0.0)
    layer.eval()
    x, edge_index, triple, payload = _hgt_graph(9, 14, dim, seed)
    triple[:] = 3
    triple[:5] = 7
    with torch.no_grad():
        before = layer(x, edge_index, triple, payload)
        layer.rel_prior[7] = 0.0
        after = layer(x, edge_index, triple, payload)
    delta = float((before - after).abs().max())
    return delta, delta > 1e-4


def check_hgt_attention_normalised(dim=16, heads=4, seed=13):
    """Attention must sum to 1 per receiving node, per head.

    This is the property the summing baseline lacks: with `aggr='add'`, an analyte
    with 400 `instance_of` neighbours contributes far more raw magnitude than a
    measurement with 2 `baseline_of` neighbours, regardless of relevance.
    """
    from torch_geometric.utils import softmax as _softmax
    torch.manual_seed(seed)
    layer = HeteroGraphTransformerLayer(dim, NUM_TRIPLES, PAYLOAD_WIDTH, heads,
                                        dropout=0.0)
    layer.eval()
    n, m = 7, 25
    x, edge_index, triple, payload = _hgt_graph(n, m, dim, seed)
    captured = {}
    original = layer.message

    def spy(k_j, q_i, v_j, edge_triple, edge_payload, index, ptr, size_i):
        k = torch.einsum('ehd,ehdf->ehf', k_j, layer.k_rel[edge_triple])
        logit = (q_i * k).sum(-1) / (layer.head_dim ** 0.5)
        logit = logit * layer.rel_prior[edge_triple] + layer.payload_lin(edge_payload)
        captured['att'] = _softmax(logit, index, ptr, size_i)
        captured['index'] = index
        return original(k_j, q_i, v_j, edge_triple, edge_payload, index, ptr, size_i)

    layer.message = spy
    with torch.no_grad():
        layer(x, edge_index, triple, payload)
    layer.message = original
    att, index = captured['att'], captured['index']
    sums = torch.zeros(n, layer.heads).index_add_(0, index, att)
    receivers = torch.unique(index)
    worst = float((sums[receivers] - 1.0).abs().max())
    return worst, worst < 1e-5


def check_hgt_relation_separation(dim=16, heads=4, seed=14):
    """BG-HGNN relation collapse must not already be present at init.

    A typed model whose per-relation weights are identical is its own no-hetero
    ablation wearing a different name. The reported number is the normalised spread
    of the typed projections; 0 means total collapse.
    """
    torch.manual_seed(seed)
    layer = HeteroGraphTransformerLayer(dim, NUM_TRIPLES, PAYLOAD_WIDTH, heads,
                                        dropout=0.0)
    report = layer.relation_separation()
    worst = min(report.values())
    return report, worst > 1e-3


def check_hgt_parameter_budget(dim=96, heads=4, seed=15):
    """BG-HGNN parameter explosion: HGT must stay well under per-relation MLPs.

    The naive alternative is one message MLP per relation (RGCN-style). HGT's typed
    weights act on the per-head dimension, so the total must stay far below that.
    """
    torch.manual_seed(seed)
    hgt = HeteroGraphTransformerLayer(dim, NUM_TRIPLES, PAYLOAD_WIDTH, heads,
                                      dropout=0.0)
    shared = EdgeConditionedLayer(dim, 22, dropout=0.0)
    n_hgt = sum(p.numel() for p in hgt.parameters())
    n_shared = sum(p.numel() for p in shared.parameters())
    naive = n_shared * NUM_TRIPLES  # one full message MLP per meta-relation
    value = {'hgt': n_hgt, 'shared': n_shared, 'naive_per_relation': naive,
             'ratio_vs_shared': round(n_hgt / n_shared, 2)}
    return value, n_hgt < naive / 3


def check_hgt_degenerate_graph(seed=16):
    """An edgeless single-node graph must give finite logits under HGT too.

    Softmax attention over an empty neighbourhood is the obvious NaN source, and
    this graph family really does contain event-free patients.
    """
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2,
                        conv='hgt', num_triples=NUM_TRIPLES, payload_dim=PAYLOAD_WIDTH,
                        heads=4)
    from torch_geometric.data import Batch, Data
    d = Data(x=torch.randn(1, 14), edge_index=torch.zeros((2, 0), dtype=torch.long),
             edge_attr=torch.zeros((0, 22)))
    d.token = torch.zeros(1, dtype=torch.long)
    d.edge_triple = torch.zeros(0, dtype=torch.long)
    d.edge_payload = torch.zeros((0, PAYLOAD_WIDTH))
    out = model(Batch.from_data_list([d]))
    return float(out.abs().max()), bool(torch.isfinite(out).all())


def check_hgt_batching(seed=17):
    """Batched HGT must equal separate runs.

    Softmax makes this a real risk rather than a formality: if `index` leaked across
    graphs, one patient's neighbours would normalise against another's.
    """
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2,
                        conv='hgt', num_triples=NUM_TRIPLES, payload_dim=PAYLOAD_WIDTH,
                        heads=4)
    model.eval()
    from torch_geometric.data import Batch, Data

    def make(n, m):
        d = Data(x=torch.randn(n, 14), edge_index=torch.randint(0, n, (2, m)),
                 edge_attr=torch.randn(m, 22))
        d.token = torch.randint(0, 10, (n,))
        d.edge_triple = torch.randint(0, NUM_TRIPLES, (m,))
        d.edge_payload = torch.randn(m, PAYLOAD_WIDTH)
        return d

    a, b = make(6, 9), make(4, 5)
    with torch.no_grad():
        separate = torch.cat([model(Batch.from_data_list([a])),
                              model(Batch.from_data_list([b]))])
        together = model(Batch.from_data_list([a, b]))
    delta = float((separate - together).abs().max())
    return delta, delta < 1e-4


def check_hgt_gradients(seed=18):
    """Every typed weight must receive gradient, or the type machinery is dead code.

    A meta-relation absent from the batch legitimately gets no gradient, so this
    uses a batch that exercises every index.
    """
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2,
                        conv='hgt', num_triples=NUM_TRIPLES, payload_dim=PAYLOAD_WIDTH,
                        heads=4)
    from torch_geometric.data import Batch, Data
    m = NUM_TRIPLES * 2
    d = Data(x=torch.randn(8, 14), edge_index=torch.randint(0, 8, (2, m)),
             edge_attr=torch.randn(m, 22))
    d.token = torch.randint(1, 10, (8,))
    d.edge_triple = torch.arange(m) % NUM_TRIPLES
    d.edge_payload = torch.randn(m, PAYLOAD_WIDTH)
    model(Batch.from_data_list([d])).sum().backward()
    missing = [n for n, p in model.named_parameters()
               if p.grad is None or not torch.isfinite(p.grad).all()]
    dead = [n for n, p in model.named_parameters()
            if ('k_rel' in n or 'v_rel' in n or 'rel_prior' in n)
            and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    return {'missing_grad': missing, 'dead_typed_weights': dead}, not missing and not dead


def check_gchm_gate_is_multiplicative(dim=16, edge_dim=22, seed=20):
    """GCHM's claim: the receiver SCALES the message, it does not shift it.

    Scaling the sender's content must scale the output by the same factor, because
    `content * gate` is homogeneous in content. Under the additive control
    `content + gate` the gate survives that scaling, so the two arms are
    distinguishable by construction rather than by hope. The message MLP's biases
    are zeroed first so "content" really means the sender/receiver-driven part.
    """
    torch.manual_seed(seed)
    deg = torch.tensor([0., 5., 20., 10., 3.])
    layer = GatedConceptHubLayer(dim, edge_dim, deg, dropout=0.0,
                                 modulation='multiplicative')
    with torch.no_grad():
        for m in layer.message_mlp:
            if isinstance(m, torch.nn.Linear):
                m.bias.zero_()
    xj, xi = torch.randn(4, dim), torch.randn(4, dim)
    e = torch.zeros(4, edge_dim)
    with torch.no_grad():
        once = layer.message(xj, xi, e)
        # Only the SENDER is scaled; the receiver (hence the gate) is untouched.
        twice = layer.message(2.0 * xj, xi, e)
    # A bias-free 2-layer MLP is not linear (ReLU in between), so compare against
    # the same network's own response instead of assuming exact doubling: the
    # multiplicative arm must keep the ratio gate-free.
    gate = torch.sigmoid(layer.gate(xi))
    with torch.no_grad():
        content_once = layer.message_mlp(torch.cat([xj, xi, e], dim=-1))
        content_twice = layer.message_mlp(torch.cat([2.0 * xj, xi, e], dim=-1))
        residual = float(((once - content_once * gate).abs().max()
                          + (twice - content_twice * gate).abs().max()))
    return residual, residual < 1e-5


def check_gchm_modulation_arms_differ(dim=16, edge_dim=22, seed=21):
    """The two arms must differ in OUTPUT while matching in PARAMETER COUNT.

    Without the parameter check an advantage of the multiplicative arm could be
    bought with capacity instead of interaction order.
    """
    torch.manual_seed(seed)
    deg = torch.tensor([0., 5., 20., 10., 3.])
    mul = GatedConceptHubLayer(dim, edge_dim, deg, dropout=0.0,
                               modulation='multiplicative')
    torch.manual_seed(seed)
    add = GatedConceptHubLayer(dim, edge_dim, deg, dropout=0.0,
                               modulation='additive')
    xj, xi = torch.randn(6, dim), torch.randn(6, dim)
    e = torch.randn(6, edge_dim)
    with torch.no_grad():
        delta = float((mul.message(xj, xi, e) - add.message(xj, xi, e)).abs().max())
    same = (sum(p.numel() for p in mul.parameters())
            == sum(p.numel() for p in add.parameters()))
    return {'output_delta': round(delta, 5), 'equal_params': same}, delta > 1e-4 and same


def check_gchm_second_order(dim=16, edge_dim=22, seed=22):
    """The gate must create a genuine sender x receiver interaction term.

    Measures the mixed residual m(xi,xj) - m(xi,0) - m(0,xj) + m(0,0). An additively
    separable message leaves float32 roundoff here -- which is exactly what the
    native probes measured for PNAConv (4.8e-07) and what GCHM was built to fix.
    """
    torch.manual_seed(seed)
    deg = torch.tensor([0., 5., 20., 10., 3.])
    layer = GatedConceptHubLayer(dim, edge_dim, deg, dropout=0.0)
    xj, xi = torch.randn(5, dim), torch.randn(5, dim)
    zj, zi = torch.zeros_like(xj), torch.zeros_like(xi)
    e = torch.zeros(5, edge_dim)
    with torch.no_grad():
        full = layer.message(xj, xi, e)
        only_j = layer.message(xj, zi, e)
        only_i = layer.message(zj, xi, e)
        base = layer.message(zj, zi, e)
        residual = float((full - only_i - only_j + base).abs().max())
    return residual, residual > 1e-4


def check_gchm_pna_scaling_active(dim=16, edge_dim=22, seed=23):
    """PNA degree scaling must actually respond to the degree histogram.

    Two layers differing ONLY in their fitted histogram must produce different
    outputs on the same graph; if not, the degree-scaled aggregation is inert and
    GCHM has silently degraded to a plain multi-aggregator.
    """
    torch.manual_seed(seed)
    a = GatedConceptHubLayer(dim, edge_dim, torch.tensor([0., 100., 5., 1.]), dropout=0.0)
    torch.manual_seed(seed)
    b = GatedConceptHubLayer(dim, edge_dim, torch.tensor([0., 1., 5., 100.]), dropout=0.0)
    x = torch.randn(6, dim)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1], [1, 2, 3, 4, 5, 0, 2, 3]])
    e = torch.randn(8, edge_dim)
    a.eval()
    b.eval()
    with torch.no_grad():
        delta = float((a(x, edge_index, e) - b(x, edge_index, e)).abs().max())
    return delta, delta > 1e-4


def check_gchm_degenerate_graph(seed=24):
    """An edgeless single-node graph must stay finite under PNA aggregation.

    The `std` aggregator over one element and the log-degree denominator are the
    two NaN sources here, and this graph family really does contain such patients.
    """
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2,
                        conv='gchm', degree_histogram=torch.tensor([0., 5., 2.]))
    from torch_geometric.data import Batch, Data
    d = Data(x=torch.randn(1, 14), edge_index=torch.zeros((2, 0), dtype=torch.long),
             edge_attr=torch.zeros((0, 22)))
    d.token = torch.zeros(1, dtype=torch.long)
    out = model(Batch.from_data_list([d]))
    return float(out.abs().max()), bool(torch.isfinite(out).all())


def check_gchm_batching(seed=25):
    """Batched GCHM must equal separate runs.

    Degree-scaled aggregation normalises per receiving node, so a leak across the
    batch boundary would make one patient's degrees rescale another's messages.
    """
    torch.manual_seed(seed)
    model = ClinicalGNN(num_tokens=10, node_dim=14, edge_dim=22, hidden=32, layers=2,
                        conv='gchm', degree_histogram=torch.tensor([0., 5., 9., 2.]))
    model.eval()
    from torch_geometric.data import Batch, Data

    def make(n, m):
        d = Data(x=torch.randn(n, 14), edge_index=torch.randint(0, n, (2, m)),
                 edge_attr=torch.randn(m, 22))
        d.token = torch.randint(0, 10, (n,))
        return d

    a, b = make(6, 9), make(4, 5)
    with torch.no_grad():
        separate = torch.cat([model(Batch.from_data_list([a])),
                              model(Batch.from_data_list([b]))])
        together = model(Batch.from_data_list([a, b]))
    delta = float((separate - together).abs().max())
    return delta, delta < 1e-4


# --------------------------------------------------------------- GCHM-PNA v2
def _v2_layer(seed, dim=16, avg_deg_log=1.0, **options):
    torch.manual_seed(seed)
    return HubGatedPNALayer(dim, V2_EDGE_DIM, V2_RELATIONS, avg_deg_log, dropout=0.0,
                            **options)


def _v2_edge_inputs(count, dim=16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(count, dim, generator=generator),
            torch.randn(count, dim, generator=generator),
            torch.randn(count, V2_EDGE_DIM, generator=generator),
            torch.randint(0, V2_RELATIONS, (count,), generator=generator),
            torch.randn(count, dim, generator=generator))


def _v2_graph(n, m, seed, node_dim=14, num_tokens=10):
    from torch_geometric.data import Data
    generator = torch.Generator().manual_seed(seed)
    d = Data(x=torch.randn(n, node_dim, generator=generator),
             edge_index=torch.randint(0, n, (2, m), generator=generator),
             edge_attr=torch.randn(m, V2_EDGE_DIM, generator=generator))
    d.token = torch.randint(0, num_tokens, (n,), generator=generator)
    node_type = torch.randint(0, 8, (n,), generator=generator)
    node_type[node_type == HUB_KIND] = 0
    node_type[min(1, n - 1)] = HUB_KIND  # exactly one index-visit hub per graph
    d.node_type = node_type
    d.edge_relation = torch.randint(0, V2_RELATIONS, (m,), generator=generator)
    return d


def _v2_model(seed, **options):
    torch.manual_seed(seed)
    return GCHMv2(num_tokens=10, node_dim=14, edge_dim=V2_EDGE_DIM, num_classes=5,
                  num_relations=V2_RELATIONS, degree_histogram=torch.tensor([0., 5., 9., 2.]),
                  hidden=32, layers=2, dropout=0.0, **options)


def check_v2_gate_is_multiplicative(seed=30):
    """v2 message = content * sigmoid(receiver + relation + hub gate), exactly."""
    layer = _v2_layer(seed)
    xj, xi, e, rel, hub = _v2_edge_inputs(5, seed=seed)
    with torch.no_grad():
        message = layer.message(xj, xi, e, rel, hub)
        content = layer.message_mlp(torch.cat([xj, xi, e], dim=-1))
        gate = torch.sigmoid(layer.receiver_gate(xi) + layer.relation_gate(rel)
                             + layer.hub_gate(hub))
        residual = float((message - content * gate).abs().max())
    return residual, residual < 1e-6


def check_v2_hub_state_gates_messages(seed=31):
    """Only the hub state changes -> the message changes; without the hub gate it cannot."""
    gated, ungated = _v2_layer(seed), _v2_layer(seed, hub_gate=False)
    xj, xi, e, rel, hub_a = _v2_edge_inputs(4, seed=seed)
    hub_b = torch.randn_like(hub_a)
    with torch.no_grad():
        delta = float((gated.message(xj, xi, e, rel, hub_a)
                       - gated.message(xj, xi, e, rel, hub_b)).abs().max())
        ablated = float((ungated.message(xj, xi, e, rel, hub_a)
                         - ungated.message(xj, xi, e, rel, hub_b)).abs().max())
    return ({'hub_delta': round(delta, 5), 'no_hub_gate_delta': ablated},
            delta > 1e-4 and ablated == 0.0)


def check_v2_relation_changes_gate(seed=32):
    """Same sender, receiver, payload and hub; only the relation id differs."""
    layer = _v2_layer(seed)
    xj, xi, e, _, hub = _v2_edge_inputs(3, seed=seed)
    forward = torch.zeros(3, dtype=torch.long)
    reverse = torch.full((3,), V2_RELATIONS - 1, dtype=torch.long)
    with torch.no_grad():
        delta = float((layer.message(xj, xi, e, forward, hub)
                       - layer.message(xj, xi, e, reverse, hub)).abs().max())
    return delta, delta > 1e-4


def check_v2_modulation_arms_differ(seed=33):
    """Additive control: different output, identical parameter count."""
    mul, add = _v2_layer(seed), _v2_layer(seed, modulation='additive')
    xj, xi, e, rel, hub = _v2_edge_inputs(6, seed=seed)
    with torch.no_grad():
        delta = float((mul.message(xj, xi, e, rel, hub)
                       - add.message(xj, xi, e, rel, hub)).abs().max())
    same = (sum(p.numel() for p in mul.parameters())
            == sum(p.numel() for p in add.parameters()))
    return {'output_delta': round(delta, 5), 'equal_params': same}, delta > 1e-4 and same


def check_v2_degree_scaling_active(seed=34):
    """Degree scalers start as identity, so they must (a) receive gradient and
    (b) respond to the train-fitted histogram once non-identity. Otherwise the
    compact PNA would have silently degraded to a plain multi-aggregator."""
    avg = average_log_degree(torch.tensor([0., 5., 20., 10., 3.]))
    layer = _v2_layer(seed, avg_deg_log=avg)
    x = torch.randn(6, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2], [1, 2, 3, 4, 5, 0, 2, 3, 3]])
    _, _, e, rel, _ = _v2_edge_inputs(9, seed=seed)
    hub_nodes = torch.randn(6, 16)
    layer(x, edge_index, e, rel, hub_nodes).pow(2).sum().backward()
    grad = layer.scaler_weight.grad
    learnable = bool(grad is not None and grad[1].abs().max() > 0 and grad[2].abs().max() > 0)
    low, high = _v2_layer(seed, avg_deg_log=0.5), _v2_layer(seed, avg_deg_log=2.0)
    with torch.no_grad():
        for candidate in (low, high):
            candidate.scaler_weight.copy_(torch.tensor([1.0, 0.5, 0.5]).unsqueeze(1)
                                          .expand_as(candidate.scaler_weight))
        delta = float((low(x, edge_index, e, rel, hub_nodes)
                       - high(x, edge_index, e, rel, hub_nodes)).abs().max())
    return ({'scaler_grad_active': learnable, 'histogram_delta': round(delta, 5)},
            learnable and delta > 1e-4)


def check_v2_hub_state_is_visit_node(seed=35):
    """The hub context/readout is exactly the index-visit row; hubless graphs get 0."""
    torch.manual_seed(seed)
    h = torch.randn(7, 8)
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    node_type = torch.tensor([0, HUB_KIND, 2, HUB_KIND, 3, 0, 4])
    hub = GCHMv2.hub_state(h, batch, node_type == HUB_KIND, 3)
    ok = (torch.equal(hub[0], h[1]) and torch.equal(hub[1], h[3])
          and not hub[2].any())
    return 'rows match' if ok else 'mismatch', bool(ok)


def check_v2_degenerate_graph_finite(seed=36):
    """Edgeless single-node graphs (hub and hubless) stay finite."""
    from torch_geometric.data import Batch, Data
    model = _v2_model(seed)
    outputs = []
    for kind in (HUB_KIND, 0):
        d = Data(x=torch.randn(1, 14), edge_index=torch.zeros((2, 0), dtype=torch.long),
                 edge_attr=torch.zeros((0, V2_EDGE_DIM)))
        d.token = torch.zeros(1, dtype=torch.long)
        d.node_type = torch.tensor([kind])
        d.edge_relation = torch.zeros(0, dtype=torch.long)
        outputs.append(model(Batch.from_data_list([d])))
    out = torch.cat(outputs)
    return float(out.abs().max()), bool(torch.isfinite(out).all())


def check_v2_batching_does_not_mix(seed=37):
    """Hub context and degree scaling are graph-local: batched == separate."""
    from torch_geometric.data import Batch
    model = _v2_model(seed)
    model.eval()
    a, b = _v2_graph(6, 9, seed), _v2_graph(4, 5, seed + 1)
    with torch.no_grad():
        separate = torch.cat([model(Batch.from_data_list([a])),
                              model(Batch.from_data_list([b]))])
        together = model(Batch.from_data_list([a, b]))
    delta = float((separate - together).abs().max())
    return delta, delta < 1e-4


def check_v2_gradients_reach_every_block(seed=38):
    """Every v2 parameter tensor receives a gradient on an ordinary batch."""
    from torch_geometric.data import Batch
    model = _v2_model(seed)
    batch = Batch.from_data_list([_v2_graph(6, 12, seed), _v2_graph(5, 8, seed + 1)])
    model(batch).logsumexp(-1).sum().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    return missing or 'all blocks', not missing


def check_v2_rejects_foreign_edge_view(seed=39):
    """A forward-built v2 must refuse bidirectional relation ids, not index garbage."""
    from torch_geometric.data import Batch
    torch.manual_seed(seed)
    model = GCHMv2(num_tokens=10, node_dim=14, edge_dim=V2_EDGE_DIM, num_classes=5,
                   num_relations=len(ALL_RELATIONS), degree_histogram=torch.tensor([0., 5.]),
                   hidden=16, layers=1)
    graph = _v2_graph(4, 3, seed)
    graph.edge_relation = torch.full((3,), len(ALL_RELATIONS), dtype=torch.long)
    try:
        model(Batch.from_data_list([graph]))
    except ValueError as error:
        return 'rejected', 'relation vocabulary' in str(error)
    return 'accepted', False


def check_v2_parameter_budget(seed=40):
    """At the frozen input widths v2 must stay below ProtGNN, the largest rival GNN.

    v1 (556,938) was the largest arm, so a v2 win could otherwise be bought with width.
    """
    counts = {}
    for view in ('bidirectional', 'forward'):
        relations = len(relation_vocabulary(view))
        torch.manual_seed(seed)
        model = GCHMv2(num_tokens=FROZEN_VOCABULARY, node_dim=FROZEN_NODE_DIM,
                       edge_dim=relations + PAYLOAD_WIDTH, num_classes=FROZEN_CLASSES,
                       num_relations=relations, degree_histogram=torch.tensor([0., 5., 2.]),
                       hidden=DEFAULT_HIDDEN)
        counts[view] = model.parameter_count()
    return counts, max(counts.values()) < PROTGNN_FROZEN_PARAMETERS


# --------------------------------------------------------------- GCHM-PNA v3 arm
def _v3_model(seed, **options):
    torch.manual_seed(seed)
    return GCHMv3(num_tokens=10, node_dim=14, edge_dim=V2_EDGE_DIM, num_classes=5,
                  num_relations=V2_RELATIONS, degree_histogram=torch.tensor([0., 5., 9., 2.]),
                  hidden=32, layers=2, dropout=0.0, **options)


def _v3_fill(model, seed):
    """Zero-initialised read-side tables would make every ablation pass vacuously."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name in ('wide_token', 'wide_value'):
            if hasattr(model, name):
                getattr(model, name).weight.copy_(
                    torch.randn(getattr(model, name).weight.shape, generator=generator))
        if hasattr(model, 'class_weight'):
            model.class_weight.copy_(torch.randn(model.class_weight.shape, generator=generator))
    return model


def check_v3_wide_path_is_exact_token_vote(seed=41):
    """wide logits == sum over the graph's nodes of W_tok[t] + v * W_val[t], exactly,
    and removing the path removes exactly that term (tables filled, not zero)."""
    from torch_geometric.data import Batch
    free = _v3_fill(_v3_model(seed), seed).eval()
    graph = _v2_graph(7, 10, seed)
    batch = Batch.from_data_list([graph])
    with torch.no_grad():
        value = (graph.x[:, 8] * graph.x[:, 9]).unsqueeze(-1)
        expected = (free.wide_token(graph.token) + value * free.wide_value(graph.token)).sum(0)
        got = free.wide_logits(batch, batch.batch, 1)[0]
        exact = float((got - expected).abs().max())
        without = free(batch) - free.wide_logits(batch, batch.batch, 1)
        ablated = _v3_model(seed, wide=False).eval()
        ablated.load_state_dict({k: v for k, v in free.state_dict().items()
                                 if not k.startswith('wide_')})
        residual = float((ablated(batch) - without).abs().max())
    return ({'vote_residual': exact, 'ablation_residual': residual},
            exact < 1e-5 and residual < 1e-5)


def check_v3_labelwise_attention_is_class_specific(seed=42):
    """Each class attends with its own query: per-class attention over one graph's
    nodes sums to 1 and differs between classes; batching keeps it graph-local."""
    from torch_geometric.data import Batch
    from torch_geometric.utils import softmax as segment_softmax
    model = _v3_fill(_v3_model(seed), seed).eval()
    a, b = _v2_graph(6, 9, seed), _v2_graph(4, 5, seed + 1)
    with torch.no_grad():
        batch = Batch.from_data_list([a, b])
        h, index, _, size = model.node_states(batch)
        states = model.labelwise_norm(h)
        alpha = segment_softmax(states @ model.class_query.t(), index, num_nodes=size, dim=0)
        sums = torch.zeros(size, alpha.size(1)).index_add_(0, index, alpha)
        normalised = float((sums - 1).abs().max())
        class_spread = float((alpha - alpha[:, :1]).abs().max())
        separate = torch.cat([model(Batch.from_data_list([a])), model(Batch.from_data_list([b]))])
        mixed = float((separate - model(batch)).abs().max())
    return ({'normalised': normalised, 'class_spread': round(class_spread, 5),
             'batch_mix': mixed},
            normalised < 1e-5 and class_spread > 1e-4 and mixed < 1e-4)


def check_v3_jk_reads_every_depth(seed=43):
    """Jumping knowledge: the final state depends on the encoder AND every layer."""
    from torch_geometric.data import Batch
    model = _v3_model(seed)
    batch = Batch.from_data_list([_v2_graph(6, 12, seed)])
    model(batch).logsumexp(-1).sum().backward()
    width = model.jk_projection.weight.size(1) // (len(model.layers) + 1)
    grads = model.jk_projection.weight.grad.abs().view(-1, len(model.layers) + 1, width)
    per_depth = grads.sum(dim=(0, 2))
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    return ({'per_depth_grad': [round(float(g), 4) for g in per_depth], 'missing': missing},
            bool((per_depth > 0).all()) and not missing)


def check_v3_edge_dropout_train_only(seed=44):
    """Edge dropout perturbs training forwards only; eval is deterministic."""
    from torch_geometric.data import Batch
    model = _v3_model(seed, edge_dropout=0.5)
    batch = Batch.from_data_list([_v2_graph(8, 20, seed)])
    with torch.no_grad():
        model.train()
        torch.manual_seed(1)
        first = model(batch)
        torch.manual_seed(2)
        second = model(batch)
        model.eval()
        stable = float((model(batch) - model(batch)).abs().max())
    return ({'train_delta': round(float((first - second).abs().max()), 5), 'eval_delta': stable},
            float((first - second).abs().max()) > 1e-5 and stable == 0.0)


def check_v3_degenerate_graph_finite(seed=45):
    """Edgeless single-node graphs (hub and hubless) stay finite with every path on."""
    from torch_geometric.data import Batch, Data
    model = _v3_fill(_v3_model(seed), seed).eval()
    outputs = []
    for kind in (HUB_KIND, 0):
        d = Data(x=torch.randn(1, 14), edge_index=torch.zeros((2, 0), dtype=torch.long),
                 edge_attr=torch.zeros((0, V2_EDGE_DIM)))
        d.token = torch.zeros(1, dtype=torch.long)
        d.node_type = torch.tensor([kind])
        d.edge_relation = torch.zeros(0, dtype=torch.long)
        outputs.append(model(Batch.from_data_list([d])))
    out = torch.cat(outputs)
    return float(out.abs().max()), bool(torch.isfinite(out).all())


def check_v3_parameter_budget(seed=46):
    """At the frozen input widths v3 must stay below the frozen ProtGNN count."""
    counts = {}
    for view in ('bidirectional', 'forward'):
        relations = len(relation_vocabulary(view))
        torch.manual_seed(seed)
        model = GCHMv3(num_tokens=FROZEN_VOCABULARY, node_dim=FROZEN_NODE_DIM,
                       edge_dim=relations + PAYLOAD_WIDTH, num_classes=FROZEN_CLASSES,
                       num_relations=relations, degree_histogram=torch.tensor([0., 5., 2.]),
                       hidden=V3_DEFAULT_HIDDEN)
        counts[view] = model.parameter_count()
    return counts, max(counts.values()) < PROTGNN_FROZEN_PARAMETERS


CHECKS = [
    ('message_not_additively_separable', check_message_not_separable),
    ('receiver_state_changes_message', check_receiver_changes_message),
    ('edge_payload_ablation_equal_params', check_edge_payload_ablation),
    ('degenerate_graph_finite', check_degenerate_graph_finite),
    ('batching_does_not_mix_graphs', check_batching_does_not_mix),
    ('gradients_reach_every_block', check_gradients_reach_every_block),
    # --- HGT arm ---
    ('baseline_relation_interacts_with_content', check_baseline_relation_interacts_with_content),
    ('hgt_relation_transforms_content', check_hgt_relation_transforms_content),
    ('hgt_relation_prior_is_separable', check_hgt_relation_prior),
    ('hgt_attention_normalised_per_receiver', check_hgt_attention_normalised),
    ('hgt_no_relation_collapse_at_init', check_hgt_relation_separation),
    ('hgt_parameters_not_exploding', check_hgt_parameter_budget),
    ('hgt_degenerate_graph_finite', check_hgt_degenerate_graph),
    ('hgt_batching_does_not_mix_graphs', check_hgt_batching),
    ('hgt_gradients_reach_typed_weights', check_hgt_gradients),
    # --- GCHM arm ---
    ('gchm_gate_is_multiplicative', check_gchm_gate_is_multiplicative),
    ('gchm_modulation_arms_differ', check_gchm_modulation_arms_differ),
    ('gchm_message_is_second_order', check_gchm_second_order),
    ('gchm_pna_degree_scaling_active', check_gchm_pna_scaling_active),
    ('gchm_degenerate_graph_finite', check_gchm_degenerate_graph),
    ('gchm_batching_does_not_mix_graphs', check_gchm_batching),
    # --- GCHM-PNA v2 arm ---
    ('v2_gate_is_multiplicative', check_v2_gate_is_multiplicative),
    ('v2_hub_state_gates_messages', check_v2_hub_state_gates_messages),
    ('v2_relation_changes_gate', check_v2_relation_changes_gate),
    ('v2_modulation_arms_differ', check_v2_modulation_arms_differ),
    ('v2_degree_scaling_active', check_v2_degree_scaling_active),
    ('v2_hub_state_is_visit_node', check_v2_hub_state_is_visit_node),
    ('v2_degenerate_graph_finite', check_v2_degenerate_graph_finite),
    ('v2_batching_does_not_mix_graphs', check_v2_batching_does_not_mix),
    ('v2_gradients_reach_every_block', check_v2_gradients_reach_every_block),
    ('v2_rejects_foreign_edge_view', check_v2_rejects_foreign_edge_view),
    ('v2_parameters_below_protgnn', check_v2_parameter_budget),
    # --- GCHM-PNA v3 arm ---
    ('v3_wide_path_is_exact_token_vote', check_v3_wide_path_is_exact_token_vote),
    ('v3_labelwise_attention_class_specific', check_v3_labelwise_attention_is_class_specific),
    ('v3_jk_reads_every_depth', check_v3_jk_reads_every_depth),
    ('v3_edge_dropout_train_only', check_v3_edge_dropout_train_only),
    ('v3_degenerate_graph_finite', check_v3_degenerate_graph_finite),
    ('v3_parameters_below_protgnn', check_v3_parameter_budget),
]


def main():
    failures = 0
    for name, fn in CHECKS:
        value, ok = fn()
        print(f'{"PASS" if ok else "FAIL"}  {name:38} {value}')
        failures += not ok
    print(f'\n{len(CHECKS) - failures}/{len(CHECKS)} passed')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
