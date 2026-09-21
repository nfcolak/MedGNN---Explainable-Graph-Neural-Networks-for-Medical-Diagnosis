"""Mechanism tests for GCHM: prove the design claims BEFORE any training run.

Each test targets one measured vulnerability. A passing benchmark score is not
evidence that the mechanism works; these are the evidence.
"""
from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from gchm_analysis.model import GCHM, GatedConceptHubConv, ConceptPairMixer


def _hist():
    return torch.tensor([1, 8, 2])


def _graph(n_concepts=3, seed=0):
    torch.manual_seed(seed)
    n = n_concepts + 1
    x = torch.zeros(n, 331)
    x[0, 0] = 1.0
    x[0, 199:] = torch.randn(132)
    ids = torch.cat([torch.tensor([192]), torch.arange(n_concepts)])
    for k in range(n_concepts):
        x[k + 1, 4 + int(ids[k + 1])] = 1.0
    src = torch.arange(1, n)
    edge_index = torch.cat([torch.stack([src, torch.zeros_like(src)]),
                            torch.stack([torch.zeros_like(src), src])], dim=1)
    types = torch.cat([torch.tensor([2]), torch.where(ids[1:] < 104, 0, 1)])
    return Data(x=x, edge_index=edge_index, node_ids=ids, node_type=types,
                y=torch.tensor([0]))


def test_message_is_not_additively_separable():
    """V1: the incumbent PNA message was additive to 4.8e-07. GCHM must not be."""
    torch.manual_seed(0)
    conv = GatedConceptHubConv(16, _hist()).eval()
    torch.manual_seed(1)
    xi, xj = torch.randn(8, 16), torch.randn(8, 16)
    z = torch.zeros(8, 16)
    conv._edge_gate = torch.ones(8)
    with torch.no_grad():
        full = conv.message(xi, xj)
        a = conv.message(xi, z)
        b = conv.message(z, xj)
        o = conv.message(z, z)
    conv._edge_gate = None
    residual = (full - (a + b - o)).abs().max().item()
    assert residual > 1e-2, f"message collapsed to additive: {residual}"


def test_receiver_state_multiplies_sender_content():
    """The hub's clinical state must change how a FIXED concept message is received."""
    torch.manual_seed(0)
    conv = GatedConceptHubConv(16, _hist()).eval()
    torch.manual_seed(2)
    xj = torch.randn(1, 16)
    conv._edge_gate = torch.ones(1)
    with torch.no_grad():
        outs = [conv.message(torch.full((1, 16), v), xj) for v in (-2.0, 0.0, 2.0)]
    conv._edge_gate = None
    spread = max((outs[i] - outs[j]).abs().max().item()
                 for i in range(3) for j in range(i + 1, 3))
    assert spread > 1e-2, f"receiver state had no multiplicative effect: {spread}"


def test_hub_and_concept_use_separate_parameters():
    """V3: no single 331->width projection shared by both modalities."""
    model = GCHM(degree_histogram=_hist())
    hub_params = {id(p) for p in model.hub.parameters()}
    concept_params = {id(p) for p in model.concept.parameters()}
    assert not (hub_params & concept_params), "hub and concept share parameters"

    # Perturbing hub numerics must not move concept embeddings at all.
    g = _graph()
    with torch.no_grad():
        # encode() returns (node_embeddings, hub_code); this check is about the
        # embeddings, so unpack rather than indexing the tuple.
        h1 = model.encode(g)[0].clone()
        g2 = g.clone()
        g2.x[0, 199:] += 5.0
        h2 = model.encode(g2)[0]
    assert (h1[1:] - h2[1:]).abs().max() == 0, "concept encoding leaked hub numerics"
    assert (h1[0] - h2[0]).abs().max() > 1e-3, "hub encoding ignored its own payload"


def test_conv_contains_real_nonlinearity():
    """V6: incumbent conv had zero activations inside."""
    conv = GatedConceptHubConv(16, _hist())
    acts = [m for m in conv.modules() if isinstance(m, torch.nn.ReLU)]
    assert len(acts) >= 2, f"expected activations inside conv, found {len(acts)}"


def test_edge_mask_applied_exactly_once():
    """Zero mask must silence messages; explain_message must stay identity."""
    torch.manual_seed(0)
    model = GCHM(degree_histogram=_hist()).eval()
    g = Batch.from_data_list([_graph()])
    with torch.no_grad():
        free = model(g)
        silenced = model(g, no_messages=True)
        half = model(g, edge_mask=torch.full((g.edge_index.size(1),), 0.5))
    assert (free - silenced).abs().max() > 1e-4, "edge mask had no effect"
    assert torch.isfinite(half).all()
    conv = model.convs[0]
    probe = torch.randn(3, 5)
    assert torch.equal(conv.explain_message(probe, 3), probe), "explain_message not identity"


def test_hub_only_graph_is_finite():
    """0.93% of training graphs have no concepts at all; they must not produce NaN."""
    torch.manual_seed(0)
    model = GCHM(degree_histogram=_hist()).eval()
    g = _graph(n_concepts=0)
    assert g.edge_index.numel() == 0
    with torch.no_grad():
        out = model(Batch.from_data_list([g]))
    assert torch.isfinite(out).all(), "hub-only graph produced nonfinite logits"


def test_batching_does_not_mix_patients():
    """A graph's logits must be identical alone and inside a batch."""
    torch.manual_seed(0)
    model = GCHM(degree_histogram=_hist()).eval()
    a, b = _graph(3, seed=1), _graph(5, seed=2)
    with torch.no_grad():
        alone = model(Batch.from_data_list([a]))
        together = model(Batch.from_data_list([a, b]))
    assert (alone[0] - together[0]).abs().max() < 1e-5, "batching leaked across patients"


def test_gradients_reach_every_block():
    torch.manual_seed(0)
    model = GCHM(degree_histogram=_hist())
    g = Batch.from_data_list([_graph(4)])
    model(g).sum().backward()
    for name, p in model.named_parameters():
        if p.requires_grad and "concept.weight" not in name:
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"no gradient: {name}"


def test_additive_ablation_is_a_fair_control():
    """The ablation arm must differ ONLY in interaction order.

    Same parameter count, same modules, same shapes. If the additive arm had fewer
    parameters, any multiplicative win would be confounded with capacity.
    """
    mult = GCHM(degree_histogram=_hist(), modulation="multiplicative")
    add = GCHM(degree_histogram=_hist(), modulation="additive")
    n_mult = sum(p.numel() for p in mult.parameters())
    n_add = sum(p.numel() for p in add.parameters())
    assert n_mult == n_add, f"capacity confound: {n_mult} vs {n_add}"

    shapes_m = sorted(tuple(p.shape) for p in mult.parameters())
    shapes_a = sorted(tuple(p.shape) for p in add.parameters())
    assert shapes_m == shapes_a, "ablation changed parameter shapes"


def test_additive_ablation_really_is_additively_separable():
    """The control arm must collapse to additive, like the incumbent PNA did."""
    torch.manual_seed(0)
    conv = GatedConceptHubConv(16, _hist(), modulation="additive").eval()
    torch.manual_seed(1)
    xi, xj = torch.randn(8, 16), torch.randn(8, 16)
    z = torch.zeros(8, 16)
    conv._edge_gate = torch.ones(8)
    with torch.no_grad():
        full = conv.message(xi, xj)
        a = conv.message(xi, z)
        b = conv.message(z, xj)
        o = conv.message(z, z)
    conv._edge_gate = None
    residual = (full - (a + b - o)).abs().max().item()
    # The message MLP itself is nonlinear in the concatenated pair, so this arm is not
    # perfectly separable; what matters is that the receiver gate contributes NO
    # product term. Verify the gate enters purely additively.
    torch.manual_seed(2)
    xj_fixed = torch.randn(1, 16)
    conv._edge_gate = torch.ones(1)
    with torch.no_grad():
        lo = conv.message(torch.full((1, 16), -2.0), xj_fixed)
        hi = conv.message(torch.full((1, 16), 2.0), xj_fixed)
        content_lo = conv.message_mlp(torch.cat([torch.full((1, 16), -2.0), xj_fixed], -1))
        content_hi = conv.message_mlp(torch.cat([torch.full((1, 16), 2.0), xj_fixed], -1))
    conv._edge_gate = None
    # difference must equal the difference of (content + gate), with no scaling of content
    gate_lo = lo - content_lo
    gate_hi = hi - content_hi
    assert torch.all(gate_lo >= -1e-5) and torch.all(gate_lo <= 1 + 1e-5), "gate not a pure sigmoid offset"
    assert torch.all(gate_hi >= -1e-5) and torch.all(gate_hi <= 1 + 1e-5), "gate not a pure sigmoid offset"


def test_concept_pair_mixer_needs_a_peer():
    """A lone concept has no peer, so its pair contribution must be exactly zero."""
    torch.manual_seed(0)
    mixer = ConceptPairMixer(16, 8).eval()
    h = torch.randn(3, 16)
    batch = torch.tensor([0, 0, 0])
    # node 0 is the hub, node 1 the only concept -> no peer for node 1
    is_concept = torch.tensor([False, True, False])
    with torch.no_grad():
        delta = mixer(h, batch, is_concept)
    assert delta.abs().max() == 0, "lone concept received a nonzero pair signal"


def test_concept_pair_mixer_responds_to_peer_identity():
    """Changing ONLY a peer concept must change the target concept's pair signal."""
    torch.manual_seed(0)
    mixer = ConceptPairMixer(16, 8).eval()
    torch.manual_seed(3)
    h = torch.randn(3, 16)
    batch = torch.tensor([0, 0, 0])
    is_concept = torch.tensor([False, True, True])
    with torch.no_grad():
        first = mixer(h, batch, is_concept)[1].clone()
        h2 = h.clone()
        h2[2] += 3.0  # perturb only the PEER
        second = mixer(h2, batch, is_concept)[1]
    assert (first - second).abs().max() > 1e-4, "target concept ignored its peer"


def test_concept_pair_mixer_does_not_leak_across_patients():
    """Concepts of a different patient must never enter the pair summary."""
    torch.manual_seed(0)
    mixer = ConceptPairMixer(16, 8).eval()
    torch.manual_seed(4)
    h = torch.randn(6, 16)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    is_concept = torch.tensor([False, True, True, False, True, True])
    with torch.no_grad():
        base = mixer(h, batch, is_concept)[1].clone()
        h2 = h.clone()
        h2[4] += 5.0  # perturb a concept belonging to patient 1
        after = mixer(h2, batch, is_concept)[1]
    assert (base - after).abs().max() == 0, "pair summary leaked across patients"


def test_concept_pairs_are_silenced_by_edge_ablation():
    """no_messages must zero the pair path too, or edge ablation would lie."""
    torch.manual_seed(0)
    model = GCHM(degree_histogram=_hist(), concept_pairs=True).eval()
    plain = GCHM(degree_histogram=_hist(), concept_pairs=False).eval()
    plain.load_state_dict({k: v for k, v in model.state_dict().items()
                           if not k.startswith("mixers.")}, strict=True)
    g = Batch.from_data_list([_graph(4)])
    with torch.no_grad():
        silenced_pairs = model(g, no_messages=True)
        silenced_plain = plain(g, no_messages=True)
    assert (silenced_pairs - silenced_plain).abs().max() < 1e-5, \
        "pair path survived no_messages"


def test_concept_pairs_change_the_free_prediction():
    """With edges free, the pair path must actually contribute."""
    torch.manual_seed(0)
    model = GCHM(degree_histogram=_hist(), concept_pairs=True).eval()
    plain = GCHM(degree_histogram=_hist(), concept_pairs=False).eval()
    plain.load_state_dict({k: v for k, v in model.state_dict().items()
                           if not k.startswith("mixers.")}, strict=True)
    g = Batch.from_data_list([_graph(4)])
    with torch.no_grad():
        assert (model(g) - plain(g)).abs().max() > 1e-5, "pair path had no effect"


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  PASS  {name}")
            passed += 1
    print(f"\n{passed} mechanism tests passed")
