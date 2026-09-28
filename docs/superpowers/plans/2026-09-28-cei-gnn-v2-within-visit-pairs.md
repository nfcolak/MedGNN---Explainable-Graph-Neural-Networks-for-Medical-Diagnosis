# CEI-GNN v2 Within-Visit Pair Interactions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `cei_gnn_v2` clinical plugin whose only multiplicative interaction is a within-visit evidence-pair term. Add a bounded study runner that trains three modes (`product`, `additive`, `off`) on three seeds and applies a pre-registered decision rule.

**Architecture:** A new core module (`cei_gnn_v2.py`) holds three pieces: the deterministic pair builder, the kind-pair gate index and `PairEvidenceNetwork`. A thin plugin adapter (`plugin_cei_gnn_v2.py`) registers it with the existing runner. A new study module (`cei_v2_study.py`) plans, executes, replays, validates and analyzes runs, reusing generic helpers from `cei_pilot.py`. No v1, ProtGNN or shared runner file changes.

**Tech Stack:** Python 3.9 (system `python3`), torch 2.8.0, torch_geometric 2.6.1, numpy, scikit-learn, pytest 8.4.2.

**Spec:** `docs/superpowers/specs/2026-09-28-cei-gnn-v2-within-visit-pairs-design.md`

## Global Constraints

- Work only in worktree `/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN/.worktrees/cei-v2-pairs`, branch `feature/cei-v2-pairs`, base `5f23c829`.
- Do not modify: `methods/cei_gnn.py`, `methods/plugin_cei_gnn.py`, `cei_pilot.py`, `cei_graphxai.py`, `train.py`, `tensorize.py`, `methods/base.py`, `methods/__init__.py`, any ProtGNN file, any artifact or existing run directory.
- TDD: each task commits a `red:` commit (new tests failing on an assertion, not an ImportError) before its `green:` commit.
- Never load test-fold tensors; never evaluate validation; `--final-eval none`, `--selection-fold dev` only.
- No real data reading, preprocessing or training before the explicit approval gates in Tasks 7 and 8.
- Method defaults are copied from v1 unchanged: `hidden=128, layers=1, dropout=0.3, lr=1.79e-3, weight_decay=4.3e-5, batch_size=128, epochs=40, patience=10, min_delta=0.005`, grad clip value 2.0, `pair_rank=16`.
- Study policy: model seeds `1234, 2025, 7`; sample seed `1234`; `--train-limit 10000 --dev-limit 5000 --epochs 40 --patience 40`; smoke `256/128/2`, seed 1234, mode `product`.
- Evidence kinds are `complaint`, `measurement`, `vital`. `NODE_KINDS` ids: patient 0, visit 1, complaint 2, measurement 3, analyte 4, vital 5, knowledge 6, diagnosis 7.
- Bootstrap: 1,000 patient-cluster resamples, numpy `default_rng(2026)`.
- Test command (explicit paths avoid vendored test collections): `python3 -m pytest <files> -q -p no:cacheprovider`.

## File Structure

| File | Responsibility |
|---|---|
| Create `comparison/standardized/clinical_graph_v2/methods/cei_gnn_v2.py` | `within_visit_pairs`, `kind_pair_index`, `PairEvidenceNetwork` |
| Create `comparison/standardized/clinical_graph_v2/methods/plugin_cei_gnn_v2.py` | `PairEvidenceAdapter`, `REGISTER` |
| Create `comparison/standardized/clinical_graph_v2/cei_v2_study.py` | plan, binding validation, replay, execution, preflight, analysis, CLI |
| Create `tests/test_cei_gnn_v2_core.py` | pair builder and network tests |
| Create `tests/test_plugin_cei_gnn_v2.py` | adapter tests |
| Create `tests/test_cei_v2_study.py` | study tests |
| Create (Task 8) `docs/cei-gnn-v2-pairs-2026-09-28.md` | result report |

---

### Task 1: Pair builder and kind-pair index

**Files:**
- Create: `comparison/standardized/clinical_graph_v2/methods/cei_gnn_v2.py`
- Test: `tests/test_cei_gnn_v2_core.py`

**Interfaces:**
- Produces: `EVIDENCE_KINDS`, `EVIDENCE_KIND_IDS`, `KIND_PAIR_COUNT == 6`, `PAIR_MODES == ("product", "additive", "off")`, `within_visit_pairs(membership: LongTensor[2, P], node_type: LongTensor[N], node_count: int) -> LongTensor[2, Q]` (rows `i < j`, sorted by `i * N + j`), `kind_pair_index(node_type, pairs) -> LongTensor[Q]` in `0..5`.

- [ ] **Step 1: Write the failing tests**

```python
"""Synthetic tests for the CEI-GNN v2 core: pair builder and pair-evidence network."""
import importlib
import importlib.util

import pytest
import torch

CORE = "comparison.standardized.clinical_graph_v2.methods.cei_gnn_v2"
EXPECTED_PAIRS = [(2, 3), (2, 4), (2, 6), (2, 7), (3, 4), (3, 6), (3, 7), (6, 7)]


def _v2():
    assert importlib.util.find_spec(CORE) is not None, "CEI-GNN v2 core module is missing"
    return importlib.import_module(CORE)


def _graph(scale=1.0, seed=7):
    """Two visits. Node 2 (complaint) and node 3 (measurement) belong to both."""
    from comparison.standardized.clinical_graph_v2.tensorize import ClinicalGraphData

    generator = torch.Generator().manual_seed(seed)
    graph = ClinicalGraphData(
        x=torch.randn(8, 3, generator=generator) * scale,
        edge_index=torch.tensor([[0, 1, 1, 1, 3, 6], [1, 2, 3, 4, 5, 5]]),
        edge_attr=torch.randn(6, 2, generator=generator))
    # kinds: 0 patient, 1 visit, 2 complaint, 3 measurement, 5 vital, 4 analyte
    graph.node_type = torch.tensor([0, 1, 2, 3, 5, 4, 3, 2])
    graph.token = torch.tensor([1, 2, 3, 4, 5, 6, 4, 3])
    graph.edge_relation = torch.tensor([0, 2, 3, 5, 4, 4])
    graph.edge_triple = torch.tensor([0, 1, 2, 3, 1, 1])
    graph.visit_membership_index = torch.tensor(
        [[0, 0, 0, 0, 0, 1, 1, 1, 1], [1, 2, 3, 4, 5, 2, 6, 7, 3]])
    graph.num_visits = torch.tensor([2])
    graph.y = torch.tensor([1])
    return graph


def _pairs_of(graph):
    v2 = _v2()
    pairs = v2.within_visit_pairs(graph.visit_membership_index, graph.node_type,
                                  int(graph.num_nodes))
    return [tuple(pair) for pair in pairs.t().tolist()]


def test_pairs_are_unique_within_visit_evidence_pairs():
    assert _pairs_of(_graph()) == EXPECTED_PAIRS


def test_pairs_follow_pyg_batch_offsets_and_never_cross_graphs():
    from torch_geometric.data import Batch

    first, second = _graph(), _graph(seed=8)
    batch = Batch.from_data_list([first, second])
    pairs = _pairs_of(batch)
    assert pairs == EXPECTED_PAIRS + [(i + 8, j + 8) for i, j in EXPECTED_PAIRS]
    left, right = torch.tensor(pairs).t()
    assert torch.equal(batch.batch[left], batch.batch[right])


def test_pairs_are_empty_with_fewer_than_two_evidence_nodes_per_visit():
    v2 = _v2()
    membership = torch.tensor([[0, 0, 1], [0, 1, 2]])
    node_type = torch.tensor([1, 2, 5])
    pairs = v2.within_visit_pairs(membership, node_type, 3)
    assert pairs.shape == (2, 0) and pairs.dtype == torch.long
    empty = v2.within_visit_pairs(torch.zeros((2, 0), dtype=torch.long), node_type, 3)
    assert empty.shape == (2, 0)


def test_pair_builder_rejects_malformed_membership():
    v2 = _v2()
    node_type = torch.tensor([2, 3])
    with pytest.raises(ValueError, match="shape"):
        v2.within_visit_pairs(torch.tensor([0, 1]), node_type, 2)
    with pytest.raises(ValueError, match="outside"):
        v2.within_visit_pairs(torch.tensor([[0, 0], [0, 5]]), node_type, 2)


def test_kind_pair_index_is_symmetric_and_covers_six_unordered_pairs():
    v2 = _v2()
    assert v2.KIND_PAIR_COUNT == 6
    node_type = torch.tensor([2, 3, 5])
    seen = {}
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            index = v2.kind_pair_index(node_type, torch.tensor([[i], [j]])).item()
            seen.setdefault(frozenset((i, j)), set()).add(index)
    assert all(len(values) == 1 for values in seen.values())
    same_kind = torch.tensor([2, 2, 3, 3, 5, 5])
    diagonal = v2.kind_pair_index(same_kind, torch.tensor([[0, 2, 4], [1, 3, 5]])).tolist()
    mixed = {next(iter(values)) for values in seen.values()}
    assert sorted(mixed | set(diagonal)) == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="evidence"):
        v2.kind_pair_index(torch.tensor([1, 2]), torch.tensor([[0], [1]]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cei_gnn_v2_core.py -q -p no:cacheprovider`
Expected: 5 failed, each with `AssertionError: CEI-GNN v2 core module is missing`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_cei_gnn_v2_core.py
git commit -m "red: V2-1 require within-visit evidence pair builder"
```

- [ ] **Step 4: Write minimal implementation**

```python
"""CEI-GNN v2: node, endpoint-additive edge and within-visit pair evidence."""
from __future__ import annotations

import torch

from .. import NODE_KINDS

EVIDENCE_KINDS = ("complaint", "measurement", "vital")
EVIDENCE_KIND_IDS = tuple(NODE_KINDS.index(kind) for kind in EVIDENCE_KINDS)
KIND_PAIR_COUNT = len(EVIDENCE_KINDS) * (len(EVIDENCE_KINDS) + 1) // 2
PAIR_MODES = ("product", "additive", "off")


def _empty_pairs(device):
    return torch.zeros((2, 0), dtype=torch.long, device=device)


def within_visit_pairs(membership, node_type, node_count):
    """Unique unordered evidence-node pairs (i < j) that share at least one visit.

    Membership rows are (visit id, node id) after PyG batching, so visit ids are
    unique across graphs and pairs never cross graphs. Deterministic; no sampling.
    """
    device = node_type.device
    if membership.ndim != 2 or membership.size(0) != 2:
        raise ValueError("visit_membership_index must have shape [2, pairs]")
    if membership.size(1) == 0:
        return _empty_pairs(device)
    membership = membership.to(device=device, dtype=torch.long)
    visit, node = membership[0], membership[1]
    if int(node.min()) < 0 or int(node.max()) >= int(node_count) or int(visit.min()) < 0:
        raise ValueError("visit membership refers to a node outside the batch")
    evidence = torch.tensor(EVIDENCE_KIND_IDS, device=device)
    keep = torch.isin(node_type[node], evidence)
    visit, node = visit[keep], node[keep]
    if node.numel() < 2:
        return _empty_pairs(device)
    count = int(node_count)
    key = torch.unique(visit * count + node)
    visit, node = key // count, key % count
    _, sizes = torch.unique_consecutive(visit, return_counts=True)
    width = int(sizes.max())
    if width < 2:
        return _empty_pairs(device)
    group = torch.arange(sizes.numel(), device=device).repeat_interleave(sizes)
    starts = torch.cumsum(sizes, 0) - sizes
    position = torch.arange(node.numel(), device=device) - starts.repeat_interleave(sizes)
    dense = node.new_full((sizes.numel(), width), -1)
    dense[group, position] = node
    left_slot, right_slot = torch.triu_indices(width, width, offset=1, device=device)
    left, right = dense[:, left_slot], dense[:, right_slot]
    valid = (left >= 0) & (right >= 0)
    pair_key = torch.unique(left[valid] * count + right[valid])
    return torch.stack((pair_key // count, pair_key % count))


def kind_pair_index(node_type, pairs):
    """Index 0..5 of the unordered evidence-kind pair of each node pair."""
    device = node_type.device
    lookup = torch.full((len(NODE_KINDS),), -1, dtype=torch.long, device=device)
    lookup[torch.tensor(EVIDENCE_KIND_IDS, device=device)] = torch.arange(
        len(EVIDENCE_KINDS), device=device)
    first, second = lookup[node_type[pairs[0]]], lookup[node_type[pairs[1]]]
    if bool((first < 0).any()) or bool((second < 0).any()):
        raise ValueError("pair endpoint is not an evidence node")
    low, high = torch.minimum(first, second), torch.maximum(first, second)
    kinds = len(EVIDENCE_KINDS)
    return low * kinds - low * (low - 1) // 2 + (high - low)
```

Within each visit, `key` is sorted, so nodes are ascending and `left < right` holds.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cei_gnn_v2_core.py -q -p no:cacheprovider`
Expected: 5 passed.

- [ ] **Step 6: Commit green**

```bash
git add comparison/standardized/clinical_graph_v2/methods/cei_gnn_v2.py
git commit -m "green: V2-1 build deterministic within-visit evidence pairs"
```

---

### Task 2: PairEvidenceNetwork

**Files:**
- Modify: `comparison/standardized/clinical_graph_v2/methods/cei_gnn_v2.py` (append)
- Test: `tests/test_cei_gnn_v2_core.py` (append)

**Interfaces:**
- Consumes: Task 1 functions; `EdgeEvidenceAggregator` from `methods/cei_gnn.py` (imported, not modified); `ClinicalBatch` from `methods/base.py`.
- Produces: `PairEvidenceNetwork(*, num_tokens, node_dim, edge_dim, num_classes, hidden, token_dim, num_triples, num_relations, dropout, pair_rank, pair_mode, num_node_types)`.
  - `.continuous_inputs(clinical) -> Tensor[N, node_dim + token_dim + hidden]`
  - `.set_edge_mask(mask | None)` (same semantics as v1)
  - `.forward_continuous(features, edge_index, metadata: ClinicalBatch, membership, *, return_parts=False)`. Returns logits `[G, C]`, or with `return_parts=True` a dict with keys `logits, node_contributions [N, C], edge_contributions [E, C], pair_contributions [Q, C], pairs [2, Q], bias [C]`.
  - Parameter names: `token_embedding, node_type_embedding, node_encoder, node_norm, node_head, relation_embedding, triple_embedding, edge_feature_projection, edge_source, edge_target, edge_context_vote, edge_context_gate, pair_projection, pair_vote, pair_gate, bias`.

- [ ] **Step 1: Write the failing tests (append)**

```python
def _network(mode="product", seed=11):
    v2 = _v2()
    assert hasattr(v2, "PairEvidenceNetwork"), "PairEvidenceNetwork is missing"
    torch.manual_seed(seed)
    network = v2.PairEvidenceNetwork(
        num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8, token_dim=4,
        num_triples=4, num_relations=15, dropout=0.0, pair_rank=4, pair_mode=mode,
        num_node_types=8).eval()
    with torch.no_grad():
        network.pair_gate.normal_()
        network.edge_context_gate.bias.normal_()
    return network


def _metadata(graph):
    from comparison.standardized.clinical_graph_v2.methods.base import read_clinical_batch

    return read_clinical_batch(graph, method="test", node_dim=3, edge_dim=2, num_tokens=8,
                               num_triples=4, num_relations=15)


def _run(network, graph, features=None, **kwargs):
    metadata = _metadata(graph)
    if features is None:
        features = network.continuous_inputs(metadata)
    return network.forward_continuous(features, metadata.edge_index, metadata,
                                      graph.visit_membership_index, **kwargs)


def _block_mixed_difference(network, graph, block, first, second):
    metadata = _metadata(graph)
    base = network.continuous_inputs(metadata).detach()
    generator = torch.Generator().manual_seed(3)
    delta_first = torch.randn(base.size(1), generator=generator) * 0.5
    delta_second = torch.randn(base.size(1), generator=generator) * 0.5
    zero = torch.zeros_like(delta_first)

    def total(shift_first, shift_second):
        features = base.clone()
        features[first] += shift_first
        features[second] += shift_second
        parts = network.forward_continuous(features, metadata.edge_index, metadata,
                                           graph.visit_membership_index, return_parts=True)
        return parts[block].sum(0)

    return (total(delta_first, delta_second) - total(delta_first, zero)
            - total(zero, delta_second) + total(zero, zero))


@pytest.mark.parametrize("mode", ["product", "additive", "off"])
def test_parts_reconstruct_logits_in_every_mode(mode):
    parts = _run(_network(mode), _graph(), return_parts=True)
    rebuilt = (parts["bias"] + parts["node_contributions"].sum(0)
               + parts["edge_contributions"].sum(0) + parts["pair_contributions"].sum(0))
    torch.testing.assert_close(parts["logits"][0], rebuilt, rtol=1e-5, atol=1e-5)
    assert parts["pairs"].t().tolist() == [list(pair) for pair in EXPECTED_PAIRS]
    assert parts["pair_contributions"].shape == (8, 3)
    if mode == "off":
        assert torch.count_nonzero(parts["pair_contributions"]) == 0
    else:
        assert parts["pair_contributions"].abs().sum() > 0


def test_modes_share_parameter_names_and_shapes_and_change_predictions():
    networks = {mode: _network(mode) for mode in ("product", "additive", "off")}
    shapes = {mode: [(name, tuple(p.shape)) for name, p in net.named_parameters()]
              for mode, net in networks.items()}
    assert shapes["product"] == shapes["additive"] == shapes["off"]
    state = networks["product"].state_dict()
    for net in networks.values():
        net.load_state_dict(state)
    logits = {mode: _run(net, _graph()) for mode, net in networks.items()}
    assert not torch.allclose(logits["product"], logits["additive"])
    assert not torch.allclose(logits["product"], logits["off"])
    assert not torch.allclose(logits["additive"], logits["off"])


def test_only_product_mode_has_a_pair_cross_term():
    product = _block_mixed_difference(_network("product"), _graph(), "pair_contributions", 2, 3)
    additive = _block_mixed_difference(_network("additive"), _graph(), "pair_contributions", 2, 3)
    assert product.abs().max() > 1e-4, "product pair term collapsed to an additive response"
    assert additive.abs().max() < 1e-5, "additive control contains a hidden cross term"


def test_edge_block_has_no_endpoint_cross_term():
    for mode in ("product", "additive"):
        difference = _block_mixed_difference(_network(mode), _graph(), "edge_contributions", 1, 2)
        assert difference.abs().max() < 1e-5, f"edge endpoints interact in {mode} mode"


def test_zero_edge_mask_removes_only_the_edge_block():
    network = _network("product")
    graph = _graph()
    ordinary = _run(network, graph, return_parts=True)
    network.set_edge_mask(torch.zeros(graph.num_edges))
    masked = _run(network, graph, return_parts=True)
    network.set_edge_mask(None)
    assert torch.count_nonzero(masked["edge_contributions"]) == 0
    expected = (masked["bias"] + ordinary["node_contributions"].sum(0)
                + ordinary["pair_contributions"].sum(0))
    torch.testing.assert_close(masked["logits"][0], expected, rtol=1e-5, atol=1e-5)


def test_pyg_mask_matches_direct_probability_mask_and_carries_gradient():
    from torch_geometric.explain.algorithm.utils import clear_masks, set_masks

    network = _network("product")
    graph = _graph()
    metadata = _metadata(graph)
    raw = torch.nn.Parameter(torch.tensor([-0.9, 0.3, 1.1, -0.2, 0.5, 0.8]))
    set_masks(network, raw, metadata.edge_index, apply_sigmoid=True)
    pyg = _run(network, graph)
    (pyg * torch.tensor([[0.2, -0.7, 1.1]])).sum().backward()
    assert raw.grad is not None and raw.grad.abs().sum() > 0
    clear_masks(network)
    network.set_edge_mask(raw.detach().sigmoid())
    direct = _run(network, graph)
    network.set_edge_mask(None)
    torch.testing.assert_close(pyg.detach(), direct, rtol=1e-6, atol=1e-6)


def test_edgeless_pairless_graph_is_finite():
    from comparison.standardized.clinical_graph_v2.tensorize import ClinicalGraphData

    graph = ClinicalGraphData(x=torch.ones((2, 3)), edge_index=torch.zeros((2, 0), dtype=torch.long),
                              edge_attr=torch.zeros((0, 2)))
    graph.node_type = torch.tensor([1, 2])
    graph.token = torch.tensor([1, 2])
    graph.edge_relation = torch.zeros(0, dtype=torch.long)
    graph.edge_triple = torch.zeros(0, dtype=torch.long)
    graph.visit_membership_index = torch.tensor([[0, 0], [0, 1]])
    graph.num_visits = torch.tensor([1])
    for mode in ("product", "additive", "off"):
        parts = _run(_network(mode), graph, return_parts=True)
        assert parts["logits"].shape == (1, 3) and torch.isfinite(parts["logits"]).all()
        assert parts["pairs"].shape == (2, 0)


def test_network_fails_closed_on_invalid_inputs():
    v2 = _v2()
    network = _network("product")
    graph = _graph()
    metadata = _metadata(graph)
    features = network.continuous_inputs(metadata).detach()
    features[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        network.forward_continuous(features, metadata.edge_index, metadata,
                                   graph.visit_membership_index)
    with pytest.raises(ValueError, match="edge mask"):
        network.set_edge_mask(torch.full((graph.num_edges,), 1.5))
    with pytest.raises(ValueError, match="pair_mode"):
        v2.PairEvidenceNetwork(num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8,
                               token_dim=4, num_triples=4, num_relations=15, dropout=0.0,
                               pair_rank=4, pair_mode="both", num_node_types=8)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python3 -m pytest tests/test_cei_gnn_v2_core.py -q -p no:cacheprovider`
Expected: 5 passed (Task 1); every new test fails with `AssertionError: PairEvidenceNetwork is missing`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_cei_gnn_v2_core.py
git commit -m "red: V2-2 require pair-evidence network accounting and controls"
```

- [ ] **Step 4: Write minimal implementation (append to `cei_gnn_v2.py`)**

Add these imports at the top of the file, next to `import torch`:

```python
import torch.nn as nn
import torch.nn.functional as F

from .cei_gnn import EdgeEvidenceAggregator
```

Append:

```python
class PairEvidenceNetwork(nn.Module):
    """Exact continuous-input core of CEI-GNN v2."""

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden, token_dim,
                 num_triples, num_relations, dropout, pair_rank, pair_mode, num_node_types):
        super().__init__()
        if pair_mode not in PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {list(PAIR_MODES)}")
        self.num_tokens, self.node_dim, self.edge_dim = int(num_tokens), int(node_dim), int(edge_dim)
        self.num_classes, self.hidden = int(num_classes), int(hidden)
        self.token_dim, self.num_triples = int(token_dim), int(num_triples)
        self.num_relations = int(num_relations)
        self.dropout_rate, self.pair_rank, self.pair_mode = float(dropout), int(pair_rank), pair_mode
        classes, width = self.num_classes, self.hidden
        self.token_embedding = nn.Embedding(self.num_tokens, self.token_dim, padding_idx=0)
        self.node_type_embedding = nn.Embedding(int(num_node_types), width)
        self.node_encoder = nn.Linear(self.node_dim + self.token_dim + width, width)
        self.node_norm = nn.LayerNorm(width)
        self.node_head = nn.Linear(width, classes * 2)
        self.relation_embedding = nn.Embedding(self.num_relations, width)
        self.triple_embedding = nn.Embedding(self.num_triples, width)
        self.edge_feature_projection = nn.Linear(self.edge_dim, width, bias=False)
        self.edge_source = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                         nn.Dropout(self.dropout_rate), nn.Linear(width, classes))
        self.edge_target = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                         nn.Dropout(self.dropout_rate), nn.Linear(width, classes))
        self.edge_context_vote = nn.Linear(width, classes)
        self.edge_context_gate = nn.Linear(width, classes)
        self.edge_aggregator = EdgeEvidenceAggregator(classes)
        self.edge_aggregator._edge_mask = None
        self.edge_aggregator._apply_sigmoid = False
        self.pair_projection = nn.Linear(width, self.pair_rank, bias=False)
        self.pair_vote = nn.Linear(self.pair_rank, classes)
        self.pair_gate = nn.Parameter(torch.zeros(KIND_PAIR_COUNT, classes))
        self.bias = nn.Parameter(torch.zeros(classes))

    @property
    def continuous_width(self):
        return self.node_dim + self.token_dim + self.hidden

    def continuous_inputs(self, clinical):
        return torch.cat((clinical.x, self.token_embedding(clinical.token),
                          self.node_type_embedding(clinical.node_type)), dim=-1)

    def set_edge_mask(self, mask):
        """Install a direct probability mask or restore PyG's neutral mask state."""
        aggregator = self.edge_aggregator
        if mask is not None:
            if mask.ndim != 1 or not torch.isfinite(mask).all():
                raise ValueError("edge mask must be a finite vector")
            if (mask < 0).any() or (mask > 1).any():
                raise ValueError("edge mask values must be in [0, 1]")
        if "_edge_mask" in aggregator._parameters:
            del aggregator._parameters["_edge_mask"]
        aggregator._edge_mask = mask
        aggregator.explain = None if mask is None else True
        aggregator._apply_sigmoid = False if mask is not None else True

    def _validate(self, features, edge_index, metadata):
        node_count = int(features.size(0))
        if features.ndim != 2 or tuple(features.shape) != (node_count, self.continuous_width):
            raise ValueError("continuous features have an incompatible shape")
        if node_count == 0:
            raise ValueError("CEI-GNN v2 requires at least one node")
        if not torch.isfinite(features).all():
            raise ValueError("continuous features must be finite")
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be a long tensor with shape [2, edges]")
        edge_count = edge_index.size(1)
        edge_mask = self.edge_aggregator._edge_mask
        if edge_mask is not None and (edge_mask.ndim != 1 or edge_mask.numel() != edge_count):
            raise ValueError("edge mask length differs from the edge list")
        batch_index = metadata.batch_index
        if edge_count:
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count:
                raise ValueError("edge_index refers to a node outside the batch")
            if not torch.equal(batch_index[edge_index[0]], batch_index[edge_index[1]]):
                raise ValueError("cross-graph edges are not allowed")
            relation, triple = metadata.edge_relation, metadata.edge_triple
            if int(relation.min()) < 0 or int(relation.max()) >= self.num_relations:
                raise ValueError("edge_relation index is outside the fitted vocabulary")
            if int(triple.min()) < 0 or int(triple.max()) >= self.num_triples:
                raise ValueError("edge_triple index is outside the fitted vocabulary")
        if int(metadata.graph_count) < 1 or batch_index.numel() != node_count:
            raise ValueError("invalid graph membership metadata")
        if not torch.isfinite(metadata.edge_attr).all():
            raise ValueError("edge_attr must be finite")

    def forward_continuous(self, features, edge_index, metadata, membership, *,
                           return_parts=False):
        self._validate(features, edge_index, metadata)
        node_count, graph_count = int(features.size(0)), int(metadata.graph_count)
        batch_index, classes = metadata.batch_index, self.num_classes
        numeric, token_vectors, type_vectors = torch.split(
            features, (self.node_dim, self.token_dim, self.hidden), dim=-1)
        h = F.gelu(self.node_norm(self.node_encoder(torch.cat(
            (numeric, token_vectors, type_vectors), dim=-1))))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        node_vote, node_gate = self.node_head(h).chunk(2, dim=-1)
        node_gate = node_gate.sigmoid()
        node_num = h.new_zeros((graph_count, classes)).index_add(0, batch_index, node_gate * node_vote)
        node_den = h.new_ones((graph_count, classes)).index_add(0, batch_index, node_gate)
        node_parts = (node_gate * node_vote) / node_den[batch_index]

        edge_count = edge_index.size(1)
        context = (self.relation_embedding(metadata.edge_relation)
                   + self.triple_embedding(metadata.edge_triple)
                   + self.edge_feature_projection(metadata.edge_attr))
        src, dst = edge_index
        edge_vote = (self.edge_source(h[src]) + self.edge_target(h[dst])
                     + self.edge_context_vote(context))
        edge_gate = self.edge_context_gate(context).sigmoid()
        vote_sum, gate_sum = self.edge_aggregator(edge_index, edge_vote, edge_gate, node_count)
        edge_vote_graph = h.new_zeros((graph_count, classes)).index_add(0, batch_index, vote_sum)
        edge_gate_graph = h.new_zeros((graph_count, classes)).index_add(0, batch_index, gate_sum)
        edge_denominator = 1.0 + edge_gate_graph
        edge_mask = self.edge_aggregator._edge_mask
        if edge_mask is None:
            effective_gate_vote = edge_gate * edge_vote
        else:
            edge_mask = edge_mask.to(device=edge_gate.device, dtype=edge_gate.dtype)
            if getattr(self.edge_aggregator, "_apply_sigmoid", False):
                edge_mask = edge_mask.sigmoid()
            effective_gate_vote = edge_mask[:, None] * edge_gate * edge_vote
        edge_parts = (effective_gate_vote / edge_denominator[batch_index[src]]
                      if edge_count else edge_vote)

        pairs = within_visit_pairs(membership, metadata.node_type, node_count)
        pair_count = pairs.size(1)
        if self.pair_mode == "off" or pair_count == 0:
            pair_total = h.new_zeros((graph_count, classes))
            pair_parts = h.new_zeros((pair_count, classes))
        else:
            z = torch.tanh(self.pair_projection(h))
            left, right = pairs
            q = z[left] * z[right] if self.pair_mode == "product" else z[left] + z[right]
            q = F.dropout(q, p=self.dropout_rate, training=self.training)
            pair_vote = self.pair_vote(q)
            pair_gate = self.pair_gate[kind_pair_index(metadata.node_type, pairs)].sigmoid()
            pair_graph = batch_index[left]
            pair_num = h.new_zeros((graph_count, classes)).index_add(
                0, pair_graph, pair_gate * pair_vote)
            pair_den = h.new_ones((graph_count, classes)).index_add(0, pair_graph, pair_gate)
            pair_total = pair_num / pair_den
            pair_parts = (pair_gate * pair_vote) / pair_den[pair_graph]

        logits = (self.bias + node_num / node_den + edge_vote_graph / edge_denominator
                  + pair_total)
        if not return_parts:
            return logits
        return {"logits": logits, "node_contributions": node_parts,
                "edge_contributions": edge_parts, "pair_contributions": pair_parts,
                "pairs": pairs, "bias": self.bias}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cei_gnn_v2_core.py tests/test_plugin_cei_gnn.py -q -p no:cacheprovider`
Expected: all passed. v1 tests are unchanged and must still pass.

- [ ] **Step 6: Commit green**

```bash
git add comparison/standardized/clinical_graph_v2/methods/cei_gnn_v2.py
git commit -m "green: V2-2 add pair-evidence network with additive edges"
```

---

### Task 3: Plugin adapter `cei_gnn_v2`

**Files:**
- Create: `comparison/standardized/clinical_graph_v2/methods/plugin_cei_gnn_v2.py`
- Test: `tests/test_plugin_cei_gnn_v2.py`

**Interfaces:**
- Consumes: `PairEvidenceNetwork`, `PAIR_MODES`; base helpers `ClinicalMethodAdapter, MethodOutput, diagnostic_float, method_option, parameter_count, read_clinical_batch, relation_count, reject_unknown_options`.
- Produces: `PairEvidenceAdapter` registered as `"cei_gnn_v2"`, with:
  - `adaptation_version = "clinical_graph_v2_cei_gnn_v2_pairs"`
  - `runner_defaults` identical to v1
  - `grad_clip_value = 2.0`
  - attributes `pair_rank`, `pair_mode`, `network`, `num_relations`
  - methods `continuous_inputs(batch)`, `forward_continuous(features, edge_index, metadata, *, return_parts=False)`, `inactive_parameter_count()`
  - `run_config()["effective_settings"] == {"pair_rank", "pair_mode"}`; `run_config()["architecture"]` keys: `node_dim, edge_dim, num_tokens, num_triples, num_relations, num_node_types, num_classes, hidden, layers, dropout, token_dim, pair_rank, parameter_count, active_parameter_count, inactive_parameter_count`.

- [ ] **Step 1: Write the failing tests**

```python
"""Synthetic adapter tests for the CEI-GNN v2 plugin."""
from argparse import Namespace

import pytest
import torch

from tests.test_cei_gnn_v2_core import EXPECTED_PAIRS, _graph


def _registry():
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY

    assert "cei_gnn_v2" in METHOD_REGISTRY, "cei_gnn_v2 plugin is not registered"
    return METHOD_REGISTRY


def _adapter(mode="product", rank=4, seed=11, **options):
    torch.manual_seed(seed)
    return _registry()["cei_gnn_v2"](
        num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8, layers=1,
        dropout=0.0, token_dim=4, num_triples=4,
        args=Namespace(method_options={"pair_rank": rank, "pair_mode": mode, **options})).eval()


def test_plugin_uses_v1_runner_defaults_and_parses_runner_options():
    from comparison.standardized.clinical_graph_v2 import train

    registry = _registry()
    assert train.plugin_defaults("cei_gnn_v2") == registry["cei_gnn"].runner_defaults
    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", "a", "--targets", "t", "--output", "o", "--method", "cei_gnn_v2",
        "--method-option", "pair_mode=additive"]), parser)
    adapter = registry["cei_gnn_v2"](
        num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=args.hidden,
        layers=args.layers, dropout=args.dropout, token_dim=args.token_dim, num_triples=4,
        args=args)
    assert (adapter.pair_mode, adapter.pair_rank, adapter.grad_clip_value) == ("additive", 16, 2.0)
    assert adapter.run_config()["effective_settings"] == {"pair_rank": 16, "pair_mode": "additive"}


def test_constructor_rejects_bad_mode_unknown_option_and_depth():
    cls = _registry()["cei_gnn_v2"]
    common = dict(num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8,
                  dropout=0.0, token_dim=4, num_triples=4)
    with pytest.raises(ValueError, match="pair_mode"):
        cls(**common, layers=1, args=Namespace(method_options={"pair_mode": "both"}))
    with pytest.raises(ValueError, match="unknown method option"):
        cls(**common, layers=1, args=Namespace(method_options={"use_interactions": "true"}))
    with pytest.raises(ValueError, match="layers=1"):
        cls(**common, layers=2, args=Namespace(method_options={}))


def test_run_config_reports_inactive_pair_parameters_only_when_off():
    product, off = _adapter("product"), _adapter("off")
    total = sum(p.numel() for p in product.parameters())
    assert product.run_config()["architecture"]["parameter_count"] == total
    assert product.run_config()["architecture"]["inactive_parameter_count"] == 0
    net = off.network
    inactive = (net.pair_projection.weight.numel() + net.pair_vote.weight.numel()
                + net.pair_vote.bias.numel() + net.pair_gate.numel())
    architecture = off.run_config()["architecture"]
    assert architecture["parameter_count"] == total
    assert architecture["inactive_parameter_count"] == inactive
    assert architecture["active_parameter_count"] == total - inactive


def test_forward_equals_continuous_core_and_graphxai_wrapper():
    from comparison.standardized.clinical_graph_v2.cei_graphxai import ClinicalGraphXAIWrapper

    adapter, graph = _adapter("product"), _graph()
    ordinary = adapter(graph, epoch=0).logits
    features = adapter.continuous_inputs(graph)
    continuous = adapter.forward_continuous(features, graph.edge_index, graph)
    torch.testing.assert_close(ordinary, continuous, rtol=0, atol=0)
    wrapped = ClinicalGraphXAIWrapper(adapter, graph)(features, graph.edge_index)
    torch.testing.assert_close(ordinary, wrapped, rtol=0, atol=0)
    parts = adapter.forward_continuous(features, graph.edge_index, graph, return_parts=True)
    assert parts["pairs"].t().tolist() == [list(pair) for pair in EXPECTED_PAIRS]


def test_batched_logits_equal_individual_graphs():
    from torch_geometric.data import Batch

    adapter = _adapter("product")
    first, second = _graph(), _graph(scale=-0.4, seed=8)
    batched = adapter(Batch.from_data_list([first, second]), epoch=0).logits
    torch.testing.assert_close(batched[0], adapter(first, epoch=0).logits[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(batched[1], adapter(second, epoch=0).logits[0], rtol=1e-5, atol=1e-5)


def test_missing_membership_and_float_indices_fail_closed():
    adapter, graph = _adapter("off"), _graph()
    missing = graph.clone()
    del missing.visit_membership_index
    with pytest.raises(ValueError, match="visit_membership_index"):
        adapter(missing, epoch=0)
    floating = graph.clone()
    floating.visit_membership_index = floating.visit_membership_index.float()
    with pytest.raises(ValueError, match="integer"):
        adapter(floating, epoch=0)


def test_gradients_reach_active_parameters_only():
    weights = torch.tensor([[0.2, -0.7, 1.1]])
    for mode in ("product", "additive", "off"):
        adapter = _adapter(mode)
        with torch.no_grad():
            adapter.network.pair_gate.normal_()
        (adapter(_graph(), epoch=0).logits * weights).sum().backward()
        for name, parameter in adapter.network.named_parameters():
            if mode == "off" and name.startswith("pair_"):
                assert parameter.grad is None, f"off mode trains inactive {name}"
            else:
                assert parameter.grad is not None, f"disconnected {name} in {mode}"
                assert torch.isfinite(parameter.grad).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_plugin_cei_gnn_v2.py -q -p no:cacheprovider`
Expected: 7 failed with `AssertionError: cei_gnn_v2 plugin is not registered`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_plugin_cei_gnn_v2.py
git commit -m "red: V2-3 require registered cei_gnn_v2 plugin adapter"
```

- [ ] **Step 4: Write minimal implementation**

```python
"""CEI-GNN v2 adapter: node, endpoint-additive edge and within-visit pair evidence."""
from __future__ import annotations

import torch

from .. import NODE_KINDS
from .base import (ClinicalMethodAdapter, MethodOutput, diagnostic_float, method_option,
                   parameter_count, read_clinical_batch, relation_count,
                   reject_unknown_options)
from .cei_gnn_v2 import PAIR_MODES, PairEvidenceNetwork

KNOWN_OPTIONS = frozenset(("pair_rank", "pair_mode"))
_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_INDEX_FIELDS = ("token", "node_type", "edge_index", "edge_relation", "edge_triple", "batch",
                 "visit_membership_index")


class PairEvidenceAdapter(ClinicalMethodAdapter):
    """CEI-GNN v2 with a within-visit evidence-pair interaction block."""

    adaptation_version = "clinical_graph_v2_cei_gnn_v2_pairs"
    runner_defaults = dict(hidden=128, layers=1, dropout=0.3, lr=1.79e-3,
                           weight_decay=4.3e-5, batch_size=128, epochs=40,
                           patience=10, min_delta=0.005)
    grad_clip_value = 2.0

    def __init__(self, *, num_tokens, node_dim, edge_dim, num_classes, hidden,
                 layers, dropout, token_dim, num_triples, args):
        super().__init__()
        reject_unknown_options(args, KNOWN_OPTIONS)
        dimensions = (num_tokens, node_dim, edge_dim, num_classes, hidden, token_dim, num_triples)
        if any(isinstance(value, bool) or int(value) != value or int(value) < 1
               for value in dimensions):
            raise ValueError("model dimensions must be positive integers")
        if isinstance(layers, bool) or int(layers) != 1:
            raise ValueError("CEI-GNN v2 supports layers=1 only")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")
        self.num_tokens, self.node_dim, self.edge_dim = map(int, (num_tokens, node_dim, edge_dim))
        self.num_classes, self.hidden = map(int, (num_classes, hidden))
        self.layers_count, self.dropout_rate = int(layers), float(dropout)
        self.token_dim, self.num_triples = int(token_dim), int(num_triples)
        self.num_relations = relation_count(args)
        self.pair_rank = method_option(args, "pair_rank", 16, int, minimum=1)
        self.pair_mode = method_option(args, "pair_mode", "product", str, choices=PAIR_MODES)
        self.network = PairEvidenceNetwork(
            num_tokens=self.num_tokens, node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_classes=self.num_classes, hidden=self.hidden, token_dim=self.token_dim,
            num_triples=self.num_triples, num_relations=self.num_relations,
            dropout=self.dropout_rate, pair_rank=self.pair_rank, pair_mode=self.pair_mode,
            num_node_types=len(NODE_KINDS))

    def _read(self, batch):
        for field in _INDEX_FIELDS:
            value = getattr(batch, field, None)
            if value is not None and (not torch.is_tensor(value) or value.dtype not in _INTEGER_DTYPES):
                raise ValueError(f"{field} must use an integer dtype")
        membership = getattr(batch, "visit_membership_index", None)
        if membership is None:
            raise ValueError("cei_gnn_v2 requires visit_membership_index")
        clinical = read_clinical_batch(
            batch, method="cei_gnn_v2", node_dim=self.node_dim, edge_dim=self.edge_dim,
            num_tokens=self.num_tokens, num_triples=self.num_triples,
            num_relations=self.num_relations)
        return clinical, membership.long()

    def continuous_inputs(self, batch) -> torch.Tensor:
        clinical, _ = self._read(batch)
        return self.network.continuous_inputs(clinical)

    def forward_continuous(self, features, edge_index, metadata, *, return_parts=False):
        clinical, membership = self._read(metadata)
        if edge_index.shape != clinical.edge_index.shape or not torch.equal(
                edge_index.to(clinical.edge_index.device), clinical.edge_index):
            raise ValueError("edge_index differs from the fixed metadata edge list")
        return self.network.forward_continuous(features, clinical.edge_index, clinical,
                                               membership, return_parts=return_parts)

    def forward(self, batch, *, epoch: int) -> MethodOutput:
        del epoch
        clinical, membership = self._read(batch)
        features = self.network.continuous_inputs(clinical)
        parts = self.network.forward_continuous(features, clinical.edge_index, clinical,
                                                membership, return_parts=True)
        logits = parts["logits"]
        auxiliary_loss = logits.sum() * 0.0
        pairs_per_graph = parts["pairs"].size(1) / max(int(clinical.graph_count), 1)
        return MethodOutput(logits=logits, auxiliary_loss=auxiliary_loss, diagnostics={
            "auxiliary_loss": diagnostic_float(auxiliary_loss, "cei_gnn_v2"),
            "pairs_per_graph": float(pairs_per_graph)})

    def on_epoch_start(self, epoch: int, train_loader) -> None:
        del train_loader
        if isinstance(epoch, bool) or int(epoch) != epoch or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")

    def optimizer_groups(self, args):
        del args
        return [{"params": list(self.parameters())}]

    def inactive_parameter_count(self) -> int:
        if self.pair_mode != "off":
            return 0
        net = self.network
        return int(net.pair_projection.weight.numel() + net.pair_vote.weight.numel()
                   + net.pair_vote.bias.numel() + net.pair_gate.numel())

    def run_config(self) -> dict:
        total, inactive = parameter_count(self), self.inactive_parameter_count()
        return {
            "method": "cei_gnn_v2",
            "adaptation_version": self.adaptation_version,
            "native_defaults": dict(self.runner_defaults),
            "effective_settings": {"pair_rank": self.pair_rank, "pair_mode": self.pair_mode},
            "architecture": {
                "node_dim": self.node_dim, "edge_dim": self.edge_dim,
                "num_tokens": self.num_tokens, "num_triples": self.num_triples,
                "num_relations": self.num_relations, "num_node_types": len(NODE_KINDS),
                "num_classes": self.num_classes, "hidden": self.hidden,
                "layers": self.layers_count, "dropout": self.dropout_rate,
                "token_dim": self.token_dim, "pair_rank": self.pair_rank,
                "parameter_count": total, "active_parameter_count": total - inactive,
                "inactive_parameter_count": inactive,
            },
        }


REGISTER = {"cei_gnn_v2": PairEvidenceAdapter}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_plugin_cei_gnn_v2.py tests/test_cei_gnn_v2_core.py tests/test_clinical_method_plugins.py tests/test_plugin_cei_gnn.py -q -p no:cacheprovider`
Expected: all passed.

- [ ] **Step 6: Commit green**

```bash
git add comparison/standardized/clinical_graph_v2/methods/plugin_cei_gnn_v2.py
git commit -m "green: V2-3 register cei_gnn_v2 plugin adapter"
```

---

### Task 4: Study plan and binding policy

**Files:**
- Create: `comparison/standardized/clinical_graph_v2/cei_v2_study.py`
- Test: `tests/test_cei_v2_study.py`

**Interfaces:**
- Consumes: `cei_pilot` as `pilot` (only `_REQUIRED_BINDINGS`, `_absolute_existing`); `train.parser`, `train.normalize_method_args`, `train.early_stopping_start_epoch`; `build_method`.
- Produces:
  - constants `STUDY_METHOD = "cei_gnn_v2"`, `MODES`, `SEEDS = (1234, 2025, 7)`, `SMOKE_BUDGET`, `FULL_BUDGET`, `PAIR_RANK = 16`
  - `Stage(name, argv, output, seed, budget, pair_mode)` (frozen dataclass)
  - `stage_specs() -> tuple[(name, budget, seed, mode), ...]`: 10 entries, smoke first, then seed-major `{mode}_seed{seed}`
  - `build_plan(*, artifact, targets, canonical, output_root) -> list[Stage]`
  - `validate_exact_plan(stages)`
  - `validate_v2_method_config(binding)`
  - `validate_v2_binding(binding, *, budget, seed, mode)`
  - `assert_arm_parity(bindings: dict[str, dict])`

- [ ] **Step 1: Write the failing tests**

```python
"""Synthetic contract tests for the CEI-GNN v2 pair study."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "comparison/standardized/clinical_graph_v2/cei_v2_study.py"


def _module():
    assert MODULE_PATH.is_file(), "CEI-GNN v2 study module must implement the locked planner"
    spec = importlib.util.spec_from_file_location("cei_v2_study_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _inputs(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    targets, canonical = tmp_path / "targets.csv", tmp_path / "canonical.json"
    targets.write_text("x")
    canonical.write_text("{}")
    return artifact, targets, canonical


@lru_cache(maxsize=None)
def _source(mode, seed, budget):
    from comparison.standardized.clinical_graph_v2 import train
    from comparison.standardized.clinical_graph_v2.methods import build_method
    from comparison.standardized.clinical_graph_v2.tensorize import PAYLOAD_WIDTH

    train_limit, dev_limit, epochs = budget
    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", "a", "--targets", "t", "--output", "o", "--method", "cei_gnn_v2",
        "--train-limit", str(train_limit), "--dev-limit", str(dev_limit),
        "--sample-seed", "1234", "--seed", str(seed), "--top-k-labels", "10",
        "--edges", "all", "--edge-direction", "forward", "--weights", "sqrt_inverse",
        "--selection-fold", "dev", "--final-eval", "none", "--epochs", str(epochs),
        "--patience", "40", "--method-option", f"pair_mode={mode}"]), parser)
    model = build_method("cei_gnn_v2", num_tokens=8, node_dim=3, edge_dim=PAYLOAD_WIDTH,
                         num_classes=2, hidden=args.hidden, layers=args.layers,
                         dropout=args.dropout, token_dim=args.token_dim, num_triples=2,
                         args=args)
    runner = {"lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
              "min_delta": args.min_delta,
              "early_stopping_start_epoch_index": train.early_stopping_start_epoch("cei_gnn_v2", model)}
    return json.dumps(model.run_config()), json.dumps(runner)


def _binding(mode="product", seed=2025, budget=(10000, 5000, 40), **updates):
    config_json, runner_json = _source(mode, seed, budget)
    config, runner = json.loads(config_json), json.loads(runner_json)
    arch = config["architecture"]
    train_limit, dev_limit, epochs = budget
    binding = {
        "method": "cei_gnn_v2", "method_config": config,
        "method_native_defaults": config["native_defaults"],
        "artifact_graphs_sha256": "g", "artifact_visit_membership_sha256": "m",
        "targets_sha256": "t", "target_binding_sha256": "tb", "label_order": ["A", "B"],
        "source_code": {"x.py": "h"}, "preprocessing_sha256": "p",
        "split_sample_ids_sha256": {"train": "tr", "dev": "dv", "validation": "va"},
        "seed": seed, "sample_seed": 1234, "selection_fold": "dev", "final_eval": "none",
        "train_limit": train_limit, "dev_limit": dev_limit, "epochs": epochs, "patience": 40,
        "test_evaluated": False, "parameter_count": arch["parameter_count"],
        "active_parameter_count": arch["active_parameter_count"], "weights": "sqrt_inverse",
        "edge_direction": "forward", "edges": "all", "top_k_labels": 10,
        "message_passing": True, "edge_payload": True, "num_classes": 2,
        "input_contract_version": "v", "vocabulary_size": 8, "node_dim": 3,
        "edge_dim": arch["edge_dim"], "hidden": arch["hidden"], "layers": arch["layers"],
        "dropout": arch["dropout"], "num_relations": arch["num_relations"],
        "num_meta_relations": 2, **runner,
    }
    binding.update(updates)
    return binding


def test_plan_has_exact_ten_dev_only_stages(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    expected = [("v2_smoke", (256, 128, 2), 1234, "product")] + [
        (f"{mode}_seed{seed}", (10000, 5000, 40), seed, mode)
        for seed in (1234, 2025, 7) for mode in ("product", "additive", "off")]
    assert [(s.name, s.budget, s.seed, s.pair_mode) for s in stages] == expected
    for stage in stages:
        argv = stage.argv
        assert argv[argv.index("--method") + 1] == "cei_gnn_v2"
        assert argv[argv.index("--seed") + 1] == str(stage.seed)
        assert argv[argv.index("--sample-seed") + 1] == "1234"
        assert argv[argv.index("--final-eval") + 1] == "none"
        assert argv[argv.index("--selection-fold") + 1] == "dev"
        assert argv[argv.index("--method-option") + 1] == f"pair_mode={stage.pair_mode}"
        assert Path(argv[argv.index("--output") + 1]) == Path(stage.output)
        assert not any("test" in token for token in argv[3:])


def test_plan_refuses_occupied_output_and_relative_paths(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    (tmp_path / "runs" / "off_seed7").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                          output_root=tmp_path / "runs")
    with pytest.raises(ValueError, match="absolute"):
        module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                          output_root="relative/runs")


def test_exact_plan_validator_rejects_edited_stage(tmp_path):
    module = _module()
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    module.validate_exact_plan(stages)
    edited = list(stages)
    edited[4] = dataclasses.replace(edited[4], argv=edited[4].argv[:-1] + ["pair_mode=product"])
    with pytest.raises(ValueError, match="argv"):
        module.validate_exact_plan(edited)
    with pytest.raises(ValueError, match="ten-stage"):
        module.validate_exact_plan(stages[:-1])


def test_binding_policy_accepts_source_config_and_rejects_drift():
    module = _module()
    module.validate_v2_binding(_binding("additive", 2025), budget=(10000, 5000, 40),
                               seed=2025, mode="additive")
    with pytest.raises(ValueError, match="seed"):
        module.validate_v2_binding(_binding("additive", 2025), budget=(10000, 5000, 40),
                                   seed=7, mode="additive")
    with pytest.raises(ValueError, match="pair_mode"):
        module.validate_v2_binding(_binding("additive", 2025), budget=(10000, 5000, 40),
                                   seed=2025, mode="product")
    with pytest.raises(ValueError, match="test_evaluated"):
        module.validate_v2_binding(_binding("off", 7, test_evaluated=True),
                                   budget=(10000, 5000, 40), seed=7, mode="off")
    drifted = _binding("product", 1234)
    drifted["method_config"]["architecture"]["pair_rank"] = 8
    with pytest.raises(ValueError, match="run_config"):
        module.validate_v2_binding(drifted, budget=(10000, 5000, 40), seed=1234, mode="product")


def test_arm_parity_allows_seed_and_mode_but_rejects_other_differences():
    module = _module()
    arms = {f"{mode}_seed{seed}": _binding(mode, seed)
            for seed in (1234, 2025, 7) for mode in ("product", "additive", "off")}
    assert module.assert_arm_parity(arms)
    arms["off_seed7"] = _binding("off", 7, split_sample_ids_sha256={
        "train": "other", "dev": "dv", "validation": "va"})
    with pytest.raises(ValueError, match="split_sample_ids_sha256"):
        module.assert_arm_parity(arms)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cei_v2_study.py -q -p no:cacheprovider`
Expected: 5 failed with `AssertionError: CEI-GNN v2 study module must implement the locked planner`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_cei_v2_study.py
git commit -m "red: V2-4 require exact ten-stage pair study plan and policy"
```

- [ ] **Step 4: Write minimal implementation**

```python
"""Bounded CEI-GNN v2 pair-interaction development study.

Default CLI output is a print-only plan. Training needs `--execute smoke` or
`--execute full`; validation and test folds are never evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from comparison.standardized.clinical_graph_v2 import cei_pilot as pilot

STUDY_METHOD = "cei_gnn_v2"
MODES = ("product", "additive", "off")
SEEDS = (1234, 2025, 7)
SAMPLE_SEED = 1234
PAIR_RANK = 16
SMOKE_BUDGET = (256, 128, 2)
FULL_BUDGET = (10000, 5000, 40)
FULL_STAGE_NAMES = tuple(f"{mode}_seed{seed}" for seed in SEEDS for mode in MODES)
_PARITY_FIELDS = (
    "artifact_graphs_sha256", "artifact_visit_membership_sha256", "targets_sha256",
    "target_binding_sha256", "label_order", "source_code", "preprocessing_sha256",
    "split_sample_ids_sha256", "sample_seed", "train_limit", "dev_limit", "epochs",
    "patience", "selection_fold", "final_eval", "weights", "edges", "edge_direction",
    "top_k_labels", "num_classes", "input_contract_version", "parameter_count", "lr",
    "weight_decay", "batch_size", "min_delta", "hidden", "layers", "dropout",
    "vocabulary_size", "node_dim", "edge_dim", "num_relations", "num_meta_relations",
)


@dataclass(frozen=True)
class Stage:
    name: str
    argv: list
    output: str
    seed: int
    budget: tuple
    pair_mode: str


def stage_specs():
    specs = [("v2_smoke", SMOKE_BUDGET, 1234, "product")]
    specs.extend((f"{mode}_seed{seed}", FULL_BUDGET, seed, mode)
                 for seed in SEEDS for mode in MODES)
    return tuple(specs)


def _stage(name, artifact, targets, canonical, output, budget, seed, mode):
    train_limit, dev_limit, epochs = budget
    argv = [
        sys.executable, "-m", "comparison.standardized.clinical_graph_v2.train",
        "--artifact", str(artifact), "--targets", str(targets),
        "--canonical", str(canonical), "--output", str(output),
        "--method", STUDY_METHOD,
        "--train-limit", str(train_limit), "--dev-limit", str(dev_limit),
        "--sample-seed", str(SAMPLE_SEED), "--seed", str(seed),
        "--top-k-labels", "10", "--edges", "all",
        "--edge-direction", "forward", "--weights", "sqrt_inverse",
        "--selection-fold", "dev", "--final-eval", "none",
        "--epochs", str(epochs), "--patience", "40",
        "--method-option", f"pair_mode={mode}",
    ]
    return Stage(name, argv, str(output), seed, budget, mode)


def build_plan(*, artifact, targets, canonical, output_root):
    """Build exactly the approved ten stages without opening input contents."""
    artifact = pilot._absolute_existing(artifact, "artifact", directory=True)
    targets = pilot._absolute_existing(targets, "targets")
    canonical = pilot._absolute_existing(canonical, "canonical")
    root = Path(output_root).expanduser()
    if not root.is_absolute():
        raise ValueError(f"output_root path must be absolute: {output_root}")
    root = root.resolve()
    stages = []
    for name, budget, seed, mode in stage_specs():
        output = root / name
        if output.exists():
            raise FileExistsError(f"Refusing occupied stage output {output}")
        stages.append(_stage(name, artifact, targets, canonical, output, budget, seed, mode))
    return stages


def validate_exact_plan(stages):
    """Reject any edit to the approved ten-stage plan."""
    specs = stage_specs()
    if len(stages) != len(specs):
        raise ValueError("execution requires the exact approved ten-stage plan")
    shared = None
    for stage, (name, budget, seed, mode) in zip(stages, specs):
        if (stage.name, tuple(stage.budget), stage.seed, stage.pair_mode) != (name, budget, seed, mode):
            raise ValueError(f"unauthorized stage tuple/order: {stage.name}")
        try:
            paths = tuple(stage.argv[stage.argv.index(flag) + 1]
                          for flag in ("--artifact", "--targets", "--canonical"))
        except (ValueError, IndexError) as error:
            raise ValueError(f"{name} argv is missing a required input path") from error
        if shared is None:
            shared = paths
        elif paths != shared:
            raise ValueError("stage plan input paths differ")
        expected = _stage(name, *paths, Path(stage.output), budget, seed, mode)
        if stage.argv != expected.argv:
            raise ValueError(f"unauthorized argv for {name}")


def validate_v2_method_config(binding):
    """Rebuild the adapter from source defaults and require an identical run_config."""
    config = binding.get("method_config")
    if not isinstance(config, dict) or config.get("method") != STUDY_METHOD \
            or binding.get("method") != STUDY_METHOD:
        raise ValueError("binding is not a cei_gnn_v2 run")
    settings = config.get("effective_settings")
    if not isinstance(settings, dict) or set(settings) != {"pair_rank", "pair_mode"}:
        raise ValueError("cei_gnn_v2 effective_settings must contain pair_rank and pair_mode")
    architecture = config.get("architecture")
    required = {"num_tokens", "node_dim", "edge_dim", "num_classes", "hidden", "layers",
                "dropout", "token_dim", "num_triples", "num_relations", "parameter_count",
                "active_parameter_count"}
    if not isinstance(architecture, dict) or not required.issubset(architecture):
        raise ValueError("cei_gnn_v2 architecture is missing dimensions/counts")
    top_level = {"num_tokens": "vocabulary_size", "node_dim": "node_dim", "edge_dim": "edge_dim",
                 "num_triples": "num_meta_relations", "num_classes": "num_classes",
                 "hidden": "hidden", "layers": "layers", "dropout": "dropout",
                 "num_relations": "num_relations"}
    for key, field in top_level.items():
        if binding.get(field) is None or architecture[key] != binding[field]:
            raise ValueError(f"runner binding {field} differs from method architecture")
    from comparison.standardized.clinical_graph_v2 import train
    from comparison.standardized.clinical_graph_v2.methods import build_method

    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", "<bound>", "--targets", "<bound>", "--output", "<bound>",
        "--method", STUDY_METHOD,
        "--train-limit", str(binding.get("train_limit")),
        "--dev-limit", str(binding.get("dev_limit")),
        "--sample-seed", str(binding.get("sample_seed")), "--seed", str(binding.get("seed")),
        "--top-k-labels", "10", "--edges", "all", "--edge-direction", "forward",
        "--weights", "sqrt_inverse", "--selection-fold", "dev", "--final-eval", "none",
        "--epochs", str(binding.get("epochs")), "--patience", str(binding.get("patience")),
        "--method-option", f"pair_mode={settings['pair_mode']}"]), parser)
    model = build_method(
        STUDY_METHOD, num_tokens=architecture["num_tokens"], node_dim=architecture["node_dim"],
        edge_dim=architecture["edge_dim"], num_classes=architecture["num_classes"],
        hidden=args.hidden, layers=args.layers, dropout=args.dropout,
        token_dim=args.token_dim, num_triples=architecture["num_triples"], args=args)
    if model.run_config() != config:
        raise ValueError("cei_gnn_v2 source-derived run_config drift")
    if binding.get("early_stopping_start_epoch_index") != train.early_stopping_start_epoch(
            STUDY_METHOD, model):
        raise ValueError("cei_gnn_v2 early-stopping schedule start differs from source")
    for field in ("lr", "weight_decay", "batch_size", "min_delta"):
        if binding.get(field) != getattr(args, field):
            raise ValueError(f"runner binding {field} differs from source defaults")


def validate_v2_binding(binding, *, budget, seed, mode):
    """Validate one stage binding against the approved study policy."""
    for field in pilot._REQUIRED_BINDINGS:
        if field not in binding or binding[field] is None:
            raise ValueError(f"binding missing required field: {field}")
    train_limit, dev_limit, epochs = budget
    expected = {
        "method": STUDY_METHOD, "train_limit": train_limit, "dev_limit": dev_limit,
        "epochs": epochs, "patience": 40, "seed": seed, "sample_seed": SAMPLE_SEED,
        "selection_fold": "dev", "final_eval": "none", "test_evaluated": False,
        "weights": "sqrt_inverse", "edges": "all", "edge_direction": "forward",
        "top_k_labels": 10,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise ValueError(f"study policy mismatch for {key}: expected {value!r}")
    settings = binding.get("method_config", {}).get("effective_settings", {})
    if settings.get("pair_mode") != mode:
        raise ValueError(f"pair_mode mismatch: expected {mode!r}")
    if settings.get("pair_rank") != PAIR_RANK:
        raise ValueError(f"pair_rank mismatch: expected {PAIR_RANK}")
    validate_v2_method_config(binding)


def assert_arm_parity(bindings):
    """All full arms share inputs, splits, source and capacity; only seed/mode differ."""
    items = sorted(bindings.items())
    if not items:
        raise ValueError("no arms to compare")
    reference_name, reference = items[0]

    def shared_architecture(binding):
        architecture = dict(binding["method_config"]["architecture"])
        for key in ("active_parameter_count", "inactive_parameter_count"):
            architecture.pop(key, None)
        return architecture

    for name, binding in items[1:]:
        for field in _PARITY_FIELDS:
            if binding.get(field) != reference.get(field):
                raise ValueError(f"arm parity differs: {field} ({name} vs {reference_name})")
        if shared_architecture(binding) != shared_architecture(reference):
            raise ValueError(f"arm parity differs: architecture ({name})")
    return True
```

`hashlib`, `subprocess`, `asdict` and `np` are imported now for Tasks 5 and 6.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cei_v2_study.py -q -p no:cacheprovider`
Expected: 5 passed.

- [ ] **Step 6: Commit green**

```bash
git add comparison/standardized/clinical_graph_v2/cei_v2_study.py
git commit -m "green: V2-4 add exact pair study plan and binding policy"
```

---

### Task 5: Replay and sequential executor

**Files:**
- Modify: `comparison/standardized/clinical_graph_v2/cei_v2_study.py` (append)
- Test: `tests/test_cei_v2_study.py` (append)

**Interfaces:**
- Consumes: Task 4 names; `pilot.capture_bindings`, `pilot._assert_runner_source_binding`, `pilot._validate_artifacts`, `pilot._load_json`, `pilot._write_journal`.
- Produces:
  - `_capture(stage) -> dict` (wraps `pilot.capture_bindings`; monkeypatchable)
  - `_identity(captured) -> tuple` = `(source_state_sha256, input_state_sha256)`
  - `replay_v2_stage(output_dir, binding, result) -> dict`: writes `replay.json` exclusively
  - `execute_plan(stages, *, journal_path, phase) -> int`, where `phase` is `"smoke"` or `"full"`
  - `validate_completed_study(output_root) -> dict[name, binding]` for the 9 full arms

- [ ] **Step 1: Write the failing tests (append)**

```python
def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_full_phase_requires_completed_smoke(tmp_path):
    module = _module()
    assert hasattr(module, "execute_plan"), "pair study executor is missing"
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    with pytest.raises(ValueError, match="smoke"):
        module.execute_plan(stages, journal_path=tmp_path / "journal.json", phase="full")
    assert not (tmp_path / "journal.json").exists()


def test_executor_journals_failure_and_launches_nothing_else(tmp_path, monkeypatch):
    module = _module()
    assert hasattr(module, "execute_plan"), "pair study executor is missing"
    artifact, targets, canonical = _inputs(tmp_path)
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=tmp_path / "runs")
    monkeypatch.setattr(module, "_capture", lambda stage: {
        "source_state_sha256": "s", "input_state_sha256": "i", "executable_sources": {}})
    launches = []

    def fail(argv, **kwargs):
        launches.append(argv)
        raise module.subprocess.CalledProcessError(2, argv)

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(SystemExit) as error:
        module.execute_plan(stages, journal_path=tmp_path / "journal.json", phase="smoke")
    assert error.value.code == 1
    journal = json.loads((tmp_path / "journal.json").read_text())
    assert journal["status"] == "failed" and journal["stages"]["v2_smoke"]["status"] == "failed"
    assert len(launches) == 1 and launches[0][-1] == "--execute"


def test_executor_refuses_source_drift_between_full_stages(tmp_path, monkeypatch):
    module = _module()
    assert hasattr(module, "execute_plan"), "pair study executor is missing"
    artifact, targets, canonical = _inputs(tmp_path)
    root = tmp_path / "runs"
    stages = module.build_plan(artifact=artifact, targets=targets, canonical=canonical,
                               output_root=root)
    _write_json(root / "v2_smoke" / "result.json", {"status": "completed"})
    identities = iter(["a", "a", "b"])
    monkeypatch.setattr(module, "_capture", lambda stage: {
        "source_state_sha256": next(identities), "input_state_sha256": "i",
        "executable_sources": {}})
    launches = []

    def fake_run(argv, **kwargs):
        launches.append(argv)
        output = Path(argv[argv.index("--output") + 1])
        binding = {"seed": 1234}
        _write_json(output / "binding.json", binding)
        _write_json(output / "result.json", {"status": "completed", "binding": binding,
                                             "metrics": None, "dev_metrics": {},
                                             "test_evaluated": False})
        return module.subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.pilot, "_assert_runner_source_binding", lambda *a: True)
    monkeypatch.setattr(module.pilot, "_validate_artifacts", lambda *a: None)
    monkeypatch.setattr(module, "validate_v2_binding", lambda *a, **k: None)
    monkeypatch.setattr(module, "replay_v2_stage", lambda *a: {"status": "verified"})
    with pytest.raises(SystemExit) as error:
        module.execute_plan(stages, journal_path=root / "journal.json", phase="full")
    assert error.value.code == 3
    assert len(launches) == 1
    journal = json.loads((root / "journal.json").read_text())
    assert journal["stages"]["product_seed1234"]["status"] == "bound"
    assert journal["stages"]["additive_seed1234"]["status"] == "refused"


def test_replay_reconstructs_v2_model_and_rejects_tampering(tmp_path, monkeypatch):
    module = _module()
    assert hasattr(module, "replay_v2_stage"), "pair study replay is missing"
    import hashlib
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    from comparison.standardized.clinical_graph_v2 import train
    from comparison.standardized.clinical_graph_v2.contracts import recursive_source_hashes, sample_ids_sha256
    from comparison.standardized.clinical_graph_v2.methods import build_method
    from comparison.standardized.clinical_graph_v2.schema import sha256
    from comparison.standardized.clinical_graph_v2.tensorize import (
        PAYLOAD_WIDTH, PREPROCESSING_VERSION, ClinicalGraphData, Scaler, Vocabulary,
        preprocessing_state)

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "graphs.jsonl").write_text("synthetic graph source\n")
    (artifact / "visit_membership.jsonl").write_text("synthetic membership\n")
    targets_path = tmp_path / "targets.csv"
    targets_path.write_text("synthetic targets\n")
    prep = {"preprocessing_version": PREPROCESSING_VERSION,
            "context_categories": {"gender": [], "race": [], "arrival_transport": []},
            "vocabulary": Vocabulary(["synthetic"], 1), "scaler": Scaler({}),
            "token_min_count": 1, "triples": ["visit|rel|complaint"], "triple_counts": {}}
    output = tmp_path / "product_seed2025"
    output.mkdir()
    (output / "preprocessing.json").write_text(json.dumps(preprocessing_state(prep), sort_keys=True))

    def row(sid, subject, label, scale):
        graph = ClinicalGraphData(
            x=torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, -0.1]]) * scale,
            edge_index=torch.tensor([[0, 0], [1, 2]]),
            edge_attr=torch.zeros((2, PAYLOAD_WIDTH)))
        graph.token = torch.tensor([0, 1, 1])
        graph.node_type = torch.tensor([1, 2, 5])
        graph.edge_relation = torch.zeros(2, dtype=torch.long)
        graph.edge_triple = torch.zeros(2, dtype=torch.long)
        graph.visit_membership_index = torch.tensor([[0, 0, 0], [0, 1, 2]])
        graph.num_visits = torch.tensor([1])
        graph.y = torch.tensor([label])
        graph.sample_id, graph.subject = sid, subject
        return graph

    splits = {"train": [row("train-1", "p1", 0, 1.0)],
              "dev": [row("dev-1", "p2", 0, 0.5), row("dev-2", "p3", 1, -1.0)],
              "validation": [row("validation-1", "p4", 1, 2.0)]}
    monkeypatch.setattr(train, "load_targets", lambda path: {"synthetic": True})
    monkeypatch.setattr(train, "select_top_labels", lambda targets, top_k: (targets, [0, 1], {}))
    monkeypatch.setattr(train, "build_dataset", lambda *args, **kwargs: (splits, prep))

    binding = _binding("product", 2025)
    torch.manual_seed(27)
    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", "a", "--targets", "t", "--output", "o", "--method", "cei_gnn_v2",
        "--method-option", "pair_mode=product"]), parser)
    model = build_method("cei_gnn_v2", num_tokens=2, node_dim=2, edge_dim=PAYLOAD_WIDTH,
                         num_classes=2, hidden=args.hidden, layers=args.layers,
                         dropout=args.dropout, token_dim=args.token_dim, num_triples=2,
                         args=args).eval()
    proba, labels = train.evaluate(model, DataLoader(splits["dev"], batch_size=128),
                                   torch.device("cpu"), epoch=0)
    torch.save(model.state_dict(), output / "best.pt")
    ids = np.asarray([graph.sample_id for graph in splits["dev"]])
    np.savez_compressed(output / "dev.npz", proba=proba, y=labels, sample_ids=ids)
    config = model.run_config()
    binding.update({
        "method_config": config, "method_native_defaults": config["native_defaults"],
        "parameter_count": config["architecture"]["parameter_count"],
        "active_parameter_count": config["architecture"]["active_parameter_count"],
        "vocabulary_size": 2, "node_dim": 2, "num_meta_relations": 2,
        "artifact": str(artifact), "artifact_graphs_sha256": sha256(artifact / "graphs.jsonl"),
        "artifact_visit_membership_file": "visit_membership.jsonl",
        "artifact_visit_membership_sha256": sha256(artifact / "visit_membership.jsonl"),
        "targets_path": str(targets_path), "targets_sha256": sha256(targets_path),
        "source_code": recursive_source_hashes(Path(train.__file__).parent),
        "preprocessing_sha256": sha256(output / "preprocessing.json"),
        "kept_label_indices": [0, 1], "token_min_count": 1, "dropped_relations": [],
        "rewired_relations": [], "min_prior_visits": 0,
        "split_sample_ids_sha256": {fold: sample_ids_sha256(g.sample_id for g in rows)
                                    for fold, rows in splits.items()},
        "selected_dev": {"epoch_index": 0, "prediction_sha256": hashlib.sha256(
            np.ascontiguousarray(proba).tobytes()).hexdigest()},
    })
    result = {"status": "completed", "binding": binding, "metrics": None,
              "dev_metrics": train.metrics(labels, proba, 2), "validation_evaluations": 0}
    proof = module.replay_v2_stage(output, binding, result)
    assert proof["exact_probabilities"] and proof["test_evaluated"] is False
    assert proof["validation_evaluated"] is False
    state = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    state["network.pair_vote.bias"] = state["network.pair_vote.bias"] + 0.5
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    for name in ("preprocessing.json", "dev.npz"):
        (tampered / name).write_bytes((output / name).read_bytes())
    torch.save(state, tampered / "best.pt")
    with pytest.raises(ValueError, match="probabilities"):
        module.replay_v2_stage(tampered, binding, result)
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python3 -m pytest tests/test_cei_v2_study.py -q -p no:cacheprovider`
Expected: Task 4 tests pass. The 4 new tests fail with `pair study executor is missing` or `pair study replay is missing`.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_cei_v2_study.py
git commit -m "red: V2-5 require pair study replay and fail-closed executor"
```

- [ ] **Step 4: Write minimal implementation (append)**

```python
def _capture(stage):
    return pilot.capture_bindings(stage)


def _identity(captured):
    # Seeds and pair modes differ between stages by design; the exact plan is
    # validated separately, so identity covers source bytes and input bytes only.
    return captured["source_state_sha256"], captured["input_state_sha256"]


def replay_v2_stage(output_dir, binding, result):
    """Rebuild the bound adapter and replay the saved dev predictions exactly."""
    output = Path(output_dir)
    if result.get("status") != "completed" or result.get("binding") != binding:
        raise ValueError("replay requires a completed result bound to binding.json")
    if (binding.get("selection_fold") != "dev" or binding.get("final_eval") != "none"
            or binding.get("test_evaluated") is not False or result.get("metrics") is not None):
        raise ValueError("replay is restricted to dev-selected, no-final-evaluation runs")
    if result.get("validation_evaluations") != 0 or (output / "validation.npz").exists():
        raise ValueError("validation evaluation is forbidden")
    import torch
    from types import SimpleNamespace
    from torch_geometric.loader import DataLoader
    from comparison.standardized.clinical_graph_v2 import train
    from comparison.standardized.clinical_graph_v2.contracts import recursive_source_hashes, sample_ids_sha256
    from comparison.standardized.clinical_graph_v2.methods import build_method
    from comparison.standardized.clinical_graph_v2.schema import sha256
    from comparison.standardized.clinical_graph_v2.tensorize import preprocessing_state

    source = recursive_source_hashes(Path(train.__file__).parent)
    if source != binding.get("source_code"):
        raise ValueError("replay source code differs from run binding")
    artifact = Path(binding["artifact"])
    if sha256(artifact / "graphs.jsonl") != binding["artifact_graphs_sha256"]:
        raise ValueError("replay graph artifact hash differs")
    if sha256(artifact / binding["artifact_visit_membership_file"]) != binding["artifact_visit_membership_sha256"]:
        raise ValueError("replay membership artifact hash differs")
    if sha256(binding["targets_path"]) != binding["targets_sha256"]:
        raise ValueError("replay targets hash differs")
    prep_path = output / "preprocessing.json"
    if sha256(prep_path) != binding["preprocessing_sha256"]:
        raise ValueError("replay preprocessing file hash differs")
    targets = train.load_targets(binding["targets_path"])
    targets, kept, _ = train.select_top_labels(targets, binding["top_k_labels"])
    if kept != binding["kept_label_indices"]:
        raise ValueError("replay top-label order differs")
    splits, prep = train.build_dataset(
        artifact, targets, binding["edges"], binding["train_limit"],
        binding["token_min_count"], binding["seed"],
        drop_relations=tuple(binding["dropped_relations"]),
        rewire_relations=tuple(binding["rewired_relations"]),
        min_prior_visits=binding["min_prior_visits"],
        edge_direction=binding["edge_direction"], dev_limit=binding["dev_limit"],
        sample_seed=binding["sample_seed"])
    if "test" in splits or {"train", "dev", "validation"} - set(splits):
        raise ValueError("replay dataset splits violate the train/dev-only contract")
    hashes = {fold: sample_ids_sha256(row.sample_id for row in rows) for fold, rows in splits.items()}
    if hashes != binding["split_sample_ids_sha256"]:
        raise ValueError("replayed split sample identities differ")
    if preprocessing_state(prep) != json.loads(prep_path.read_text()):
        raise ValueError("replayed preprocessing state differs")
    if {row.subject for row in splits["train"]} & {row.subject for row in splits["dev"]}:
        raise ValueError("replayed train/dev patient overlap")
    validate_v2_method_config(binding)
    config = binding["method_config"]
    architecture, settings = config["architecture"], config["effective_settings"]
    args = dict(binding)
    args["method_options"] = dict(settings)
    model = build_method(
        STUDY_METHOD, num_tokens=binding["vocabulary_size"], node_dim=binding["node_dim"],
        edge_dim=binding["edge_dim"], num_classes=binding["num_classes"],
        hidden=binding["hidden"], layers=binding["layers"], dropout=binding["dropout"],
        token_dim=architecture["token_dim"], num_triples=binding["num_meta_relations"],
        args=SimpleNamespace(**args))
    checkpoint = output / "best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    if sum(p.numel() for p in model.parameters()) != binding["parameter_count"]:
        raise ValueError("replayed parameter count differs")
    loader = DataLoader(splits["dev"], batch_size=binding["batch_size"], shuffle=False)
    probabilities, labels = train.evaluate(model, loader, torch.device("cpu"),
                                           epoch=binding["selected_dev"]["epoch_index"])
    metrics = train.metrics(labels, probabilities, binding["num_classes"])
    with np.load(output / "dev.npz", allow_pickle=False) as saved:
        ids = np.asarray([row.sample_id for row in splits["dev"]])
        if not np.array_equal(ids, saved["sample_ids"]):
            raise ValueError("replayed dev sample IDs differ")
        if not np.array_equal(labels, saved["y"]):
            raise ValueError("replayed dev labels differ")
        if not np.array_equal(probabilities, saved["proba"]):
            raise ValueError("checkpoint replay probabilities differ")
        prediction_hash = hashlib.sha256(np.ascontiguousarray(saved["proba"]).tobytes()).hexdigest()
    if prediction_hash != binding["selected_dev"]["prediction_sha256"]:
        raise ValueError("replayed probabilities do not match the selected checkpoint proof")
    if metrics != result.get("dev_metrics"):
        raise ValueError("replayed dev metrics differ")
    proof = {"status": "verified", "method": STUDY_METHOD,
             "pair_mode": settings["pair_mode"], "seed": binding["seed"],
             "checkpoint_sha256": sha256(checkpoint), "proba_sha256": prediction_hash,
             "split_sample_ids_sha256": hashes, "dev_count": int(len(labels)),
             "exact_probabilities": True, "exact_labels": True, "exact_sample_identity": True,
             "patient_disjoint": True, "validation_evaluated": False, "test_evaluated": False,
             "dev_metrics": metrics}
    proof_path = output / "replay.json"
    if proof_path.exists():
        existing = pilot._load_json(proof_path, "replay.json")
        if {k: existing.get(k) for k in proof} != proof:
            raise ValueError("existing replay proof differs")
        return existing
    with proof_path.open("x") as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return proof


def _check_stage_result(stage, binding, result):
    if result.get("status") != "completed" or result.get("binding") != binding:
        raise RuntimeError(f"{stage.name} did not produce a completed bound result")
    if result.get("metrics") is not None or not isinstance(result.get("dev_metrics"), dict):
        raise RuntimeError(f"{stage.name} violated the dev-only result contract")
    if result.get("test_evaluated", False) is not False:
        raise RuntimeError(f"{stage.name} evaluated the test fold")


def execute_plan(stages, *, journal_path, phase):
    """Run the smoke stage, or the nine full stages, sequentially and fail closed."""
    validate_exact_plan(stages)
    if phase not in ("smoke", "full"):
        raise ValueError("phase must be 'smoke' or 'full'")
    selected = list(stages[:1]) if phase == "smoke" else list(stages[1:])
    if phase == "full":
        smoke_result = Path(stages[0].output) / "result.json"
        if not smoke_result.is_file() or pilot._load_json(
                smoke_result, "smoke result.json").get("status") != "completed":
            raise ValueError("the smoke stage must complete before the full stages")
    journal_path = Path(journal_path)
    if journal_path.exists():
        raise FileExistsError(f"Refusing occupied journal {journal_path}")
    journal = {"status": "running", "phase": phase, "stages": {}}
    pilot._write_journal(journal_path, journal)
    expected, bindings = None, {}
    for stage in selected:
        output = Path(stage.output)
        if output.exists():
            journal["stages"][stage.name] = {"status": "refused", "reason": "occupied output"}
            journal["status"] = "failed"
            pilot._write_journal(journal_path, journal)
            raise SystemExit(2)
        before = _capture(stage)
        if expected is not None and _identity(before) != expected:
            journal["stages"][stage.name] = {"status": "refused", "reason": "source/input drift"}
            journal["status"] = "failed"
            pilot._write_journal(journal_path, journal)
            raise SystemExit(3)
        expected = expected or _identity(before)
        journal["stages"][stage.name] = {"status": "running", "preflight": before}
        pilot._write_journal(journal_path, journal)
        try:
            completed = subprocess.run(list(stage.argv) + ["--execute"], check=True,
                                       capture_output=True, text=True)
            after = _capture(stage)
            if _identity(after) != expected:
                raise RuntimeError("source/input drift detected after stage execution")
            binding = pilot._load_json(output / "binding.json", "binding.json")
            result = pilot._load_json(output / "result.json", "result.json")
            pilot._assert_runner_source_binding(after["executable_sources"], binding)
            validate_v2_binding(binding, budget=stage.budget, seed=stage.seed, mode=stage.pair_mode)
            _check_stage_result(stage, binding, result)
            pilot._validate_artifacts(output, binding, result)
            replay = replay_v2_stage(output, binding, result)
            bindings[stage.name] = binding
            journal["stages"][stage.name].update({
                "status": "bound", "postflight": after, "replay": replay,
                "stdout": completed.stdout, "stderr": completed.stderr})
        except Exception as error:
            journal["stages"][stage.name].update({"status": "failed", "error": str(error)})
            journal["status"] = "failed"
            pilot._write_journal(journal_path, journal)
            raise SystemExit(1) from error
        pilot._write_journal(journal_path, journal)
    if phase == "full":
        try:
            assert_arm_parity(bindings)
        except ValueError as error:
            journal["status"] = "failed"
            journal["validation_error"] = str(error)
            pilot._write_journal(journal_path, journal)
            raise SystemExit(1) from error
    for entry in journal["stages"].values():
        entry["status"] = "completed"
    journal["status"] = "completed"
    pilot._write_journal(journal_path, journal)
    return 0


def validate_completed_study(output_root):
    """Re-validate and replay all nine full arms; return their bindings."""
    root = Path(output_root).expanduser().resolve(strict=True)
    specs = {name: (budget, seed, mode) for name, budget, seed, mode in stage_specs()[1:]}
    bindings = {}
    for name, (budget, seed, mode) in specs.items():
        directory = root / name
        binding = pilot._load_json(directory / "binding.json", "binding.json")
        result = pilot._load_json(directory / "result.json", "result.json")
        validate_v2_binding(binding, budget=budget, seed=seed, mode=mode)
        _check_stage_result(Stage(name, [], str(directory), seed, budget, mode), binding, result)
        pilot._validate_artifacts(directory, binding, result)
        replay_v2_stage(directory, binding, result)
        bindings[name] = binding
    assert_arm_parity(bindings)
    return bindings
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cei_v2_study.py tests/test_cei_pilot.py -q -p no:cacheprovider`
Expected: all passed.

- [ ] **Step 6: Commit green**

```bash
git add comparison/standardized/clinical_graph_v2/cei_v2_study.py
git commit -m "green: V2-5 add pair study replay and sequential executor"
```

---

### Task 6: Preflight, analysis, decision rule and CLI

**Files:**
- Modify: `comparison/standardized/clinical_graph_v2/cei_v2_study.py` (append)
- Test: `tests/test_cei_v2_study.py` (append)

**Interfaces:**
- Consumes: Tasks 1–5; `within_visit_pairs`; `train.load_targets`, `train.select_top_labels`, `train.build_dataset`, `train.metrics`.
- Produces:
  - `pair_count_summary(rows) -> dict` with keys `graphs, mean, quantiles` (quantiles at `0, .25, .5, .75, .9, .99, 1`)
  - `step_time_ratio(rows, prep, *, batches=5) -> float`: v2 product forward+backward time divided by v1 `cei_gnn` time on the same batches. No optimizer step; models are discarded.
  - `preflight(*, artifact, targets, output_json) -> dict`: train-only pair counts, timing ratio, ETA
  - `weighted_macro_f1(y, pred, num_classes, weights=None) -> float`
  - `load_arm_predictions(output_root) -> (arms: dict[name, proba], y, subjects)`
  - `paired_bootstrap(arms, y, subjects, *, num_classes, resamples=1000, seed=2026) -> (point: dict[name, float], comparisons: dict)`
  - `decide(point, comparisons) -> dict` with `interaction_useful: bool`
  - `analyze(output_root, output_json) -> dict`
  - `main(argv=None)` with flags `--artifact --targets --canonical --output-root --execute {smoke,full} --journal --preflight OUT --analyze ROOT --analysis-output OUT`

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_weighted_macro_f1_matches_sklearn_including_empty_classes():
    module = _module()
    assert hasattr(module, "weighted_macro_f1"), "pair study analysis is missing"
    import numpy as np
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(1)
    y, pred = rng.integers(0, 4, 200), rng.integers(0, 4, 200)
    pred[pred == 3] = 2
    weights = rng.integers(0, 3, 200).astype(float)
    for w in (None, weights):
        expected = f1_score(y, pred, labels=np.arange(5), average="macro",
                            sample_weight=w, zero_division=0)
        assert abs(module.weighted_macro_f1(y, pred, 5, w) - expected) < 1e-12


def test_bootstrap_is_seeded_and_resamples_patients_as_clusters():
    module = _module()
    assert hasattr(module, "paired_bootstrap"), "pair study analysis is missing"
    import numpy as np

    rng = np.random.default_rng(4)
    y = rng.integers(0, 3, 60)
    subjects = np.repeat([f"p{i}" for i in range(20)], 3)
    arms = {}
    for seed in (1234, 2025, 7):
        for mode, noise in (("product", 0.2), ("additive", 0.6), ("off", 0.9)):
            proba = np.eye(3)[y] + rng.random((60, 3)) * noise * 2
            arms[f"{mode}_seed{seed}"] = proba / proba.sum(1, keepdims=True)
    point, comparisons = module.paired_bootstrap(arms, y, subjects, num_classes=3, resamples=200)
    again_point, again = module.paired_bootstrap(arms, y, subjects, num_classes=3, resamples=200)
    assert comparisons == again and point == again_point
    delta = comparisons["product_minus_additive"]
    low, high = delta["interval_95"]
    assert low <= delta["point"] <= high
    assert set(comparisons) == {"product_minus_additive", "product_minus_off", "additive_minus_off"}


def test_decision_rule_requires_every_seed_means_and_interval():
    module = _module()
    assert hasattr(module, "decide"), "pair study decision rule is missing"
    point = {f"{mode}_seed{seed}": value
             for seed in (1234, 2025, 7)
             for mode, value in (("product", 0.66), ("additive", 0.65), ("off", 0.64))}
    passing = {"product_minus_additive": {"point": 0.01, "interval_95": [0.001, 0.02]},
               "product_minus_off": {"point": 0.02, "interval_95": [0.005, 0.03]},
               "additive_minus_off": {"point": 0.01, "interval_95": [-0.001, 0.02]}}
    assert module.decide(point, passing)["interaction_useful"] is True
    straddling = json.loads(json.dumps(passing))
    straddling["product_minus_additive"]["interval_95"] = [-0.001, 0.02]
    assert module.decide(point, straddling)["interaction_useful"] is False
    one_loss = dict(point, product_seed7=0.649)
    decision = module.decide(one_loss, passing)
    assert decision["interaction_useful"] is False
    assert decision["checks"]["product_beats_additive_each_seed"] is False


def test_load_arm_predictions_requires_identical_dev_rows(tmp_path):
    module = _module()
    assert hasattr(module, "load_arm_predictions"), "pair study analysis is missing"
    import numpy as np

    ids = np.asarray(["a", "b"])
    for name in module.FULL_STAGE_NAMES:
        (tmp_path / name).mkdir()
        np.savez_compressed(tmp_path / name / "dev.npz", proba=np.full((2, 2), 0.5),
                            y=np.asarray([0, 1]), subjects=np.asarray(["p1", "p2"]),
                            sample_ids=ids)
    arms, y, subjects = module.load_arm_predictions(tmp_path)
    assert sorted(arms) == sorted(module.FULL_STAGE_NAMES) and y.tolist() == [0, 1]
    np.savez_compressed(tmp_path / "off_seed7" / "dev.npz", proba=np.full((2, 2), 0.5),
                        y=np.asarray([0, 1]), subjects=np.asarray(["p1", "p2"]),
                        sample_ids=np.asarray(["a", "c"]))
    with pytest.raises(ValueError, match="dev rows differ"):
        module.load_arm_predictions(tmp_path)


def test_preflight_reads_train_only_and_refuses_existing_output(tmp_path, monkeypatch):
    module = _module()
    assert hasattr(module, "preflight"), "pair study preflight is missing"
    from comparison.standardized.clinical_graph_v2 import train
    from tests.test_cei_gnn_v2_core import _graph

    rows = [_graph(seed=seed) for seed in range(6)]
    prep = {"vocabulary": [str(i) for i in range(8)], "triples": ["a", "b", "c"]}
    calls = []

    def fake_build(*args, **kwargs):
        calls.append(kwargs)
        return {"train": rows, "dev": rows[:2], "validation": rows[:1]}, prep

    monkeypatch.setattr(train, "load_targets", lambda path: {})
    monkeypatch.setattr(train, "select_top_labels", lambda targets, k: (targets, list(range(10)), {}))
    monkeypatch.setattr(train, "build_dataset", fake_build)
    artifact, targets, _ = _inputs(tmp_path)
    report = module.preflight(artifact=artifact, targets=targets,
                              output_json=tmp_path / "preflight.json")
    assert report["pair_counts"]["graphs"] == 6 and report["pair_counts"]["mean"] == 8.0
    assert report["step_time_ratio"] > 0 and report["test_tensors_loaded"] is False
    assert calls[0]["dev_limit"] == 5000 and calls[0]["sample_seed"] == 1234
    with pytest.raises(FileExistsError):
        module.preflight(artifact=artifact, targets=targets, output_json=tmp_path / "preflight.json")


def test_cli_default_prints_plan_without_launching(tmp_path, monkeypatch, capsys):
    module = _module()
    assert hasattr(module, "main"), "pair study CLI is missing"
    artifact, targets, canonical = _inputs(tmp_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    assert module.main(["--artifact", str(artifact), "--targets", str(targets),
                        "--canonical", str(canonical),
                        "--output-root", str(tmp_path / "runs")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "not_executed" and len(printed["stages"]) == 10
    assert not (tmp_path / "runs").exists()
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python3 -m pytest tests/test_cei_v2_study.py -q -p no:cacheprovider`
Expected: earlier tests pass. The 6 new tests fail with their `... is missing` assertion.

- [ ] **Step 3: Commit red**

```bash
git add tests/test_cei_v2_study.py
git commit -m "red: V2-6 require pair study preflight, analysis and decision rule"
```

- [ ] **Step 4: Write minimal implementation (append)**

```python
V1_MEAN_RUN_SECONDS = (127.7 + 123.9) / 2  # recorded CEI v1 pilot full-run total_seconds
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
COMPARISONS = {"product_minus_additive": ("product", "additive"),
               "product_minus_off": ("product", "off"),
               "additive_minus_off": ("additive", "off")}


def pair_count_summary(rows):
    from comparison.standardized.clinical_graph_v2.methods.cei_gnn_v2 import within_visit_pairs

    counts = np.asarray([within_visit_pairs(row.visit_membership_index, row.node_type,
                                            int(row.num_nodes)).size(1) for row in rows],
                        dtype=float)
    return {"graphs": int(counts.size), "mean": float(counts.mean()),
            "quantiles": {str(q): float(np.quantile(counts, q)) for q in QUANTILES}}


def step_time_ratio(rows, prep, *, batches=5):
    """Forward+backward seconds of v2 product over v1 on identical batches; no updates."""
    import time
    from types import SimpleNamespace
    import torch
    from torch_geometric.loader import DataLoader
    from comparison.standardized.clinical_graph_v2.methods import build_method

    def build(method, options):
        torch.manual_seed(1234)
        return build_method(method, num_tokens=len(prep["vocabulary"]),
                            node_dim=rows[0].x.shape[1], edge_dim=rows[0].edge_attr.shape[1],
                            num_classes=10, hidden=128, layers=1, dropout=0.3, token_dim=32,
                            num_triples=len(prep["triples"]) + 1,
                            args=SimpleNamespace(method_options=options))

    loader = list(DataLoader(rows, batch_size=128, shuffle=False))[:batches]
    seconds = {}
    for label, model in (("v1", build("cei_gnn", {"use_interactions": True})),
                         ("v2", build(STUDY_METHOD, {"pair_mode": "product"}))):
        model.train()
        start = time.perf_counter()
        for batch in loader:
            model.zero_grad(set_to_none=True)
            logits = model(batch, epoch=0).logits
            torch.nn.functional.cross_entropy(logits, batch.y.view(-1) % 10).backward()
        seconds[label] = time.perf_counter() - start
    return seconds["v2"] / max(seconds["v1"], 1e-9)


def preflight(*, artifact, targets, output_json):
    """Train-only pair counts and a timing ratio; no training, no dev/validation scores."""
    output = Path(output_json)
    if output.exists():
        raise FileExistsError(f"Refusing occupied preflight output {output}")
    from comparison.standardized.clinical_graph_v2 import train

    parser = train.parser()
    args = train.normalize_method_args(parser.parse_args([
        "--artifact", str(artifact), "--targets", str(targets), "--output", "<unused>",
        "--method", STUDY_METHOD]), parser)
    labels, _, _ = train.select_top_labels(train.load_targets(targets), 10)
    splits, prep = train.build_dataset(Path(artifact), labels, "all", FULL_BUDGET[0],
                                       args.token_min_count, 1234, edge_direction="forward",
                                       dev_limit=FULL_BUDGET[1], sample_seed=SAMPLE_SEED)
    if "test" in splits:
        raise ValueError("preflight must never build test tensors")
    rows = splits["train"]
    ratio = step_time_ratio(rows, prep)
    report = {"status": "verified",
              "scope": "train-only pair counts and step timing; no training or dev/validation scores",
              "pair_counts": pair_count_summary(rows), "step_time_ratio": ratio,
              "eta_minutes_nine_runs": round(9 * V1_MEAN_RUN_SECONDS * ratio / 60, 1),
              "eta_basis": "v1 recorded run seconds x measured v2/v1 step-time ratio; rough",
              "test_tensors_loaded": False, "validation_evaluated": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def weighted_macro_f1(y, pred, num_classes, weights=None):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    confusion = np.bincount(y * num_classes + pred, weights=w,
                            minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    tp = np.diag(confusion)
    fp, fn = confusion.sum(0) - tp, confusion.sum(1) - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros(num_classes), where=denominator > 0)
    return float(f1.mean())


def load_arm_predictions(output_root):
    root = Path(output_root)
    arms, reference = {}, None
    for name in FULL_STAGE_NAMES:
        with np.load(root / name / "dev.npz", allow_pickle=False) as saved:
            rows = (saved["y"].copy(), saved["subjects"].astype(str), saved["sample_ids"].astype(str))
            arms[name] = saved["proba"].copy()
        if reference is None:
            reference = rows
        elif not all(np.array_equal(a, b) for a, b in zip(rows, reference)):
            raise ValueError(f"dev rows differ across arms: {name}")
    return arms, reference[0], reference[1]


def paired_bootstrap(arms, y, subjects, *, num_classes, resamples=1000, seed=2026):
    patients, inverse = np.unique(np.asarray(subjects).astype(str), return_inverse=True)
    predictions = {name: np.asarray(proba).argmax(1) for name, proba in arms.items()}
    point = {name: weighted_macro_f1(y, pred, num_classes) for name, pred in predictions.items()}
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in COMPARISONS}
    for _ in range(resamples):
        drawn = rng.integers(0, len(patients), len(patients))
        weights = np.bincount(drawn, minlength=len(patients))[inverse].astype(float)
        scores = {name: weighted_macro_f1(y, pred, num_classes, weights)
                  for name, pred in predictions.items()}
        for key, (first, second) in COMPARISONS.items():
            samples[key].append(float(np.mean([scores[f"{first}_seed{s}"] - scores[f"{second}_seed{s}"]
                                               for s in SEEDS])))
    comparisons = {}
    for key, (first, second) in COMPARISONS.items():
        values = np.asarray(samples[key])
        comparisons[key] = {
            "point": float(np.mean([point[f"{first}_seed{s}"] - point[f"{second}_seed{s}"]
                                    for s in SEEDS])),
            "interval_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "resamples": int(resamples), "seed": int(seed)}
    return point, comparisons


def decide(point, comparisons):
    per_seed = {str(s): {m: point[f"{m}_seed{s}"] for m in MODES} for s in SEEDS}
    means = {m: float(np.mean([per_seed[str(s)][m] for s in SEEDS])) for m in MODES}
    checks = {
        "product_beats_additive_each_seed": all(
            per_seed[str(s)]["product"] > per_seed[str(s)]["additive"] for s in SEEDS),
        "product_mean_above_both_controls": (means["product"] > means["additive"]
                                             and means["product"] > means["off"]),
        "product_minus_additive_lower_bound_above_zero":
            comparisons["product_minus_additive"]["interval_95"][0] > 0,
    }
    return {"per_seed": per_seed, "seed_means": means, "checks": checks,
            "interaction_useful": all(checks.values())}


def analyze(output_root, output_json):
    output = Path(output_json)
    if output.exists():
        raise FileExistsError(f"Refusing occupied analysis output {output}")
    bindings = validate_completed_study(output_root)
    arms, y, subjects = load_arm_predictions(output_root)
    num_classes = next(iter(bindings.values()))["num_classes"]
    point, comparisons = paired_bootstrap(arms, y, subjects, num_classes=num_classes)
    for name, binding in bindings.items():
        recorded = binding["selected_dev"]["metric_value"]
        if abs(point[name] - recorded) > 1e-6:
            raise ValueError(f"recomputed macro-F1 differs from the bound result: {name}")
    report = {"status": "analyzed", "decision": decide(point, comparisons),
              "comparisons": comparisons, "dev_macro_f1": point,
              "dev_rows": int(len(y)), "patients": int(len(np.unique(subjects))),
              "test_evaluated": False, "validation_evaluated": False}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact")
    result.add_argument("--targets")
    result.add_argument("--canonical")
    result.add_argument("--output-root")
    result.add_argument("--execute", choices=("smoke", "full"))
    result.add_argument("--journal")
    result.add_argument("--preflight", metavar="OUTPUT_JSON")
    result.add_argument("--analyze", metavar="OUTPUT_ROOT")
    result.add_argument("--analysis-output", metavar="OUTPUT_JSON")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.analyze:
        if not args.analysis_output:
            parser().error("--analyze requires --analysis-output")
        print(json.dumps(analyze(args.analyze, args.analysis_output), indent=2, sort_keys=True))
        return 0
    if args.preflight:
        if not (args.artifact and args.targets):
            parser().error("--preflight requires --artifact and --targets")
        print(json.dumps(preflight(artifact=args.artifact, targets=args.targets,
                                   output_json=args.preflight), indent=2, sort_keys=True))
        return 0
    if not all((args.artifact, args.targets, args.canonical, args.output_root)):
        parser().error("--artifact, --targets, --canonical and --output-root are required")
    if args.execute == "full":
        # The smoke directory exists by then; plan the remaining stages against it.
        root = Path(args.output_root).expanduser().resolve()
        stages = [_stage(name, pilot._absolute_existing(args.artifact, "artifact", directory=True),
                         pilot._absolute_existing(args.targets, "targets"),
                         pilot._absolute_existing(args.canonical, "canonical"),
                         root / name, budget, seed, mode)
                  for name, budget, seed, mode in stage_specs()]
    else:
        stages = build_plan(artifact=args.artifact, targets=args.targets,
                            canonical=args.canonical, output_root=args.output_root)
    if not args.execute:
        print(json.dumps({"status": "not_executed", "stages": [asdict(s) for s in stages]},
                         indent=2, sort_keys=True))
        return 0
    journal = args.journal or str(Path(args.output_root).resolve() / f"journal_{args.execute}.json")
    return execute_plan(stages, journal_path=journal, phase=args.execute)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cei_v2_study.py -q -p no:cacheprovider`
Expected: all passed.

- [ ] **Step 6: Commit green, run the regression set, push, review**

```bash
git add comparison/standardized/clinical_graph_v2/cei_v2_study.py
git commit -m "green: V2-6 add pair study preflight, analysis and decision rule"
python3 -m pytest tests/test_cei_gnn_v2_core.py tests/test_plugin_cei_gnn_v2.py tests/test_cei_v2_study.py tests/test_plugin_cei_gnn.py tests/test_cei_pilot.py tests/test_cei_graphxai.py tests/test_clinical_method_plugins.py tests/test_real_graphxai.py -q -p no:cacheprovider
git push origin feature/cei-v2-pairs
```

Expected: all passed (the pre-change base passed 63 of the v1/plugin tests). Then:
- Supervisor RED replay: check out each `red:` commit's tests against its parent and confirm assertion failures.
- Independent read-only review of the branch diff against spec sections 3–7.
- Fix findings with new red/green commits, then push again.

---

### Task 7: Preflight and smoke run — APPROVAL GATE 1

Do not start without an explicit user "yes" to "preflight + smoke".

**Files:** outputs only, under the new directory `comparison/standardized/clinical_runs_cei_v2_pairs_20260928/` in the MAIN checkout (ignored run path). No source changes.

- [ ] **Step 1: Print the plan (no execution)**

Run from the worktree root:
```bash
M=/Users/necatifurkancolak/AI-Workplace/Projects/current/MedGNN
python3 -m comparison.standardized.clinical_graph_v2.cei_v2_study \
  --artifact $M/comparison/standardized/event_inputs/clinical_graph_v3_membership_max6_20260923 \
  --targets $M/comparison/standardized/event_inputs/first_recorded_lab_all_visits_v2_targets_local_v2_max6/targets.csv \
  --canonical $M/comparison/canonical_split.json \
  --output-root $M/comparison/standardized/clinical_runs_cei_v2_pairs_20260928
```
Expected: JSON with `"status": "not_executed"` and 10 stages. Verify `targets.csv` is the path bound in `clinical_runs_cei_pilot_20260928/cei_candidate/binding.json` (`targets_path`); stop if it differs.

- [ ] **Step 2: Preflight**

Same inputs plus `--preflight $M/comparison/standardized/clinical_runs_cei_v2_pairs_20260928/preflight.json`.
Expected: pair-count quantiles, `step_time_ratio`, `eta_minutes_nine_runs`.

- [ ] **Step 3: Smoke**

Same inputs plus `--execute smoke`.
Expected: `journal_smoke.json` status `completed`; `v2_smoke/replay.json` verified.

- [ ] **Step 4: Report to the user**

Report pair counts, smoke seconds, the ETA for 9 runs and any anomaly. Stop and ask for APPROVAL GATE 2.

---

### Task 8: Full study, analysis and report — APPROVAL GATE 2

Do not start without an explicit user "yes" to the 9 full runs.

**Files:**
- Outputs under `comparison/standardized/clinical_runs_cei_v2_pairs_20260928/`
- Create `docs/cei-gnn-v2-pairs-2026-09-28.md`

- [ ] **Step 1: Run the 9 full stages**

Same inputs plus `--execute full` (background, notify on exit). Poll only through the journal; do not restart a running job.
Expected: `journal_full.json` status `completed`; 9 `replay.json` verified.

- [ ] **Step 2: Analyze**

`--analyze $M/.../clinical_runs_cei_v2_pairs_20260928 --analysis-output $M/.../clinical_runs_cei_v2_pairs_20260928/analysis.json`
Expected: `analysis.json` with `decision.interaction_useful` true or false.

- [ ] **Step 3: Write the report**

`docs/cei-gnn-v2-pairs-2026-09-28.md` contains:
- the frozen contract (spec section 4)
- per-seed dev macro-F1 table for 3 modes x 3 seeds, plus seed means
- the three bootstrap comparisons with 95% intervals
- the three decision checks and the verdict, quoted verbatim from `analysis.json`
- pair counts, parameters (total/active) and seconds
- limits: dev selection only; 40-epoch cap; GraphXAI does not see pairs; `temporal_clean=false`; no test or validation scores; no clinical or causal claim

Do not add a win/loss narrative beyond the pre-registered rule.

- [ ] **Step 4: Commit and push**

```bash
git add docs/cei-gnn-v2-pairs-2026-09-28.md
git commit -m "docs: report CEI-GNN v2 pair study against pre-registered rule"
git push origin feature/cei-v2-pairs
```

## Self-Review Notes

- Spec coverage:

  | Spec section | Task |
  |---|---|
  | 3.1–3.4 | Tasks 1–2 |
  | 3.5 (wrapper parity) | Task 3 |
  | 4–5 | Tasks 4–5, 7–8 |
  | 6 | Task 6 (`decide`, `paired_bootstrap`) |
  | 7 | tests in Tasks 1–6 |
  | 8 | preflight ETA in Task 7 |
  | 9 | Global Constraints |

- Known deviation from spec section 7, "Study plan ... never passes test options": the argv test asserts that no token after the module name contains `test`. The plan does not pass `--final-eval test`.
- `validate_v2_method_config` rebuilds with `--top-k-labels 10` and the fixed edge policy. Budget/seed come from the binding and are policy-checked in `validate_v2_binding`.
