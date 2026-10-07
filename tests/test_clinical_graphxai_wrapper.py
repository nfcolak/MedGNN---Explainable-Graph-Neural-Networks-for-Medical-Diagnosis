"""Synthetic, in-memory GraphXAI-wrapper tests for every clinical_graph_v2
method, via the one shared ClinicalGraphXAIWrapper (GraphCare is the sole
exception -- GraphCareGraphXAIWrapper, below -- it needs extra
visit-membership context the others don't).

Reuses test_clinical_method_adapters.py's exact synthetic-graph helper and
dimension conventions (same NODE_DIM/EDGE_DIM/etc., same build_method() path)
so there is one definition of what a valid synthetic clinical graph looks
like, not two drifting copies.
"""
import pytest
import torch

from core.explain.graphxai_wrapper import (
    ClinicalGraphXAIWrapper,
    GraphCareGraphXAIWrapper,
)
from core.registry import build_method
from core.tensorize import ALL_RELATIONS
from tests.test_clinical_method_adapters import (
    EDGE_DIM,
    NODE_DIM,
    NUM_CLASSES,
    NUM_TOKENS,
    NUM_TRIPLES,
    method_args,
    synthetic_graph,
)

GENERIC_METHODS = (
    "gsat", "protgnn",
)
NUM_RELATIONS = len(ALL_RELATIONS)


def one_graph():
    return synthetic_graph(
        sample_id="single",
        label=0,
        node_count=5,
        num_visits=3,
        membership=[[0, 0, 1, 2], [0, 1, 2, 3]],
        global_mask=[False, False, False, False, True],
    )


def build_adapter(name, **overrides):
    return build_method(
        name,
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_classes=NUM_CLASSES, hidden=8, layers=2, dropout=0.0,
        token_dim=4, num_triples=NUM_TRIPLES, args=method_args(**overrides),
    )


def wrap_generic(name, graph, model=None):
    model = model if model is not None else build_adapter(name)
    model.eval()
    wrapper = ClinicalGraphXAIWrapper(
        model, method=name, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_tokens=NUM_TOKENS, num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    return wrapper, x


@pytest.mark.parametrize("name", GENERIC_METHODS)
def test_verify_passes_for_every_generic_method(name):
    torch.manual_seed(hash(name) % (2**31))
    graph = one_graph()
    wrapper, x = wrap_generic(name, graph)
    ok, max_diff = wrapper.verify(x, graph.edge_index)
    assert ok, f"{name}: verify() failed, max_diff={max_diff}"
    assert max_diff < 1e-5


@pytest.mark.parametrize("name", GENERIC_METHODS)
def test_edge_mask_hook_changes_prediction_and_is_differentiable(name):
    torch.manual_seed(31)
    graph = one_graph()
    model = build_adapter(name)
    model.eval()
    wrapper, x = wrap_generic(name, graph, model=model)

    baseline = wrapper(x, graph.edge_index)
    edge_count = graph.edge_index.size(1)
    ones_mask = torch.ones(edge_count, requires_grad=True)
    masked = wrapper(x, graph.edge_index, edge_mask=ones_mask)
    assert torch.allclose(masked, baseline, atol=1e-5), f"{name}: all-ones mask changed output"

    target_class = int(baseline.argmax(-1).item())
    grad = torch.autograd.grad(masked[0, target_class], ones_mask, retain_graph=True)[0]
    assert grad is not None, f"{name}: edge_mask gradient disconnected"
    assert torch.isfinite(grad).all()

    zeroed = wrapper(x, graph.edge_index, edge_mask=torch.zeros(edge_count))
    assert not torch.allclose(zeroed, baseline, atol=1e-4), f"{name}: zero mask had no effect"


@pytest.mark.parametrize("name", ["gsat"])
def test_builtin_node_importance_available_where_explain_exists(name):
    torch.manual_seed(37)
    graph = one_graph()
    model = build_adapter(name)
    model.eval()
    wrapper, x = wrap_generic(name, graph, model=model)
    importance = wrapper.builtin_node_importance(x, graph.edge_index)
    assert importance.shape[0] == graph.x.size(0)
    assert torch.isfinite(importance).all()


def _connected(nodes, edge_index):
    nodes = set(nodes)
    adjacency = {n: set() for n in nodes}
    for a, b in edge_index.t().tolist():
        if a in nodes and b in nodes:
            adjacency[a].add(b)
            adjacency[b].add(a)
    seen, stack = set(), [next(iter(nodes))]
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(adjacency[n] - seen)
    return seen == nodes


def test_protgnn_builtin_is_its_own_prototype_subgraph():
    torch.manual_seed(43)
    graph = one_graph()
    wrapper, x = wrap_generic("protgnn", graph)
    importance = wrapper.builtin_node_importance(x, graph.edge_index)
    detail = wrapper.builtin_detail()
    adapter = wrapper.adapter

    assert importance.shape[0] == graph.x.size(0)
    assert set(importance.tolist()) <= {0.0, 1.0}
    members = [int(i) for i in importance.nonzero().view(-1)]
    assert members == detail["subgraph_nodes"]
    assert adapter.min_atoms <= len(members) <= adapter.max_atoms
    assert _connected(members, graph.edge_index)

    with torch.no_grad():
        assert detail["predicted_class"] == int(wrapper(x, graph.edge_index).argmax(-1))
    # The prototype used belongs to the predicted class and is that class's
    # most-contributing prototype -- the model's own choice, not ours.
    own = [i for i in range(adapter.num_prototypes) if int(adapter.prototype_class_ids[i]) == detail["predicted_class"]]
    assert detail["prototype_index"] in own
    assert detail["prototype_index"] == max(own, key=lambda i: detail["prototype_contributions"][i])
    assert detail["kind"] == "prototype_mcts_subgraph"


def test_protgnn_builtin_is_deterministic_and_does_not_disturb_the_model():
    torch.manual_seed(47)
    graph = one_graph()
    wrapper, x = wrap_generic("protgnn", graph)
    before = wrapper(x, graph.edge_index).detach().clone()
    first, second = wrapper.builtin_detail(), wrapper.builtin_detail()
    assert first == second
    assert torch.equal(wrapper(x, graph.edge_index), before)
    assert not wrapper.adapter.training


def test_protgnn_builtin_refuses_instead_of_inventing_when_no_subgraph_exists():
    graph = one_graph()
    model = build_adapter("protgnn", min_atoms=6, max_atoms=6)   # graph has only 5 nodes
    wrapper, x = wrap_generic("protgnn", graph, model=model)
    assert wrapper.builtin_detail()["subgraph_nodes"] == []
    with pytest.raises(RuntimeError, match="no connected subgraph"):
        wrapper.builtin_node_importance(x, graph.edge_index)


def test_wrappers_without_a_builtin_still_refuse_clearly():
    from core.explain.graphxai_wrapper import ClinicalGraphXAIWrapper

    graph = one_graph()
    wrapper = ClinicalGraphXAIWrapper(
        build_adapter("gsat"), method="not_a_builtin_method", node_dim=NODE_DIM,
        edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS, num_triples=NUM_TRIPLES,
        num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    with pytest.raises(NotImplementedError, match="no per-node built-in explanation"):
        wrapper.builtin_node_importance(x, graph.edge_index)


def test_graphcare_verify_passes():
    torch.manual_seed(41)
    graph = one_graph()
    model = build_adapter("graphcare")
    model.eval()
    wrapper = GraphCareGraphXAIWrapper(
        model, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    ok, max_diff = wrapper.verify(x, graph.edge_index)
    assert ok, f"graphcare: verify() failed, max_diff={max_diff}"
    assert max_diff < 1e-5


def test_graphcare_edge_mask_hook_changes_prediction_and_is_differentiable():
    torch.manual_seed(43)
    graph = one_graph()
    model = build_adapter("graphcare")
    model.eval()
    wrapper = GraphCareGraphXAIWrapper(
        model, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)

    baseline = wrapper(x, graph.edge_index)
    edge_count = graph.edge_index.size(1)
    ones_mask = torch.ones(edge_count, requires_grad=True)
    masked = wrapper(x, graph.edge_index, edge_mask=ones_mask)
    assert torch.allclose(masked, baseline, atol=1e-5)

    target_class = int(baseline.argmax(-1).item())
    grad = torch.autograd.grad(masked[0, target_class], ones_mask, retain_graph=True)[0]
    assert grad is not None
    assert torch.isfinite(grad).all()

    zeroed = wrapper(x, graph.edge_index, edge_mask=torch.zeros(edge_count))
    assert not torch.allclose(zeroed, baseline, atol=1e-4)


def test_graphcare_builtin_node_importance():
    torch.manual_seed(47)
    graph = one_graph()
    model = build_adapter("graphcare")
    model.eval()
    wrapper = GraphCareGraphXAIWrapper(
        model, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    importance = wrapper.builtin_node_importance(x, graph.edge_index)
    assert importance.numel() > 0
    assert torch.isfinite(importance).all()


@pytest.mark.parametrize("name", GENERIC_METHODS)
def test_real_grad_and_integrated_grad_explainers_run_through_the_wrapper(name):
    from core.explain.graphxai_standardized import explain_algorithms

    torch.manual_seed(53)
    graph = one_graph()
    wrapper, x = wrap_generic(name, graph)
    batch_index = torch.zeros(graph.x.size(0), dtype=torch.long)
    result = explain_algorithms(wrapper, x, graph.edge_index, batch=batch_index, steps=4, epochs=15)
    for algo in ("GradExplainer", "IntegratedGradExplainer", "GNNExplainer"):
        assert result[algo]["status"] == "success", f"{name}/{algo}: {result[algo]}"
    assert result["GNNExplainer"]["provenance"]["edge_gradient_verified"] is True


def test_graphcare_real_explainers_run_through_the_wrapper():
    from core.explain.graphxai_standardized import explain_algorithms

    torch.manual_seed(59)
    graph = one_graph()
    model = build_adapter("graphcare")
    model.eval()
    wrapper = GraphCareGraphXAIWrapper(
        model, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    batch_index = torch.zeros(graph.x.size(0), dtype=torch.long)
    result = explain_algorithms(wrapper, x, graph.edge_index, batch=batch_index, steps=4, epochs=15)
    for algo in ("GradExplainer", "IntegratedGradExplainer", "GNNExplainer"):
        assert result[algo]["status"] == "success", f"graphcare/{algo}: {result[algo]}"
    assert result["GNNExplainer"]["provenance"]["edge_gradient_verified"] is True


def test_edge_index_mismatch_is_refused():
    """Generic _batch_with() validation, not specific to any one method --
    exercised once via gsat rather than duplicated across all eight."""
    torch.manual_seed(61)
    graph = one_graph()
    wrapper, x = wrap_generic("gsat", graph)
    wrong_edges = graph.edge_index.flip(0)
    with pytest.raises(ValueError, match="edge_index must match"):
        wrapper(x, wrong_edges)


def test_external_edge_mask_rejects_wrong_shape():
    """Generic adapter-level validation (every adapter checks edge_mask.numel()
    against its own edge count), exercised once via gsat."""
    torch.manual_seed(63)
    graph = one_graph()
    wrapper, x = wrap_generic("gsat", graph)
    with pytest.raises(ValueError, match="one value per edge"):
        wrapper(x, graph.edge_index, edge_mask=torch.ones(graph.edge_index.size(1) + 1))


def test_explain_algorithms_isolates_one_algorithms_failure_from_the_others(monkeypatch):
    """Regression test for the explain_algorithms() isolation fix itself: one
    algorithm failing must not discard the other two algorithms' already-
    computed, perfectly valid results along with it. Uses gsat as the vehicle;
    the fix lives in shared/lib/graphxai_standardized.py and applies to every
    method, old graph and new alike."""
    import core.explain.graphxai_standardized as graphxai_standardized

    def _broken_get_explanation_graph(self, *args, **kwargs):
        raise RuntimeError("synthetic forced failure")

    monkeypatch.setattr(
        graphxai_standardized.GradExplainer,
        "get_explanation_graph",
        _broken_get_explanation_graph,
    )

    torch.manual_seed(29)
    graph = one_graph()
    wrapper, x = wrap_generic("gsat", graph)
    batch_index = torch.zeros(graph.x.size(0), dtype=torch.long)
    result = graphxai_standardized.explain_algorithms(
        wrapper, x, graph.edge_index, batch=batch_index, steps=4, epochs=15
    )
    assert result["GradExplainer"]["status"] == "failed"
    assert "synthetic forced failure" in result["GradExplainer"]["error"]
    for name in ("IntegratedGradExplainer", "GNNExplainer"):
        assert result[name]["status"] == "success", result[name]
