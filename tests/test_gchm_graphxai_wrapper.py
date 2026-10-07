"""Synthetic, in-memory GraphXAI-wrapper tests for GCHMv2/GCHMv3.

Neither is a ClinicalMethodAdapter (not in METHOD_REGISTRY; train.py builds
and calls them through a separate path -- see gchm_v2.py/gchm_v3.py), so
they're wrapped via ClinicalGraphXAIWrapper(..., is_method_adapter=False)
rather than build_method(). Reuses test_clinical_method_adapters.py's exact
synthetic-graph helper and dimension conventions.
"""
import math

import pytest
import torch

from gchm_pna.gchm_v2 import GCHMv2
from gchm_pna.gchm_v3 import GCHMv3
from core.explain.graphxai_wrapper import ClinicalGraphXAIWrapper
from core.tensorize import ALL_RELATIONS
from tests.test_clinical_method_adapters import (
    EDGE_DIM, NODE_DIM, NUM_CLASSES, NUM_TOKENS, NUM_TRIPLES, synthetic_graph,
)

NUM_RELATIONS = len(ALL_RELATIONS)
DEGREE_HISTOGRAM = [0, 2, 3, 1]  # arbitrary nonzero in-degree histogram, only feeds a log-norm constant


def one_graph():
    return synthetic_graph(
        sample_id="single", label=0, node_count=5, num_visits=3,
        membership=[[0, 0, 1, 2], [0, 1, 2, 3]],
        global_mask=[False, False, False, False, True],
    )


def gchm_v2(**overrides):
    defaults = dict(
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=NUM_CLASSES,
        num_relations=NUM_RELATIONS, degree_histogram=DEGREE_HISTOGRAM,
        hidden=8, layers=2, dropout=0.0, token_dim=4,
    )
    defaults.update(overrides)
    return GCHMv2(**defaults)


def gchm_v3(**overrides):
    # wide=False: GCHMv3's wide path indexes data.x at fixed columns matching
    # the real tensorize.py layout (VALUE_COLUMN/HAS_VALUE_COLUMN at
    # len(NODE_KINDS)/+1), which the generic synthetic x (width NODE_DIM,
    # plain noise, shared across every other method's tests) doesn't satisfy.
    # The wide path never touches edges either way, so disabling it doesn't
    # weaken anything this file actually tests (the edge_mask hook in the
    # shared HubGatedPNALayer, identical for v2 and v3).
    defaults = dict(
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=NUM_CLASSES,
        num_relations=NUM_RELATIONS, degree_histogram=DEGREE_HISTOGRAM,
        hidden=8, layers=2, dropout=0.0, token_dim=4, edge_dropout=0.0, wide=False,
    )
    defaults.update(overrides)
    return GCHMv3(**defaults)


MODEL_FACTORIES = {"gchm_v2": gchm_v2, "gchm_v3": gchm_v3}


def wrap(name, graph, model=None):
    model = model if model is not None else MODEL_FACTORIES[name]()
    model.eval()
    wrapper = ClinicalGraphXAIWrapper(
        model, method=name, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS, is_method_adapter=False,
    )
    x = wrapper.set_context(graph)
    return wrapper, x


@pytest.mark.parametrize("name", ["gchm_v2", "gchm_v3"])
def test_verify_passes(name):
    torch.manual_seed(71)
    graph = one_graph()
    wrapper, x = wrap(name, graph)
    ok, max_diff = wrapper.verify(x, graph.edge_index)
    assert ok, f"{name}: verify() failed, max_diff={max_diff}"
    assert max_diff < 1e-5


@pytest.mark.parametrize("name", ["gchm_v2", "gchm_v3"])
def test_all_ones_edge_mask_reproduces_unmasked_output_exactly(name):
    """The PNA aggregator's weighted mean/std and min/max-sentinel paths only
    activate when edge_mask is not None -- this is the numerical proof they
    reduce to the exact original unweighted statistics at mask=1, not just an
    approximation. (edge_mask=None is covered by test_verify_passes already;
    this exercises the *other* branch landing on the same answer.)"""
    torch.manual_seed(73)
    graph = one_graph()
    model = MODEL_FACTORIES[name]()
    model.eval()
    wrapper, x = wrap(name, graph, model=model)
    baseline = wrapper(x, graph.edge_index)
    ones_mask = torch.ones(graph.edge_index.size(1))
    masked = wrapper(x, graph.edge_index, edge_mask=ones_mask)
    assert torch.allclose(masked, baseline, atol=1e-5), f"{name}: all-ones mask changed output"


@pytest.mark.parametrize("name", ["gchm_v2", "gchm_v3"])
def test_edge_mask_hook_changes_prediction_and_is_differentiable(name):
    torch.manual_seed(79)
    graph = one_graph()
    model = MODEL_FACTORIES[name]()
    model.eval()
    wrapper, x = wrap(name, graph, model=model)

    baseline = wrapper(x, graph.edge_index)
    edge_count = graph.edge_index.size(1)
    mask = torch.full((edge_count,), 0.5, requires_grad=True)
    masked = wrapper(x, graph.edge_index, edge_mask=mask)

    target_class = int(baseline.argmax(-1).item())
    grad = torch.autograd.grad(masked[0, target_class], mask, retain_graph=True)[0]
    assert grad is not None, f"{name}: edge_mask gradient disconnected"
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0.0

    zeroed = wrapper(x, graph.edge_index, edge_mask=torch.zeros(edge_count))
    assert not torch.allclose(zeroed, baseline, atol=1e-4), f"{name}: zero mask had no effect"
    assert torch.isfinite(zeroed).all()


@pytest.mark.parametrize("name", ["gchm_v2", "gchm_v3"])
def test_real_explainers_run_through_the_wrapper(name):
    from core.explain.graphxai_standardized import explain_algorithms

    torch.manual_seed(83)
    graph = one_graph()
    wrapper, x = wrap(name, graph)
    batch_index = torch.zeros(graph.x.size(0), dtype=torch.long)
    result = explain_algorithms(wrapper, x, graph.edge_index, batch=batch_index, steps=4, epochs=20)
    for algo in ("GradExplainer", "IntegratedGradExplainer", "GNNExplainer"):
        assert result[algo]["status"] == "success", f"{name}/{algo}: {result[algo]}"
        importance = result[algo]["node_explanation"]["node_importance"]
        assert len(importance) == graph.x.size(0)
        assert all(math.isfinite(value) for value in importance)
    assert result["GNNExplainer"]["provenance"]["edge_gradient_verified"] is True


def test_builtin_node_importance_not_available_for_gchm():
    """No .explain() on either model -- raises clearly rather than fabricating one."""
    graph = one_graph()
    wrapper, x = wrap("gchm_v2", graph)
    with pytest.raises(NotImplementedError, match="no per-node built-in explanation"):
        wrapper.builtin_node_importance(x, graph.edge_index)
