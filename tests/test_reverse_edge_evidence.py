"""GraphXAI improvement plan item #5: do not count a reverse edge as separate
evidence from its forward counterpart, and report relation-type vs numeric
payload contribution separately.

Two independent pieces, tested independently:
- `reversible_edge_pairs` (tensorize.py): a pure function identifying which
  edges are forward/reverse pairs of the same clinical fact, from
  `edge_relation` alone.
- `aggregate_reverse_edge_pairs` (shared/lib/graphxai_standardized.py): the
  mean-collapse applied to GNNExplainer's edge mask before node reduction.
- `relation_vs_payload_contribution` (fidelity_v2.py): the zero-and-measure
  decomposition, exercised against a real adapter.
"""
import pytest
import torch

from core.explain import fidelity_v2 as fv2
from core.tensorize import (
    ALL_RELATIONS, REVERSIBLE_RELATIONS, RELATION_INDEX, reversible_edge_pairs,
)
from core.explain.graphxai_standardized import aggregate_reverse_edge_pairs
from tests.test_clinical_graphxai_wrapper import NUM_RELATIONS, one_graph, wrap_generic


def _reverse_id(relation):
    return len(ALL_RELATIONS) + REVERSIBLE_RELATIONS.index(relation)


def test_reversible_edge_pairs_matches_construction_order():
    # Mirrors encode_graph's own construction: the forward block first (not
    # every forward edge is reversible -- has_visit already has an explicit
    # inverse pair, so it gets no 'rev:' duplicate), then the reverse block in
    # the same relative order as the reversible edges within the forward block.
    forward = [
        RELATION_INDEX['has_visit'],
        RELATION_INDEX['reports_complaint'],
        RELATION_INDEX['observed_vital'],
    ]
    reverse = [_reverse_id('reports_complaint'), _reverse_id('observed_vital')]
    edge_relation = torch.tensor(forward + reverse)
    assert reversible_edge_pairs(edge_relation) == [(1, 3), (2, 4)]


def test_reversible_edge_pairs_forward_only_graph_returns_empty():
    edge_relation = torch.tensor(
        [RELATION_INDEX['reports_complaint'], RELATION_INDEX['observed_vital']]
    )
    assert reversible_edge_pairs(edge_relation) == []


def test_reversible_edge_pairs_raises_on_mismatched_counts():
    # A malformed (non-encode_graph) tensor -- one reverse edge but two
    # reversible forward edges -- must fail loudly, not silently mispair.
    forward = [RELATION_INDEX['reports_complaint'], RELATION_INDEX['observed_vital']]
    reverse = [_reverse_id('reports_complaint')]
    edge_relation = torch.tensor(forward + reverse)
    with pytest.raises(ValueError, match='differ'):
        reversible_edge_pairs(edge_relation)


def test_aggregate_reverse_edge_pairs_collapses_to_the_mean():
    edge_imp = torch.tensor([0.9, 0.1, 0.5, 0.5, 0.5])
    collapsed = aggregate_reverse_edge_pairs(edge_imp, [(0, 1)])
    assert torch.allclose(collapsed, torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5]))
    # Original must be untouched (caller elsewhere still needs it unmodified).
    assert torch.allclose(edge_imp, torch.tensor([0.9, 0.1, 0.5, 0.5, 0.5]))


def test_aggregate_reverse_edge_pairs_is_a_noop_without_pairs():
    edge_imp = torch.tensor([0.9, 0.1, 0.5])
    assert torch.equal(aggregate_reverse_edge_pairs(edge_imp, None), edge_imp)
    assert torch.equal(aggregate_reverse_edge_pairs(edge_imp, []), edge_imp)


@pytest.mark.parametrize("name", ("gsat", "protgnn", "graphcare"))
def test_relation_vs_payload_contribution_runs_on_a_real_adapter(name):
    torch.manual_seed(71)
    graph = one_graph()
    if name == "graphcare":
        from core.explain.graphxai_wrapper import (
            GraphCareGraphXAIWrapper,
        )
        from tests.test_clinical_graphxai_wrapper import build_adapter
        from tests.test_clinical_method_adapters import EDGE_DIM, NODE_DIM, NUM_TOKENS, NUM_TRIPLES

        model = build_adapter("graphcare")
        model.eval()
        wrapper = GraphCareGraphXAIWrapper(
            model, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
            num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
        )
        x = wrapper.set_context(graph)
    else:
        wrapper, x = wrap_generic(name, graph)

    with torch.no_grad():
        prediction = int(wrapper(x, graph.edge_index).argmax(-1).item())
    result = fv2.relation_vs_payload_contribution(
        wrapper, x, graph.edge_index, graph.edge_attr, NUM_RELATIONS, prediction,
    )
    assert result["edge_count"] == graph.edge_index.size(1)
    assert torch.isfinite(torch.tensor(result["relation_type_prob_drop"]))
    assert torch.isfinite(torch.tensor(result["numeric_payload_prob_drop"]))


def test_relation_vs_payload_contribution_rejects_edge_attr_without_payload_columns():
    torch.manual_seed(73)
    graph = one_graph()
    wrapper, x = wrap_generic("gsat", graph)
    with pytest.raises(ValueError, match="payload column"):
        fv2.relation_vs_payload_contribution(
            wrapper, x, graph.edge_index, graph.edge_attr,
            graph.edge_attr.size(1), 0,  # num_relation_columns == full width
        )
