"""Synthetic tests for the independent CEI-GNN method."""


def test_cei_is_independent_registered_method():
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY

    assert "cei_gnn" in METHOD_REGISTRY, "independent evidence-interaction method absent"
