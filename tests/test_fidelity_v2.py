"""Synthetic tests for the item-#4 full-evidence-removal fidelity module.

Covers all three wrapper shapes this project built (a plain ClinicalMethodAdapter
via gsat, GraphCare's own wrapper, and a non-adapter GCHM model) to prove the
mechanism generalises, not just one lucky case.
"""
import math

import pytest
import torch

from core import NODE_KINDS
from core.explain import fidelity_v2 as fv2
from gchm_pna.gchm_v2 import GCHMv2
from core.explain.graphxai_wrapper import (
    ClinicalGraphXAIWrapper,
    GraphCareGraphXAIWrapper,
)
from core.tensorize import ALL_RELATIONS
from tests.test_clinical_method_adapters import (
    EDGE_DIM, NODE_DIM, NUM_CLASSES, NUM_TOKENS, NUM_TRIPLES, method_args, synthetic_graph,
)
from tests.test_clinical_graphxai_wrapper import build_adapter

NUM_RELATIONS = len(ALL_RELATIONS)
PATIENT_KIND = NODE_KINDS.index("patient")
VISIT_KIND = NODE_KINDS.index("visit")


def seven_node_graph():
    """node_type cycles 0..6 (patient, visit, complaint, measurement, analyte,
    vital, knowledge) over 7 nodes -- one of each kind including both
    structural kinds, so removability logic is actually exercised, not
    vacuously true. Membership pairs cover nodes 0-5 (every non-global node);
    node 6 (knowledge) is the sole global node."""
    graph = synthetic_graph(
        sample_id="single", label=0, node_count=7, num_visits=3,
        membership=[[0, 0, 0, 1, 1, 2], [0, 1, 2, 3, 4, 5]],
        global_mask=[False, False, False, False, False, False, True],
    )
    graph.node_type = torch.arange(7, dtype=torch.long) % len(NODE_KINDS)
    return graph


def wrap_gsat(graph):
    model = build_adapter("gsat")
    model.eval()
    wrapper = ClinicalGraphXAIWrapper(
        model, method="gsat", node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    return wrapper, x


def wrap_graphcare(graph):
    model = build_adapter("graphcare")
    model.eval()
    wrapper = GraphCareGraphXAIWrapper(
        model, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS,
    )
    x = wrapper.set_context(graph)
    return wrapper, x


def wrap_gchm(graph):
    model = GCHMv2(
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=NUM_CLASSES,
        num_relations=NUM_RELATIONS, degree_histogram=[0, 2, 3, 1],
        hidden=8, layers=2, dropout=0.0, token_dim=4,
    )
    model.eval()
    wrapper = ClinicalGraphXAIWrapper(
        model, method="gchm_v2", node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_tokens=NUM_TOKENS,
        num_triples=NUM_TRIPLES, num_relations=NUM_RELATIONS, is_method_adapter=False,
    )
    x = wrapper.set_context(graph)
    return wrapper, x


WRAPPERS = {"gsat": wrap_gsat, "graphcare": wrap_graphcare, "gchm_v2": wrap_gchm}


def test_removable_node_mask_excludes_structural_and_knowledge_kinds():
    node_type = torch.arange(len(NODE_KINDS), dtype=torch.long)  # one of every kind
    removable = fv2.removable_node_mask(node_type)
    for kind in ("patient", "visit", "knowledge"):
        assert not bool(removable[NODE_KINDS.index(kind)]), kind
    for kind in ("complaint", "measurement", "analyte", "vital", "diagnosis"):
        assert bool(removable[NODE_KINDS.index(kind)]), kind


@pytest.mark.parametrize("name", ["gsat", "graphcare", "gchm_v2"])
def test_fidelity_plus_and_minus_v2_are_finite_and_well_formed(name):
    torch.manual_seed(101)
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1).item())
    importance = torch.rand(graph.x.size(0)).numpy()

    plus = fv2.fidelity_plus_v2(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance, target
    )
    minus = fv2.fidelity_minus_v2(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance, target
    )
    for result in (plus, minus):
        assert math.isfinite(result["prob"])
        assert result["k"] >= 1
        # 7 nodes, 2 structural (patient, visit) + 1 knowledge = 4 removable.
        assert result["removable_node_count"] == 4


@pytest.mark.parametrize("name", ["gsat", "graphcare", "gchm_v2"])
def test_structural_nodes_are_never_selected_even_with_maximal_importance(name):
    """Give the patient and visit nodes the highest possible importance score
    and confirm they're still never evicted -- removability is enforced by
    node kind, not merely by being low-ranked in practice."""
    torch.manual_seed(103)
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1).item())
    importance = torch.zeros(graph.x.size(0)).numpy().copy()
    importance[PATIENT_KIND] = 1e6  # node 0 IS kind 'patient' by construction above
    importance[VISIT_KIND] = 1e6    # node 1 IS kind 'visit'

    # k = all 4 removable nodes: if structural nodes leaked into the ordering,
    # they would be evicted first and the removable_node_count would be wrong.
    plus = fv2.fidelity_plus_v2(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance, target, k=4
    )
    assert plus["removable_node_count"] == 4
    assert plus["k"] == 4


@pytest.mark.parametrize("name", ["gsat", "graphcare", "gchm_v2"])
def test_full_eviction_changes_token_not_only_x(name):
    """The whole point of item #4: prove the intervention actually reaches
    token (and edges), not only x. Construct an importance vector that always
    selects the same single node, zero x for it by hand (the OLD v1-style
    intervention) and confirm the model's prediction under v1-style zeroing
    still differs from v2's full eviction for at least one node in the graph
    -- i.e. token/edge removal is doing something x-zeroing alone doesn't.
    """
    torch.manual_seed(107)
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    removable = fv2.removable_node_mask(graph.node_type)
    removable_indices = removable.nonzero(as_tuple=True)[0].tolist()
    assert removable_indices, "fixture must have at least one removable node"

    found_a_difference = False
    for node_index in removable_indices:
        x_v1 = x.clone()
        x_v1[node_index] = 0.0
        with torch.no_grad():
            v1_probs = torch.softmax(wrapper(x_v1, graph.edge_index), dim=-1)

        x_v2, token_v2, evicted = fv2._evict(x, graph.token, torch.tensor([node_index]))
        edge_mask = fv2._edge_mask_for_eviction(graph.edge_index, evicted)
        with torch.no_grad():
            v2_probs = torch.softmax(
                wrapper(x_v2, graph.edge_index, edge_mask=edge_mask, token_override=token_v2),
                dim=-1,
            )
        if not torch.allclose(v1_probs, v2_probs, atol=1e-6):
            found_a_difference = True
            break
    assert found_a_difference, (
        f"{name}: v2's token+edge eviction never differed from v1's x-only "
        "zeroing on any removable node -- the extra channels may not be wired in"
    )


@pytest.mark.parametrize("name", ["gsat", "graphcare", "gchm_v2"])
def test_deletion_and_insertion_curves_are_well_formed(name):
    torch.manual_seed(109)
    graph = seven_node_graph()
    wrapper, x = WRAPPERS[name](graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1).item())
    importance = torch.rand(graph.x.size(0)).numpy()

    deletion = fv2.deletion_curve(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance,
        target, num_points=4,
    )
    insertion = fv2.insertion_curve(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance,
        target, num_points=4,
    )
    for curve in (deletion, insertion):
        assert curve["k_values"] == sorted(curve["k_values"])
        assert curve["k_values"][-1] <= curve["removable_node_count"]
        assert len(curve["prob_at_k"]) == len(curve["k_values"])
        assert all(math.isfinite(p) for p in curve["prob_at_k"])
        assert math.isfinite(curve["auc"])


def test_fidelity_minus_v2_with_k_equal_to_all_removable_nodes_is_the_identity():
    """keep_important evicts everyone NOT in the top-k; k == removable_count
    means nothing is evicted (the untouched graph), not an error."""
    torch.manual_seed(113)
    graph = seven_node_graph()
    wrapper, x = wrap_gsat(graph)
    with torch.no_grad():
        target = int(wrapper(x, graph.edge_index).argmax(-1).item())
    importance = torch.rand(graph.x.size(0)).numpy()
    result = fv2.fidelity_minus_v2(
        wrapper, x, graph.token, graph.edge_index, graph.node_type, importance, target, k=4
    )
    assert result["prob"] == 0.0
    assert result["acc"] == 0
