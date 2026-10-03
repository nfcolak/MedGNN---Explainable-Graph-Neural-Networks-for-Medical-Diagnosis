"""Regression tests for CEI-GNN v2 pair-mode RNG parity."""
import torch

from test_cei_gnn_v2_core import _graph, _metadata
from cei.cei_gnn_v2 import PairEvidenceNetwork


def _networks():
    result = {}
    for mode in ("product", "additive", "off"):
        torch.manual_seed(123)
        result[mode] = PairEvidenceNetwork(
            num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8,
            token_dim=4, num_triples=4, num_relations=15, dropout=0.3,
            pair_rank=4, pair_mode=mode, num_node_types=8).train()
    return result


def _three_forwards(network):
    graph = _graph()
    metadata = _metadata(graph)
    features = network.continuous_inputs(metadata)
    torch.manual_seed(947)
    states, outputs = [], []
    for _ in range(3):
        outputs.append(network.forward_continuous(
            features, metadata.edge_index, metadata, graph.visit_membership_index,
            return_parts=True))
        states.append(torch.get_rng_state().clone())
    return states, outputs


def test_rng_states_and_node_edge_parts_match_across_pair_modes():
    runs = {mode: _three_forwards(network) for mode, network in _networks().items()}
    for index in range(3):
        expected_state = runs["product"][0][index]
        expected = runs["product"][1][index]
        for mode in ("additive", "off"):
            assert torch.equal(runs[mode][0][index], expected_state), f"RNG state differs: {mode}, forward {index + 1}"
            torch.testing.assert_close(runs[mode][1][index]["node_contributions"], expected["node_contributions"])
            torch.testing.assert_close(runs[mode][1][index]["edge_contributions"], expected["edge_contributions"])


def test_off_mode_pair_parts_are_zero_and_logits_reconstruct_in_training():
    network = _networks()["off"]
    graph = _graph()
    metadata = _metadata(graph)
    features = network.continuous_inputs(metadata)
    torch.manual_seed(947)
    parts = network.forward_continuous(features, metadata.edge_index, metadata,
                                       graph.visit_membership_index, return_parts=True)
    assert torch.count_nonzero(parts["pair_contributions"]) == 0
    rebuilt = (parts["bias"] + parts["node_contributions"].sum(0)
               + parts["edge_contributions"].sum(0) + parts["pair_contributions"].sum(0))
    torch.testing.assert_close(parts["logits"][0], rebuilt, rtol=1e-5, atol=1e-5)


def test_next_epoch_shuffle_matches_across_pair_modes():
    runs = {mode: _three_forwards(network) for mode, network in _networks().items()}
    permutations = {}
    for mode, (states, _) in runs.items():
        torch.set_rng_state(states[-1])
        permutations[mode] = torch.randperm(50)
    assert torch.equal(permutations["product"], permutations["additive"])
    assert torch.equal(permutations["product"], permutations["off"])
