"""Synthetic-only tests for the ProtoNode clinical adapter."""
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch

from comparison.standardized.clinical_graph_v2 import train
from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY, build_method
from comparison.standardized.clinical_graph_v2.tensorize import ALL_RELATIONS, PAYLOAD_WIDTH, ClinicalGraphData

NODE_DIM = 6
EDGE_DIM = len(ALL_RELATIONS) + PAYLOAD_WIDTH
NUM_TOKENS = 12
NUM_TRIPLES = 5
NUM_CLASSES = 2


def graph(label=0, node_count=2):
    x = torch.arange(node_count * NODE_DIM, dtype=torch.float32).view(node_count, NODE_DIM) / 10
    edge_index = torch.tensor([[i for i in range(node_count)], [(i + 1) % node_count for i in range(node_count)]])
    edge_count = edge_index.size(1)
    data = ClinicalGraphData(x=x, edge_index=edge_index,
        edge_attr=torch.zeros(edge_count, EDGE_DIM),
        edge_relation=torch.zeros(edge_count, dtype=torch.long),
        edge_triple=torch.zeros(edge_count, dtype=torch.long),
        token=torch.arange(1, node_count + 1) % NUM_TOKENS,
        node_type=torch.arange(node_count) % 8, y=torch.tensor([label]))
    return data


def batch():
    return Batch.from_data_list([graph(0, 2), graph(1, 3)])


def make_protonode(**overrides):
    if "protonode" not in METHOD_REGISTRY:
        pytest.fail("protonode adapter not registered")
    settings = dict(protonode_warm_epochs=0, protonode_proj_epochs=1,
                    protonode_proj_interval=10, protonode_nearest_graphs=2,
                    protonode_prototypes_per_class=2)
    settings.update(overrides)
    return build_method("protonode", num_tokens=NUM_TOKENS, node_dim=NODE_DIM,
        edge_dim=EDGE_DIM, num_classes=NUM_CLASSES, hidden=8, layers=1,
        dropout=0.0, token_dim=4, num_triples=NUM_TRIPLES, args=Namespace(**settings))


@pytest.mark.parametrize("readout", ["both", "node_max", "graph_mean"])
def test_protonode_registered_constructs_and_has_finite_forward_backward(readout):
    model = make_protonode(protonode_readout=readout)
    output = model(batch(), epoch=0)
    assert output.logits.shape == (2, NUM_CLASSES)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.auxiliary_loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in output.diagnostics.values())
    (output.logits.sum() + output.auxiliary_loss).backward()
    assert model.prototype_vectors.grad is not None
    assert torch.isfinite(model.prototype_vectors.grad).all()


def test_node_readout_is_closest_node_while_graph_mean_uses_the_mean():
    model = make_protonode(protonode_readout="node_max", protonode_prototypes_per_class=1)
    with torch.no_grad():
        model.prototype_vectors.zero_()
    nodes = torch.tensor([[1.0] + [0.0] * 7, [4.0] + [0.0] * 7])
    graph_ids = torch.zeros(2, dtype=torch.long)
    d1, _ = model._node_min_distances(nodes, graph_ids, 1)
    far_added = torch.cat([nodes, torch.tensor([[20.0] + [0.0] * 7])])
    d2, _ = model._node_min_distances(far_added, torch.zeros(3, dtype=torch.long), 1)
    expected = torch.tensor([[1.0, 1.0]])
    assert torch.equal(d1, expected)
    assert torch.equal(d2, expected)
    node_activation = model._similarity(d1)
    assert node_activation[0, 0].item() == pytest.approx(torch.log(torch.tensor(2.0 / 1.0001)).item())
    graph_distance = model._distances(nodes.mean(0, keepdim=True))
    extended_mean_distance = model._distances(far_added.mean(0, keepdim=True))
    assert not torch.equal(graph_distance, extended_mean_distance)


def test_wide_votes_start_zero_receive_gradients_and_can_be_disabled():
    torch.manual_seed(22)
    model = make_protonode()
    assert torch.count_nonzero(model.wide_token_votes.weight) == 0
    out = model(batch(), epoch=0)
    (out.logits.sum() + out.auxiliary_loss).backward()
    assert model.wide_token_votes.weight.grad is not None
    assert model.wide_token_votes.weight.grad.abs().sum() > 0
    disabled = make_protonode(protonode_no_wide=True)
    assert not hasattr(disabled, "wide_token_votes")
    assert sum(p.numel() for p in disabled.parameters()) < sum(p.numel() for p in model.parameters())


def test_cluster_separation_and_cross_class_losses_match_hand_computation():
    model = make_protonode(protonode_readout="node_max", protonode_prototypes_per_class=1,
                           protonode_cluster_weight=1.0, protonode_separation_weight=1.0,
                           protonode_margin=5.0, protonode_no_wide=True)
    with torch.no_grad():
        model.prototype_vectors.zero_()
        model.prototype_classifier.weight.copy_(torch.tensor([[1.0, -0.5], [-0.5, 1.0]]))
    labels = batch().y.view(-1)
    encoded = model._encode(batch())
    node_distances, _ = model._node_min_distances(encoded[0], encoded[2], encoded[3])
    own = model.prototype_class_ids[None, :] == labels[:, None]
    expected_cluster = node_distances.masked_fill(~own, torch.inf).min(1).values.mean()
    wrong = node_distances.masked_fill(own, torch.inf).min(1).values
    expected_separation = torch.relu(model.margin - wrong).mean()
    output = model(batch(), epoch=0)
    expected_cross_l1 = (model.prototype_classifier.weight * ~model.prototype_class_identity.t().bool()).abs().sum()
    assert output.diagnostics["cluster_loss"] == pytest.approx(expected_cluster.item())
    assert output.diagnostics["separation_loss"] == pytest.approx(expected_separation.item())
    assert output.diagnostics["cross_class_l1"] == pytest.approx(expected_cross_l1.item())
    assert model.prototype_classifier.weight[0, 1].item() == pytest.approx(-0.5)


def test_projection_is_train_only_and_copies_a_same_class_encoded_node():
    model = make_protonode(protonode_warm_epochs=2, protonode_proj_epochs=2,
                           protonode_prototypes_per_class=1)
    model.train()
    model.on_epoch_start(1, None)
    assert not model.prototype_classifier.weight.requires_grad
    source = Batch.from_data_list([graph(0, 2)])
    with torch.no_grad():
        node_states = model._encode(source)[0]
    model.on_epoch_start(2, [source])
    assert model.training
    assert model.prototype_classifier.weight.requires_grad
    assert model.projection_summary["epoch"] == 2
    assert model.projection_summary["projected_prototypes"] == 1
    assert model.projection_summary["candidate_graphs_by_class"] == {"0": 1, "1": 0}
    assert any(torch.equal(model.prototype_vectors[0], row) for row in node_states)
    assert len(model.projection_summary["projected_node_type_ids"]) == 1
    with pytest.raises(ValueError, match="training loader"):
        model._project_from_training_loader(None, 3)


def test_protonode_runner_namespaced_options_and_early_stopping(tmp_path):
    base = ["--artifact", str(tmp_path / "a"), "--targets", str(tmp_path / "t"),
            "--output", str(tmp_path / "o")]
    parser = train.parser()
    try:
        parsed = parser.parse_args(base + ["--method", "protonode", "--protonode-readout", "node_max",
                                          "--protonode-warm-epochs", "2"])
    except SystemExit:
        pytest.fail("protonode runner CLI wiring missing")
    configured = train.normalize_method_args(parsed, parser)
    assert configured.protonode_readout == "node_max"
    assert configured.protonode_warm_epochs == 2
    assert configured._method_overrides == ["protonode_readout", "protonode_warm_epochs"]
    assert train.early_stopping_start_epoch("protonode", make_protonode()) == 1
    with pytest.raises(SystemExit):
        train.normalize_method_args(parser.parse_args(base + ["--method", "protgnn",
            "--protonode-readout", "node_max"]), parser)
    with pytest.raises(SystemExit):
        train.normalize_method_args(parser.parse_args(base + ["--method", "protonode",
            "--protgnn-warm-epochs", "2"]), parser)
    assert not (tmp_path / "o").exists()
