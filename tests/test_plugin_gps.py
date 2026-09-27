"""Synthetic-only contract tests for the GraphGPS clinical method plugin."""
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch, Data

from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY, build_method
from comparison.standardized.clinical_graph_v2.methods.base import relation_count


NODE_DIM = 63
EDGE_DIM = 22
NUM_TOKENS = 391
NUM_TRIPLES = 16
NUM_CLASSES = 10
TOKEN_DIM = 32


def make_model(**option_overrides):
    if "gps" not in METHOD_REGISTRY:
        pytest.fail("gps plugin not registered")
    options = option_overrides.pop("method_options", {})
    return build_method(
        "gps", num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_classes=NUM_CLASSES, hidden=80, layers=3, dropout=0.3,
        token_dim=TOKEN_DIM, num_triples=NUM_TRIPLES,
        args=Namespace(edge_direction="forward", method_options=options),
    )


def graph(node_count, feature_seed, label):
    generator = torch.Generator().manual_seed(feature_seed)
    x = torch.randn(node_count, NODE_DIM, generator=generator)
    edges = [(i, i + 1) for i in range(node_count - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.randn(len(edges), EDGE_DIM, generator=generator)
    return Data(
        x=x,
        token=torch.arange(1, node_count + 1) % NUM_TOKENS,
        node_type=torch.arange(node_count) % 8,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_relation=torch.zeros(len(edges), dtype=torch.long),
        edge_triple=torch.zeros(len(edges), dtype=torch.long),
        y=torch.tensor([label]),
    )


def two_graph_batch():
    return Batch.from_data_list([graph(3, 31, 0), graph(2, 47, 1)])


def test_gps_registration_forward_backward_defaults_and_real_dimension_budget():
    torch.manual_seed(101)
    model = make_model()
    batch = two_graph_batch()
    output = model(batch, epoch=0)
    assert output.logits.shape == (2, NUM_CLASSES)
    assert torch.isfinite(output.logits).all()
    assert output.auxiliary_loss.ndim == 0
    assert torch.isfinite(output.auxiliary_loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in output.diagnostics.values())
    (output.logits.sum() + output.auxiliary_loss).backward()
    assert model.node_encoder[0].weight.grad is not None
    assert torch.isfinite(model.node_encoder[0].weight.grad).all()

    assert set(model.runner_defaults) == {
        "hidden", "layers", "dropout", "lr", "weight_decay", "batch_size",
        "epochs", "patience", "min_delta",
    }
    assert model.runner_defaults["hidden"] == 80
    assert model.runner_defaults["layers"] == 3
    assert model.grad_clip_value == 2.0
    assert model.early_stopping_start() == 20
    assert sum(p.numel() for p in model.parameters()) <= 405_000
    assert model.run_config()["architecture"]["parameter_count"] == sum(
        p.numel() for p in model.parameters())
    assert relation_count(None) == model.num_relations


def test_gps_rwse_path_step_two_and_attention_masks_graph_padding():
    model = make_model().eval()
    batch = two_graph_batch()
    data = model._validated_batch(batch)
    encoding = model._random_walk_encoding(data)
    assert torch.allclose(encoding[:3, 1], torch.tensor([0.5, 1.0, 0.5]))

    state = model._input_state(data)
    _, weights, valid = model._global_attention(
        state, data.batch_index, data.graph_count, 0, return_weights=True)
    assert valid.tolist() == [[True, True, True], [True, True, False]]
    assert torch.all(weights[1, :, :, 2] == 0)
    assert torch.allclose(weights[0].sum(dim=-1), torch.ones_like(weights[0].sum(dim=-1)))
    assert torch.allclose(weights[1, :, :2, :2].sum(dim=-1), torch.ones((4, 2)))


def test_gps_batch_attention_and_importance_are_graph_local():
    torch.manual_seed(103)
    model = make_model().eval()
    batch = two_graph_batch()
    with torch.no_grad():
        original = model(batch, epoch=0).logits
        importance = model.explain(batch)
        changed = batch.clone()
        changed.x[3:] += 100.0
        changed_logits = model(changed, epoch=0).logits
        changed_importance = model.explain(changed)

    assert torch.equal(original[0], changed_logits[0])
    assert torch.allclose(importance[:3], changed_importance[:3])
    assert importance.shape == (batch.num_nodes,)
    assert torch.isfinite(importance).all()
    assert torch.all(importance >= 0)
    assert importance[:3].sum().item() == pytest.approx(1.0)
    assert importance[3:].sum().item() == pytest.approx(1.0)


def test_gps_options_defaults_effects_and_unknown_option_validation():
    default = make_model()
    assert default.rwse is True
    assert default.rwse_steps == 8
    assert default.rwse_projection is not None
    assert default.run_config()["native_defaults"]["rwse"] is True
    assert default.run_config()["native_defaults"]["rwse_steps"] == 8

    disabled = make_model(method_options={"rwse": "false"})
    assert disabled.rwse is False
    assert disabled.rwse_projection is None
    assert disabled.run_config()["effective_settings"]["rwse"] is False

    short_walk = make_model(method_options={"rwse_steps": "2"})
    assert short_walk.rwse_steps == 2
    assert short_walk.rwse_projection.in_features == 2
    assert short_walk.run_config()["effective_settings"]["rwse_steps"] == 2
    batch = two_graph_batch()
    assert default._random_walk_encoding(default._validated_batch(batch)).shape == (5, 8)
    assert short_walk._random_walk_encoding(short_walk._validated_batch(batch)).shape == (5, 2)

    more_heads = make_model(method_options={"heads": "8"})
    assert more_heads.heads == 8
    assert more_heads.run_config()["effective_settings"]["heads"] == 8

    with pytest.raises(ValueError, match="unknown method option"):
        make_model(method_options={"rwse_step": "4"})
    with pytest.raises(ValueError, match="rwse_steps"):
        make_model(method_options={"rwse_steps": "0"})


def test_gps_attention_dropout_is_applied_only_after_multihead_attention():
    model = make_model()
    assert model.dropout.p == pytest.approx(0.3)
    assert all(layer.dropout == 0.0 for layer in model.attention_layers)
