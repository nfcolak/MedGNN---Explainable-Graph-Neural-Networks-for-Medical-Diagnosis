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
