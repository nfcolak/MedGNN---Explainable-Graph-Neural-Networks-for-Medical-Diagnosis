"""Synthetic-only tests for the CAML label-wise clinical graph plugin."""
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch

from comparison.standardized.clinical_graph_v2 import methods, train
from comparison.standardized.clinical_graph_v2.methods.base import relation_count
from comparison.standardized.clinical_graph_v2.tensorize import (
    ALL_RELATIONS, PAYLOAD_WIDTH, ClinicalGraphData,
)

NODE_DIM = 63
EDGE_DIM = 22
NUM_TOKENS = 391
NUM_TRIPLES = 16
NUM_CLASSES = 10
TOKEN_DIM = 32


def require_labelattn():
    if "labelattn" not in methods.METHOD_REGISTRY:
        pytest.fail("labelattn plugin not registered")
    return methods.METHOD_REGISTRY["labelattn"]


def graph(node_count, offset=0):
    x = torch.arange(node_count * NODE_DIM, dtype=torch.float32).reshape(node_count, NODE_DIM)
    x = x / 100.0 + offset
    edge_index = torch.tensor(
        [[i for i in range(node_count)], [(i + 1) % node_count for i in range(node_count)]],
        dtype=torch.long,
    )
    edge_count = edge_index.size(1)
    return ClinicalGraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=torch.zeros(edge_count, EDGE_DIM),
        edge_relation=torch.arange(edge_count, dtype=torch.long) % len(ALL_RELATIONS),
        edge_triple=torch.arange(edge_count, dtype=torch.long) % NUM_TRIPLES,
        token=torch.arange(1, node_count + 1, dtype=torch.long) % NUM_TOKENS,
        node_type=torch.arange(node_count, dtype=torch.long) % 8,
        y=torch.tensor([offset % NUM_CLASSES], dtype=torch.long),
    )


def batch():
    return Batch.from_data_list([graph(3, 0), graph(2, 1)])


def make(**options):
    adapter_type = require_labelattn()
    return methods.build_method(
        "labelattn", num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_classes=NUM_CLASSES, hidden=128, layers=3, dropout=0.3,
        token_dim=TOKEN_DIM, num_triples=NUM_TRIPLES,
        args=Namespace(method_options=options),
    )


def test_labelattn_registration_forward_backward_defaults_and_budget():
    model = make()
    assert isinstance(model, require_labelattn())
    output = model(batch(), epoch=0)
    assert output.logits.shape == (2, NUM_CLASSES)
    assert torch.isfinite(output.logits).all()
    assert output.auxiliary_loss.ndim == 0 and torch.isfinite(output.auxiliary_loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in output.diagnostics.values())
    (output.logits.sum() + output.auxiliary_loss).backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    defaults = model.runner_defaults
    assert set(defaults) == {
        "hidden", "layers", "dropout", "lr", "weight_decay", "batch_size",
        "epochs", "patience", "min_delta",
    }
    assert defaults["hidden"] == 128
    assert model.grad_clip_value == 2.0
    assert model.early_stopping_start() == 20
    assert sum(parameter.numel() for parameter in model.parameters()) <= 405_000
