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
