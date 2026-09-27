"""Synthetic-only tests for the GMT clinical plugin."""
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch

from comparison.standardized.clinical_graph_v2 import train
from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY, build_method
from comparison.standardized.clinical_graph_v2.methods.base import relation_count
from comparison.standardized.clinical_graph_v2.tensorize import ALL_RELATIONS, PAYLOAD_WIDTH, ClinicalGraphData

NODE_DIM = 6
EDGE_DIM = len(ALL_RELATIONS) + PAYLOAD_WIDTH
NUM_TOKENS = 12
NUM_TRIPLES = 5
NUM_CLASSES = 2


def graph(label=0, node_count=3):
    x = torch.arange(node_count * NODE_DIM, dtype=torch.float32).view(node_count, NODE_DIM) / 10
    edge_index = torch.tensor([[i for i in range(node_count)], [(i + 1) % node_count for i in range(node_count)]])
    edge_count = edge_index.size(1)
    return ClinicalGraphData(
        x=x, edge_index=edge_index, edge_attr=torch.zeros(edge_count, EDGE_DIM),
        edge_relation=torch.zeros(edge_count, dtype=torch.long),
        edge_triple=torch.zeros(edge_count, dtype=torch.long),
        token=torch.arange(1, node_count + 1) % NUM_TOKENS,
        node_type=torch.arange(node_count) % 8, y=torch.tensor([label]))


def batch():
    return Batch.from_data_list([graph(0, 2), graph(1, 4)])


def make_gmt(**options):
    if "gmt" not in METHOD_REGISTRY:
        pytest.fail("gmt plugin not registered")
    return build_method(
        "gmt", num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        num_classes=NUM_CLASSES, hidden=16, layers=1, dropout=0.0,
        token_dim=4, num_triples=NUM_TRIPLES,
        args=Namespace(method_options=options))


def test_gmt_registers_constructs_runs_backward_and_fits_real_dimension_budget():
    model = make_gmt()
    output = model(batch(), epoch=0)
    assert output.logits.shape == (2, NUM_CLASSES)
    assert torch.isfinite(output.logits).all()
    assert output.auxiliary_loss.ndim == 0 and torch.isfinite(output.auxiliary_loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in output.diagnostics.values())
    (output.logits.sum() + output.auxiliary_loss).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert set(model.runner_defaults) == {"hidden", "layers", "dropout", "lr", "weight_decay",
                                          "batch_size", "epochs", "patience", "min_delta"}
    assert model.grad_clip_value == 2.0
    assert model.early_stopping_start() == 20
    real = build_method("gmt", num_tokens=391, node_dim=63, edge_dim=22,
        num_classes=10, hidden=model.runner_defaults["hidden"], layers=3,
        dropout=model.runner_defaults["dropout"], token_dim=32, num_triples=16, args=None)
    assert sum(p.numel() for p in real.parameters()) <= 405_000
    assert train.method_defaults("gmt") == model.runner_defaults


def test_gmt_attention_normalizes_real_nodes_and_zeros_padding():
    model = make_gmt()
    state = torch.randn(2, 4, model.hidden)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    _, weights = model.gmpool_g(state, mask)
    assert weights.shape == (2, model.heads, model.seeds, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
    assert torch.count_nonzero(weights[0, :, :, 2:]) == 0


def test_gmt_options_change_architecture_behavior_and_reject_unknown():
    defaults = make_gmt()
    changed = make_gmt(seeds="3", heads="2", sab="false", mean_skip="false")
    assert (defaults.seeds, defaults.heads, defaults.sab_enabled, defaults.mean_skip) == (8, 4, True, True)
    assert (changed.seeds, changed.heads, changed.sab_enabled, changed.mean_skip) == (3, 2, False, False)
    assert changed.gmpool_g.seeds.shape[0] == 3
    with pytest.raises(ValueError, match="unknown method option"):
        make_gmt(typo="1")


def test_gmt_explain_is_node_local_and_cross_graph_logits_are_isolated():
    torch.manual_seed(13)
    model = make_gmt().eval()
    source = batch()
    changed = source.clone()
    changed.x[changed.batch == 1] += 100
    with torch.no_grad():
        original_logits = model(source, epoch=0).logits
        changed_logits = model(changed, epoch=0).logits
        scores = model.explain(source)
        changed_scores = model.explain(changed)
    assert torch.allclose(original_logits[0], changed_logits[0], atol=1e-6)
    assert scores.shape == (source.x.size(0),)
    assert torch.isfinite(scores).all() and (scores >= 0).all()
    graph_a = source.batch == 0
    assert torch.allclose(scores[graph_a], changed_scores[graph_a], atol=1e-6)
