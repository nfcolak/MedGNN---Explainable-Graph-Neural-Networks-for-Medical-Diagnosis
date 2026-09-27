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


def test_class_attention_normalizes_and_matches_hand_computed_caml_logits():
    model = make(mean_path="false")
    node_state = torch.zeros((5, 128))
    node_state[:, :2] = torch.tensor([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0],
                                      [-1.0, 1.0], [0.5, 3.0]])
    graph_ids = torch.tensor([0, 0, 0, 1, 1])
    model.attn_temperature = 1.0
    with torch.no_grad():
        model.attention_query[:2, :2].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        model.classifier_weight[:2, :2].copy_(torch.tensor([[2.0, 1.0], [-1.0, 3.0]]))
        model.classifier_bias[:2].copy_(torch.tensor([0.25, -0.5]))

    logits, alpha = model._attention_readout(node_state, graph_ids, 2)
    expected_alpha = torch.zeros_like(alpha)
    expected_context = torch.zeros((2, NUM_CLASSES, 128))
    for graph_id in range(2):
        selected = node_state[graph_ids == graph_id]
        scores = selected @ model.attention_query.t()
        weights = torch.softmax(scores, dim=0)
        expected_alpha[graph_ids == graph_id] = weights
        expected_context[graph_id] = weights.t() @ selected
    expected_logits = torch.einsum(
        "bch,ch->bc", expected_context, model.classifier_weight)
    expected_logits = expected_logits + model.classifier_bias
    assert torch.allclose(alpha, expected_alpha)
    assert torch.allclose(logits, expected_logits)
    for graph_id in range(2):
        assert torch.allclose(alpha[graph_ids == graph_id].sum(0), torch.ones(NUM_CLASSES))


def test_temperature_cross_graph_isolation_and_explain_are_graph_local():
    torch.manual_seed(31)
    model = make().eval()
    with torch.no_grad():
        model.attention_query.zero_()
        model.attention_query[:, 0] = 1.0
    encoded = torch.zeros((3, 128))
    encoded[:, 0] = torch.tensor([0.0, 4.0, 1.0])
    ids = torch.tensor([0, 0, 0])
    model.attn_temperature = 0.5
    _, sharp = model._attention_readout(encoded, ids, 1)
    sharp_entropy = -(sharp * sharp.clamp_min(1e-12).log()).sum(0).mean()
    model.attn_temperature = 2.0
    _, diffuse = model._attention_readout(encoded, ids, 1)
    diffuse_entropy = -(diffuse * diffuse.clamp_min(1e-12).log()).sum(0).mean()
    assert diffuse_entropy > sharp_entropy

    source = batch()
    original = model(source, epoch=0).logits.detach().clone()
    changed = source.clone()
    changed.x[changed.batch == 1] += 100.0
    altered = model(changed, epoch=0).logits.detach()
    assert torch.allclose(original[0], altered[0], atol=1e-6)
    first_explanation = model.explain(source)
    second_explanation = model.explain(changed)
    assert first_explanation.shape == (source.num_nodes,)
    assert torch.isfinite(first_explanation).all() and torch.all(first_explanation >= 0)
    assert torch.allclose(first_explanation[source.batch == 0],
                          second_explanation[changed.batch == 0], atol=1e-6)


def test_native_options_defaults_effects_and_validation():
    defaults = make()
    configured = make(mean_path="false", attn_temperature="0.5", attn_dropout="0.4")
    assert defaults.run_config()["effective_settings"] == {
        "mean_path": True, "attn_temperature": 1.0, "attn_dropout": 0.0,
    }
    assert configured.run_config()["effective_settings"] == {
        "mean_path": False, "attn_temperature": 0.5, "attn_dropout": 0.4,
    }
    assert configured.mean_classifier is None
    assert defaults.mean_classifier is not None
    assert defaults.run_config()["native_defaults"]["mean_path"] is True
    assert defaults.run_config()["mechanism_settings"]["base_paper"].endswith("NAACL 2018")
    with pytest.raises(ValueError, match="unknown method option"):
        make(attn_temprature="2")
    with pytest.raises(ValueError, match="attn_temperature"):
        make(attn_temperature="0")
    with pytest.raises(ValueError, match="attn_dropout"):
        make(attn_dropout="1")


def test_labelattn_parser_smoke_prints_not_executed_defaults(capsys):
    args = train.parser().parse_args([
        "--artifact", "x", "--targets", "y", "--output", "z", "--method", "labelattn",
    ])
    normalized = train.normalize_method_args(args, train.parser())
    assert normalized.hidden == 128 and normalized.epochs == 40
