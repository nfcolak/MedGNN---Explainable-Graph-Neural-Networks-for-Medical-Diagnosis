"""Synthetic-only tests for the clinical virtual-node plugin."""
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch, Data

from comparison.standardized.clinical_graph_v2 import methods
from comparison.standardized.clinical_graph_v2.methods.base import relation_count


def make_batch(seed=4):
    generator = torch.Generator().manual_seed(seed)
    graphs = []
    for count in (3, 2):
        graphs.append(Data(
            x=torch.randn(count, 63, generator=generator),
            token=torch.randint(1, 391, (count,), generator=generator),
            node_type=torch.randint(0, 8, (count,), generator=generator),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 22)),
            edge_relation=torch.empty((0,), dtype=torch.long),
            edge_triple=torch.empty((0,), dtype=torch.long),
            y=torch.tensor([0]),
        ))
    return Batch.from_data_list(graphs)


def make(**overrides):
    adapter = methods.METHOD_REGISTRY.get("vnode")
    if adapter is None:
        pytest.fail("vnode plugin not registered")
    return adapter(
        num_tokens=391, node_dim=63, edge_dim=22, num_classes=10,
        hidden=104, layers=3, dropout=0.3, token_dim=32,
        num_triples=16, args=overrides.pop("args", Namespace()), **overrides)


def test_registration_forward_backward_defaults_and_real_dimension_budget():
    model = make()
    batch = make_batch()
    output = model(batch, epoch=0)
    assert output.logits.shape == (2, 10)
    assert output.auxiliary_loss.ndim == 0
    assert output.auxiliary_loss.requires_grad
    assert torch.isfinite(output.logits).all()
    (output.logits.sum() + output.auxiliary_loss).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.runner_defaults == {
        "hidden": 104, "layers": 3, "dropout": 0.3, "lr": 1.79e-3,
        "weight_decay": 4.3e-5, "batch_size": 128, "epochs": 40,
        "patience": 10, "min_delta": 0.005,
    }
    assert model.grad_clip_value == 2.0
    assert model.early_stopping_start() == 20
    assert sum(p.numel() for p in model.parameters()) <= 405_000
    groups = model.optimizer_groups(Namespace())
    assert len(groups) == 1
    assert groups[0]["params"] == list(model.parameters())


def test_virtual_node_transmits_information_and_isolates_graphs():
    model = make().eval()
    batch = make_batch()
    baseline = model._encode(batch)[0].detach()
    changed = batch.clone()
    changed.x[1] += 5.0  # node 1 has no edges and shares graph 0 with node 0
    after_same_graph_change = model._encode(changed)[0].detach()
    assert not torch.allclose(baseline[0], after_same_graph_change[0]), (
        "isolated node should receive same-graph information through the virtual node")

    output = model(batch, epoch=0).logits.detach()
    changed_other_graph = batch.clone()
    changed_other_graph.x[3:] -= 7.0
    output_after_other_graph_change = model(changed_other_graph, epoch=0).logits.detach()
    assert torch.allclose(output[0], output_after_other_graph_change[0], atol=1e-7, rtol=1e-6), (
        "virtual-node state must not leak across graphs")


def test_vn_pool_option_changes_outputs():
    summed = make().eval()
    averaged = make(args=Namespace(method_options={"vn_pool": "mean"})).eval()
    averaged.load_state_dict(summed.state_dict())
    batch = make_batch()
    sum_logits = summed(batch, epoch=0).logits
    mean_logits = averaged(batch, epoch=0).logits
    assert not torch.allclose(sum_logits, mean_logits), "vn_pool=sum and mean must differ"
    assert summed.run_config()["effective_settings"]["vn_pool"] == "sum"
    assert averaged.run_config()["effective_settings"]["vn_pool"] == "mean"


def test_options_are_validated_and_explain_is_finite_nonnegative_and_graph_local():
    with pytest.raises(ValueError, match="unknown method option"):
        make(args=Namespace(method_options={"vn_poo": "sum"}))
    with pytest.raises(ValueError, match="vn_pool"):
        make(args=Namespace(method_options={"vn_pool": "max"}))

    model = make().eval()
    batch = make_batch()
    importance = model.explain(batch)
    assert importance.shape == (batch.num_nodes,)
    assert torch.isfinite(importance).all()
    assert (importance >= 0).all()
    changed = batch.clone()
    changed.x[3:] += 11.0
    changed_importance = model.explain(changed)
    assert torch.allclose(importance[:3], changed_importance[:3], atol=1e-7, rtol=1e-6)
    assert model.run_config()["native_defaults"]["vn_pool"] == "sum"
