"""Synthetic contract tests for the token-count Wide & Deep plugin."""
from argparse import Namespace

import pytest
import torch
from torch_geometric.data import Batch, Data

from comparison.standardized.clinical_graph_v2 import methods
from comparison.standardized.clinical_graph_v2.methods.base import relation_count


def make(args=None, *, hidden=16, layers=2, dropout=0.3, num_tokens=8,
         node_dim=4, edge_dim=22, num_classes=3, token_dim=4, num_triples=3):
    adapter_type = methods.METHOD_REGISTRY.get("tokenfusion")
    if adapter_type is None:
        pytest.fail("tokenfusion plugin not registered")
    return adapter_type(num_tokens=num_tokens, node_dim=node_dim, edge_dim=edge_dim,
                        num_classes=num_classes, hidden=hidden, layers=layers,
                        dropout=dropout, token_dim=token_dim,
                        num_triples=num_triples, args=args or Namespace())


def graph(tokens, *, node_dim=4, offset=0):
    n = len(tokens)
    x = torch.arange(n * node_dim, dtype=torch.float32).reshape(n, node_dim) / 10 + offset
    edges = torch.tensor([[i for i in range(n - 1)], [i + 1 for i in range(n - 1)]], dtype=torch.long)
    if n <= 1:
        edges = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, token=torch.tensor(tokens), node_type=torch.zeros(n, dtype=torch.long),
                edge_index=edges, edge_attr=torch.zeros((edges.size(1), 22)),
                edge_relation=torch.zeros(edges.size(1), dtype=torch.long),
                edge_triple=torch.zeros(edges.size(1), dtype=torch.long))


def test_registration_forward_backward_runner_profile_and_parameter_budget():
    adapter = make()
    batch = Batch.from_data_list([graph([1, 2, 2]), graph([3, 4])])
    output = adapter(batch, epoch=0)
    assert output.logits.shape == (2, 3)
    assert output.auxiliary_loss.ndim == 0 and output.auxiliary_loss.requires_grad
    assert all(torch.isfinite(value).all() for value in (output.logits, output.auxiliary_loss))
    (output.logits.sum() + output.auxiliary_loss).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in adapter.parameters())
    assert set(adapter.runner_defaults) == {
        "hidden", "layers", "dropout", "lr", "weight_decay", "batch_size",
        "epochs", "patience", "min_delta"}
    assert adapter.grad_clip_value == 2.0
    assert adapter.early_stopping_start() == 20

    real = make(hidden=adapter.runner_defaults["hidden"], layers=3, dropout=adapter.runner_defaults["dropout"],
                num_tokens=391, node_dim=63, edge_dim=22, num_classes=10,
                token_dim=32, num_triples=16)
    assert sum(p.numel() for p in real.parameters()) <= 405_000


def test_count_features_are_sums_and_wide_logits_have_exact_token_effect():
    args = Namespace(method_options={"count_transform": "raw", "tab_mlp": "false"})
    adapter = make(args=args, num_tokens=8, num_classes=3)
    first = Batch.from_data_list([graph([1, 2]), graph([3])])
    duplicated = Batch.from_data_list([graph([1, 1, 2]), graph([3])])
    with torch.no_grad():
        adapter.wide.weight[:, 1].copy_(torch.tensor([0.25, -0.5, 1.25]))
        adapter.wide.weight[:, 2].copy_(torch.tensor([0.2, 0.3, -0.4]))
        adapter.wide.bias.zero_()
        first_clinical, _, _ = adapter._encode(first)
        duplicate_clinical, _, _ = adapter._encode(duplicated)
        first_counts = adapter._count_features(first_clinical)
        duplicate_counts = adapter._count_features(duplicate_clinical)
        assert first_counts[0, 1].item() == 1
        assert duplicate_counts[0, 1].item() == 2
        effect = adapter.wide(duplicate_counts)[0] - adapter.wide(first_counts)[0]
    torch.testing.assert_close(effect, adapter.wide.weight[:, 1])


def test_wide_and_tab_logits_are_zero_initialized_and_tab_branch_is_optional():
    batch = Batch.from_data_list([graph([1, 2]), graph([3])])
    adapter = make()
    adapter.eval()
    _, _, _, _, wide, tab = adapter._forward_parts(batch)
    assert torch.equal(wide, torch.zeros_like(wide))
    assert torch.equal(tab, torch.zeros_like(tab))
    no_tab = make(args=Namespace(method_options={"tab_mlp": "false"}))
    assert no_tab.tab_head is None
    output = no_tab(batch, epoch=0)
    assert output.logits.shape == (2, 3)
    config = no_tab.run_config()
    assert config["effective_settings"]["tab_mlp"] is False


def test_options_change_count_features_and_unknown_options_fail_closed():
    batch = Batch.from_data_list([graph([1, 1, 2])])
    models = [make(args=Namespace(method_options={"count_transform": mode}))
              for mode in ("log1p", "binary", "raw")]
    features = []
    for adapter in models:
        clinical, _, _ = adapter._encode(batch)
        features.append(adapter._count_features(clinical))
    assert features[0][0, 1].item() == pytest.approx(torch.log1p(torch.tensor(2.)).item())
    assert features[1][0, 1].item() == 1
    assert features[2][0, 1].item() == 2
    assert len({tuple(model.run_config()["effective_settings"].items()) for model in models}) == 3

    default_tab = make()
    narrow_tab = make(args=Namespace(method_options={"tab_hidden": "7"}))
    assert default_tab.tab_head[0].out_features == 64
    assert narrow_tab.tab_head[0].out_features == 7
    weighted = make(args=Namespace(method_options={"wide_l1": "0.2"}))
    with torch.no_grad():
        weighted.wide.weight.fill_(1.0)
    out = weighted(batch, epoch=0)
    assert out.auxiliary_loss.item() == pytest.approx(0.2 * weighted.num_tokens * weighted.num_classes)

    with pytest.raises(ValueError, match="unknown method option"):
        make(args=Namespace(method_options={"count_tranform": "raw"}))
    with pytest.raises(ValueError, match="wide_l1"):
        make(args=Namespace(method_options={"wide_l1": "-1"}))
    with pytest.raises(ValueError, match="tab_hidden"):
        make(args=Namespace(method_options={"tab_hidden": "0"}))


def test_explain_is_finite_nonnegative_and_graph_local():
    adapter = make(args=Namespace(method_options={"count_transform": "raw"}))
    with torch.no_grad():
        adapter.wide.weight.copy_(torch.arange(adapter.wide.weight.numel()).reshape_as(adapter.wide.weight) / 10)
    batch_a = Batch.from_data_list([graph([1, 1, 2]), graph([3, 4])])
    changed_b = Batch.from_data_list([graph([1, 1, 2]), graph([5, 5, 6, 6])])
    adapter.eval()
    with torch.no_grad():
        scores_a = adapter.explain(batch_a)
        scores_changed = adapter.explain(changed_b)
    assert scores_a.shape == (batch_a.num_nodes,)
    assert torch.isfinite(scores_a).all() and (scores_a >= 0).all()
    first_graph_nodes = batch_a.batch == 0
    changed_first_graph_nodes = changed_b.batch == 0
    torch.testing.assert_close(scores_a[first_graph_nodes],
                               scores_changed[changed_first_graph_nodes])


def test_graphs_do_not_leak_into_each_others_logits():
    adapter = make()
    adapter.eval()
    original = Batch.from_data_list([graph([1, 2], offset=0), graph([3, 4], offset=0)])
    changed = Batch.from_data_list([graph([1, 2], offset=0), graph([3, 4], offset=100)])
    with torch.no_grad():
        logits_a = adapter(original, epoch=0).logits
        logits_b = adapter(changed, epoch=0).logits
    torch.testing.assert_close(logits_a[0], logits_b[0])
