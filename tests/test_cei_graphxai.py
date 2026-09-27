import importlib.util
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


MODULE_PATH = Path(__file__).resolve().parents[1] / "comparison/standardized/clinical_graph_v2/cei_graphxai.py"


def graph_fixture():
    return Data(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[2.0], [3.0]]),
        edge_relation=torch.tensor([4, 5]),
        edge_triple=torch.tensor([6, 7]),
        batch=torch.tensor([0, 0]),
        num_graphs=1,
    )


class SyntheticAdapter(torch.nn.Module):
    class PassEdges(MessagePassing):
        def __init__(self):
            super().__init__(aggr="add")

        def forward(self, x, edge_index):
            return self.propagate(edge_index, x=x)

        def message(self, x_j):
            return x_j

    def __init__(self):
        super().__init__()
        self.predictor = torch.nn.Linear(2, 2, bias=False)
        self.edge_pass = self.PassEdges()
        self.seen_features = None
        with torch.no_grad():
            self.predictor.weight.copy_(torch.tensor([[1.0, 2.0], [-1.0, 1.0]]))

    def continuous_inputs(self, graph):
        rows = int(graph.num_nodes)
        base = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        return base[:rows]

    def forward_continuous(self, features, edge_index, metadata, *, return_parts=False):
        self.seen_features = features
        # Both continuous channels and fixed edge payloads affect prediction; edge
        # presence travels through a real PyG MessagePassing explanation hook.
        node = self.predictor(features) + self.edge_pass(features, edge_index)
        signal = node.sum().view(1, 1) + metadata.edge_attr.sum().view(1, 1)
        return torch.cat([signal, -signal], dim=1)

    def forward(self, graph, *, epoch):
        return type("Output", (), {"logits": self.forward_continuous(
            self.continuous_inputs(graph), graph.edge_index, graph
        )})()


def module():
    if not MODULE_PATH.is_file():
        pytest.fail("clinical dev-only GraphXAI wrapper absent")
    spec = importlib.util.spec_from_file_location("cei_graphxai_under_test", MODULE_PATH)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_wrapper_preserves_numeric_and_embedding_channels_as_predictor_inputs():
    implementation = module()
    adapter, graph = SyntheticAdapter(), graph_fixture()
    wrapper = implementation.ClinicalGraphXAIWrapper(adapter, graph).eval()
    features = adapter.continuous_inputs(graph).detach().clone().requires_grad_(True)
    logits = wrapper(features, graph.edge_index, batch=graph.batch)
    assert torch.equal(logits, adapter(graph, epoch=0).logits)
    logits[0, 0].backward()
    assert torch.equal(adapter.seen_features, features)
    assert torch.all(features.grad.abs().sum(dim=0) > 0), "numeric and embedding channels must reach predictor"


def test_wrapper_rejects_reordered_edges_and_changed_batch_identity():
    implementation = module()
    adapter, graph = SyntheticAdapter(), graph_fixture()
    wrapper = implementation.ClinicalGraphXAIWrapper(adapter, graph)
    features = adapter.continuous_inputs(graph)
    with pytest.raises(ValueError, match="edge"):
        wrapper(features, graph.edge_index.flip(1), batch=graph.batch)
    with pytest.raises(ValueError, match="batch"):
        wrapper(features, graph.edge_index, batch=torch.tensor([0, 1]))


def test_explain_graph_runs_real_algorithms_and_cleans_mask_state():
    implementation = module()
    from shared.lib.graphxai_standardized import ALGORITHMS

    adapter, graph = SyntheticAdapter(), graph_fixture()
    objectives = []
    original_backward = torch.Tensor.backward
    def checked_backward(loss, *args, **kwargs):
        objectives.append(loss.detach())
        assert torch.isfinite(loss).all(), "mask optimization objective must remain finite"
        return original_backward(loss, *args, **kwargs)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(torch.Tensor, "backward", checked_backward)
    try:
        result = implementation.explain_graph(adapter, graph, steps=4, epochs=3)
    finally:
        monkeypatch.undo()
    assert objectives, "GNNExplainer optimization objective was not exercised"
    assert set(result) == set(ALGORITHMS)
    assert all(item["status"] == "success" for item in result.values())
    assert result["GNNExplainer"]["provenance"]["edge_gradient_verified"] is True
    assert all(torch.isfinite(torch.as_tensor(item["node_explanation"]["node_importance"])).all() for item in result.values())
    assert not adapter.edge_pass.explain


def test_dev_cohort_requires_exact_sample_ids_and_fail_closed_provenance():
    implementation = module()
    cohort = implementation.validate_dev_cohort(
        requested_ids=["visit-1", "visit-2"],
        frozen_dev_ids=["visit-1", "visit-2", "visit-3"],
        validation_ids=["visit-v"],
        test_ids=["visit-t"],
    )
    assert cohort == ("visit-1", "visit-2")
    with pytest.raises(ValueError):
        implementation.validate_dev_cohort(
            requested_ids=["subject-1"], frozen_dev_ids=["visit-1"],
            validation_ids=[], test_ids=[],
        )


def test_real_algorithms_support_one_node_edgeless_graph():
    implementation = module()
    adapter = SyntheticAdapter()
    graph = Data(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1)),
        edge_relation=torch.empty((0,), dtype=torch.long),
        edge_triple=torch.empty((0,), dtype=torch.long),
        batch=torch.zeros(1, dtype=torch.long),
        num_nodes=1,
        num_graphs=1,
    )
    result = implementation.explain_graph(adapter, graph, steps=4, epochs=3)
    assert set(result) == {"GradExplainer", "IntegratedGradExplainer", "GNNExplainer"}
    assert result["GNNExplainer"]["provenance"]["edge_gradient_verified"] is False
    assert result["GNNExplainer"]["provenance"]["edge_gradient_status"] == "not_applicable_edgeless"
    assert all(torch.isfinite(torch.as_tensor(item["node_explanation"]["node_importance"])).all() for item in result.values())


def test_reconstruction_refuses_unbound_or_incompatible_adaptation():
    implementation = module()
    with pytest.raises(ValueError, match="incompatible method/adaptation"):
        implementation.reconstruct_adapter(
            {"method": "protgnn", "adaptation_version": "wrong"},
            "unused.pt",
            expected_source_sha256={},
        )


def test_export_refuses_occupied_directory_and_incomplete_algorithm_records(tmp_path):
    implementation = module()
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        implementation.export_explanations(out, records=[], manifest={})
    fresh = tmp_path / "fresh"
    with pytest.raises(ValueError, match="incomplete"):
        implementation.export_explanations(fresh, records=[{"graphxai": {}}], manifest={})
    assert not fresh.exists(), "failed export must not leave a completed output directory"


def test_export_race_preserves_directory_created_by_another_actor(tmp_path, monkeypatch):
    implementation = module()
    out = tmp_path / "raced"
    original_mkstemp = implementation.tempfile.mkstemp

    def create_foreign_directory(*args, **kwargs):
        out.mkdir()
        (out / "foreign_marker.txt").write_text("not ours")
        return original_mkstemp(*args, **kwargs)

    monkeypatch.setattr(implementation.tempfile, "mkstemp", create_foreign_directory)
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        implementation.export_explanations(out, records=[{"graphxai": {}}], manifest={})
    assert (out / "foreign_marker.txt").read_text() == "not ours", "exporter deleted unowned output"
