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


def test_wrapper_predictions_are_immutable_to_caller_metadata_mutation():
    implementation = module()
    adapter, graph = SyntheticAdapter(), graph_fixture()
    wrapper = implementation.ClinicalGraphXAIWrapper(adapter, graph).eval()
    features = adapter.continuous_inputs(graph)
    expected = wrapper(features, graph.edge_index, batch=graph.batch).detach().clone()
    graph.edge_attr.add_(100)
    graph.edge_relation.fill_(99)
    graph.edge_triple.fill_(88)
    graph.edge_index[:] = graph.edge_index.flip(1)
    actual = wrapper(features, wrapper._edge_index, batch=wrapper._batch)
    assert torch.equal(actual, expected), "caller mutation altered bound graph prediction payload"


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


def test_reconstruction_replays_candidate_and_product_off_state_dicts(tmp_path):
    implementation = module()
    from types import SimpleNamespace
    from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY
    from torch_geometric.data import Data

    graph = Data(
        x=torch.tensor([[0.2, -0.3, 0.5], [0.8, 0.1, -0.4]]),
        token=torch.tensor([1, 2]), node_type=torch.tensor([0, 1]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[0.2, 0.1], [0.4, -0.3]]),
        edge_relation=torch.tensor([0, 1]), edge_triple=torch.tensor([0, 1]),
    )
    for enabled in (True, False):
        torch.manual_seed(88)
        args = SimpleNamespace(method_options={"interaction_rank": 16, "use_interactions": enabled},
                              edge_direction="forward")
        adapter = METHOD_REGISTRY["cei_gnn"](
            num_tokens=8, node_dim=3, edge_dim=2, num_classes=3, hidden=8,
            layers=1, dropout=0.0, token_dim=4, num_triples=3, args=args,
        ).eval()
        config = adapter.run_config()
        checkpoint = tmp_path / f"cei-{enabled}.pt"
        torch.save(adapter.state_dict(), checkpoint)
        from comparison.standardized.clinical_graph_v2.contracts import recursive_source_hashes
        import hashlib
        from pathlib import Path
        binding = {
            "method": "cei_gnn", "adaptation_version": config["adaptation_version"],
            "method_config": config,
            "source_code": recursive_source_hashes(MODULE_PATH.parents[0]),
            "runner_settings": {"edge_direction": "forward"},
            "vocabulary_size": 8, "node_dim": 3, "edge_dim": 2, "num_classes": 3,
            "hidden": 8, "layers": 1, "dropout": 0.0, "token_dim": 4,
            "num_meta_relations": 3, "num_relations": config["architecture"]["num_relations"],
        }
        reconstructed = None
        try:
            reconstructed = implementation.reconstruct_adapter(
                binding, checkpoint,
                expected_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            )
        except Exception:
            pass
        assert reconstructed is not None, f"failed to reconstruct use_interactions={enabled}"
        expected = adapter(graph, epoch=0).logits.softmax(-1)
        actual = reconstructed(graph, epoch=0).logits.softmax(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reconstruction_refuses_unbound_or_incompatible_adaptation():
    implementation = module()
    with pytest.raises(ValueError, match="incompatible method/adaptation"):
        implementation.reconstruct_adapter(
            {"method": "protgnn", "adaptation_version": "wrong"},
            "unused.pt",
            expected_checkpoint_sha256="0" * 64,
        )


def test_export_refuses_occupied_directory_and_incomplete_algorithm_records(tmp_path):
    implementation = module()
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        implementation.export_explanations(out, records=[], manifest={})
    fresh = tmp_path / "fresh"
    with pytest.raises(ValueError, match="incomplete"):
        implementation.export_explanations(
            fresh, records=[{"sample_id": "dev-1", "graphxai": {}}],
            manifest=_valid_export_manifest(["dev-1"]),
            binding=_runner_binding(["dev-1"]), frozen_dev_ids=["dev-1"],
        )
    assert fresh.is_dir(), "failed output reservation is retained for recovery"
    assert not (fresh / "manifest.json").exists(), "failed export must not publish completion"


def _valid_export_manifest(sample_ids, **overrides):
    import hashlib
    import json
    digest = hashlib.sha256(json.dumps(sample_ids, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "source_sha256": {"module.py": "a" * 64},
        "graph_sha256": "b" * 64,
        "membership_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "cohort_ids_sha256": digest,
        "fold": "dev",
        "seed": 1234,
        "class_order": ["class-a", "class-b"],
        "cohort_identity": "synthetic-dev",
        "interpretation_boundaries": {
            "node_gradients": "continuous numeric and embedding channels, conditional on fixed edge metadata",
            "feature_zeroing": "continuous-representation intervention; not clinical event deletion",
            "causal_claim": False,
        },
    }
    manifest.update(overrides)
    return manifest


def _runner_binding(dev_ids):
    import hashlib
    import json
    digest = hashlib.sha256(json.dumps(dev_ids, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    return {"split_sample_ids_sha256": {"dev": digest}, "label_order": ["class-a", "class-b"]}


def test_export_requires_binding_checked_frozen_dev_membership(tmp_path):
    implementation = module()
    import inspect
    assert {"binding", "frozen_dev_ids"} <= set(inspect.signature(implementation.export_explanations).parameters)
    ids = ["dev-sample-1"]
    binding = _runner_binding(ids)
    record = {"sample_id": "validation-sample-id", "graphxai": {}}
    with pytest.raises(ValueError, match="dev roster|membership|hash"):
        implementation.export_explanations(
            tmp_path / "not-dev", records=[record],
            manifest=_valid_export_manifest(["validation-sample-id"]),
            binding=binding, frozen_dev_ids=ids,
        )


def test_export_refuses_status_only_graphxai_records(tmp_path):
    implementation = module()
    record = {
        "sample_id": "dev-1",
        "graphxai": {name: {"status": "success"} for name in implementation.REQUIRED_ALGORITHMS},
    }
    with pytest.raises(ValueError, match="explanation|provenance|finite"):
        implementation.export_explanations(
            tmp_path / "status-only", records=[record],
            manifest=_valid_export_manifest(["dev-1"]),
            binding=_runner_binding(["dev-1"]), frozen_dev_ids=["dev-1"],
        )
    assert not (tmp_path / "status-only" / "manifest.json").exists()


def test_export_generator_failure_fsyncs_and_preserves_prior_records(tmp_path, monkeypatch):
    implementation = module()
    out = tmp_path / "interrupted"
    fsync_calls = []
    original_fsync = implementation.os.fsync

    def observed_fsync(fd):
        fsync_calls.append(fd)
        return original_fsync(fd)

    monkeypatch.setattr(implementation.os, "fsync", observed_fsync)

    def partial_records():
        yield {"sample_id": "dev-1", "graphxai": {}}
        raise RuntimeError("synthetic interrupted generator")

    with pytest.raises(RuntimeError, match="synthetic interrupted"):
        implementation.export_explanations(out, records=partial_records(), manifest={})
    journals = list(tmp_path.glob(".*.failed.journal"))
    assert journals, "partial journal must remain discoverable after iterator failure"
    assert fsync_calls, "each yielded record must be fsynced before requesting the next"
    assert journals[0].read_text().strip() == '{"graphxai": {}, "sample_id": "dev-1"}'


def test_export_race_preserves_directory_created_by_another_actor(tmp_path, monkeypatch):
    implementation = module()
    out = tmp_path / "raced"
    original_mkdir = Path.mkdir

    def create_foreign_directory(path, *args, **kwargs):
        if path == out:
            original_mkdir(path, *args, **kwargs)
            (out / "foreign_marker.txt").write_text("not ours")
            raise FileExistsError(path)
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", create_foreign_directory)
    with pytest.raises(FileExistsError):
        implementation.export_explanations(out, records=[], manifest={})
    assert (out / "foreign_marker.txt").read_text() == "not ours", "exporter deleted unowned output"
