"""Focused synthetic mechanism checks for clinical method adapters.

These tests exercise in-memory PyG batches only. They do not generate clinical
artifacts, train benchmark arms, or open the held-out test fold.
"""
import copy
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch

from core import aggregate, paths, train
from core.contracts import recursive_source_hashes
from core.registry import build_method
from gsat import _ClinicalGINLayer
from core.tensorize import (
    ALL_RELATIONS,
    PAYLOAD_WIDTH,
    ClinicalGraphData,
)


NODE_DIM = 6
EDGE_DIM = len(ALL_RELATIONS) + PAYLOAD_WIDTH
NUM_TOKENS = 12
NUM_TRIPLES = 5
NUM_CLASSES = 2


def synthetic_graph(*, sample_id, label, node_count, num_visits, membership, global_mask):
    generator = torch.Generator().manual_seed(100 + label + node_count)
    x = torch.randn((node_count, NODE_DIM), generator=generator)
    edge_pairs = [(index, (index + 1) % node_count) for index in range(node_count)]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.randn((len(edge_pairs), EDGE_DIM), generator=generator)
    edge_relation = torch.arange(len(edge_pairs), dtype=torch.long) % len(ALL_RELATIONS)
    edge_triple = torch.arange(len(edge_pairs), dtype=torch.long) % NUM_TRIPLES
    data = ClinicalGraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        visit_membership_index=torch.tensor(membership, dtype=torch.long),
        num_visits=torch.tensor([num_visits], dtype=torch.long),
        global_node_mask=torch.tensor(global_mask, dtype=torch.bool),
    )
    data.token = torch.arange(1, node_count + 1, dtype=torch.long) % NUM_TOKENS
    data.node_type = torch.arange(node_count, dtype=torch.long) % 8
    data.edge_relation = edge_relation
    data.edge_triple = edge_triple
    data.y = torch.tensor([label], dtype=torch.long)
    data.sample_id = sample_id
    return data


def synthetic_batch():
    first = synthetic_graph(
        sample_id="train-a",
        label=0,
        node_count=4,
        num_visits=3,
        membership=[[0, 0, 2, 2], [0, 2, 0, 1]],
        global_mask=[False, False, False, True],
    )
    second = synthetic_graph(
        sample_id="train-b",
        label=1,
        node_count=3,
        num_visits=2,
        membership=[[0, 1], [0, 1]],
        global_mask=[False, False, True],
    )
    return Batch.from_data_list([first, second])


def method_args(**overrides):
    values = {
        "epochs": 4,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 2,
        "patience": 2,
        "min_delta": 0.0,
        "warm_epochs": 1,
        "proj_epochs": 1,
        "proj_interval": 5,
        "nearest_graphs": 2,
        "prototypes_per_class": 1,
        "rollout": 1,
        "min_atoms": 2,
        "max_atoms": 3,
        "expand_atoms": 2,
        "c_puct": 1.0,
        "gsat_extractor_dropout": 0.0,
        "graphcare_message_dropout": 0.0,
    }
    values.update(overrides)
    return Namespace(**values)


def adapter(name, **overrides):
    return build_method(
        name,
        num_tokens=NUM_TOKENS,
        node_dim=NODE_DIM,
        edge_dim=EDGE_DIM,
        num_classes=NUM_CLASSES,
        hidden=8,
        layers=2,
        dropout=0.0,
        token_dim=4,
        num_triples=NUM_TRIPLES,
        args=method_args(**overrides),
    )


def assert_finite_gradient(parameter):
    assert parameter.grad is not None
    assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("name", ["protgnn", "gsat", "graphcare"])
def test_all_adapters_have_finite_forward_backward(name):
    torch.manual_seed(7)
    model = adapter(name)
    model.train()
    batch = synthetic_batch()
    output = model(batch, epoch=0)

    assert output.logits.shape == (2, NUM_CLASSES)
    assert torch.isfinite(output.logits).all()
    assert output.auxiliary_loss.ndim == 0
    assert torch.isfinite(output.auxiliary_loss)
    assert all(torch.isfinite(torch.tensor(value)) for value in output.diagnostics.values())

    (output.logits.sum() + output.auxiliary_loss).backward()
    if name == "protgnn":
        assert_finite_gradient(model.prototype_vectors)
        assert_finite_gradient(model.relation_embedding.weight)
        assert_finite_gradient(model.edge_feature_projection.weight)
    elif name == "gsat":
        assert_finite_gradient(model.extractor.mlp[-1].weight)
        assert_finite_gradient(model.predictor.relation_embedding.weight)
        assert_finite_gradient(model.predictor.edge_feature_projection.weight)
    else:
        assert_finite_gradient(model.layers[0].relation_gate.weight)
        assert_finite_gradient(model.alpha_projection[0].weight)
        assert_finite_gradient(model.beta_projection[0].weight)
        assert_finite_gradient(model.edge_feature_projection.weight)


def test_protgnn_objective_warmup_and_train_only_projection():
    torch.manual_seed(11)
    model = adapter("protgnn")
    batch = synthetic_batch()

    output = model(batch, epoch=0)
    assert output.diagnostics["cluster_loss"] >= 0.0
    assert output.diagnostics["separation_loss"] >= 0.0
    assert model.run_config()["mechanism_settings"]["projection"]["candidate_source"] == (
        "training_loader_only"
    )

    model.on_epoch_start(0, None)
    assert model.prototype_classifier.weight.requires_grad is False

    class RecordingTrainLoader:
        def __init__(self, source_batch):
            self.source_batch = source_batch
            self.consumed = []

        def __iter__(self):
            self.consumed.extend(self.source_batch.sample_id)
            yield self.source_batch

    loader = RecordingTrainLoader(batch)
    model.on_epoch_start(1, loader)
    assert model.prototype_classifier.weight.requires_grad is True
    assert loader.consumed == ["train-a", "train-b"]
    assert model.projection_summary["epoch"] == 1
    assert model.projection_summary["projected_prototypes"] == NUM_CLASSES
    assert model.projection_summary["candidate_graphs_by_class"] == {"0": 1, "1": 1}

    class ForbiddenLoader:
        def __iter__(self):
            raise AssertionError("non-projection epoch consumed a loader")

    model.on_epoch_start(2, ForbiddenLoader())


def test_protgnn_projection_reencodes_connected_induced_subgraphs():
    torch.manual_seed(12)
    model = adapter("protgnn")
    model.eval()
    batch = synthetic_batch()
    with torch.no_grad():
        node_state, _, graph_index = model._node_and_graph_embeddings(batch)
        candidate = model._projection_candidate(batch, 0, node_state, graph_index)
        vector, _, coalition = model._mcts_project(
            candidate, model.prototype_vectors[0])

        assert vector is not None
        assert len(coalition) >= model.min_atoms
        direct = model._induced_graph_embedding(candidate, coalition)
        assert torch.allclose(vector, direct)
        adjacency = [set() for _ in range(candidate["x"].size(0))]
        for source, target in candidate["edge_index"].t().tolist():
            adjacency[source].add(target)
            adjacency[target].add(source)
        assert model._connected(coalition, adjacency)

        edgeless = dict(candidate)
        edgeless["edge_index"] = candidate["edge_index"][:, :0]
        edgeless["edge_attr"] = candidate["edge_attr"][:0]
        edgeless["edge_relation"] = candidate["edge_relation"][:0]
        edgeless["edge_triple"] = candidate["edge_triple"][:0]
        without_messages = model._induced_graph_embedding(edgeless, coalition)
        assert not torch.allclose(vector, without_messages)

        unmatched, score, empty_coalition = model._mcts_project(
            edgeless, model.prototype_vectors[0])
        assert unmatched is None
        assert score == float("-inf")
        assert empty_coalition == ()


def test_gsat_mask_is_applied_once_with_finite_ib_and_deterministic_eval():
    torch.manual_seed(13)
    model = adapter("gsat")
    batch = synthetic_batch()
    attention_calls = [[] for _ in model.predictor.layers]
    hooks = []
    for index, layer in enumerate(model.predictor.layers):
        def observe(module, args, kwargs, layer_index=index):
            attention_calls[layer_index].append(kwargs.get("edge_attention") is not None)

        hooks.append(layer.register_forward_pre_hook(observe, with_kwargs=True))

    model.eval()
    first = model(batch, epoch=0)
    first_attention = model.last_node_attention.detach().clone()
    second = model(batch, epoch=0)
    for hook in hooks:
        hook.remove()

    assert attention_calls == [[False, True, False, True] for _ in model.predictor.layers]
    assert torch.equal(first.logits, second.logits)
    assert torch.equal(first_attention, model.last_node_attention)
    assert torch.all((first_attention >= 0.0) & (first_attention <= 1.0))
    assert first.diagnostics["information_bottleneck"] >= 0.0
    assert model.r_for_epoch(0) == pytest.approx(0.9)
    assert model.r_for_epoch(10) == pytest.approx(0.8)
    assert model.r_for_epoch(20) == pytest.approx(0.7)
    assert model.r_for_epoch(99) == pytest.approx(0.7)
    assert model.run_config()["mechanism_settings"]["message_attention_applications"] == 1


def test_gsat_fractional_attention_scales_each_message_exactly_once():
    layer = _ClinicalGINLayer(hidden=2)
    layer.mlp = nn.Identity()
    with torch.no_grad():
        layer.eps.zero_()
    node_state = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    edge_index = torch.tensor([[0], [1]])
    edge_context = torch.tensor([[0.5, 1.0]])
    attention = torch.tensor([0.5])

    output = layer(node_state, edge_index, edge_context, edge_attention=attention)
    once = node_state.clone()
    once[1] = node_state[1] + (node_state[0] + edge_context[0]) * attention[0]
    squared = node_state.clone()
    squared[1] = node_state[1] + (node_state[0] + edge_context[0]) * attention[0].square()

    assert torch.allclose(output, once)
    assert not torch.allclose(output, squared)


def test_graphcare_membership_empty_visit_global_routing_and_gradients():
    torch.manual_seed(17)
    model = adapter("graphcare")
    model.eval()
    batch = synthetic_batch()

    first = model(batch, epoch=0)
    first_beta = model.last_beta.detach().clone()
    first_weight = model.last_visit_weight.detach().clone()
    first_graph = model.last_graph_state.detach().clone()
    first_direct = model.last_direct_state.detach().clone()
    first_global = model.last_global_state.detach().clone()
    assert first_beta[1].item() == 0.0
    assert torch.equal(model.last_visit_state[1], torch.zeros_like(model.last_visit_state[1]))
    assert first.diagnostics["empty_visits"] == 1.0
    assert first_global[0].abs().sum() > 0

    remapped = batch.clone()
    pair = torch.nonzero(
        (remapped.visit_membership_index[0] == 2)
        & (remapped.visit_membership_index[1] == 1),
        as_tuple=False,
    ).view(-1)
    assert pair.numel() == 1
    remapped.visit_membership_index[0, pair.item()] = 0
    model(remapped, epoch=0)
    assert not torch.allclose(first_beta, model.last_beta)
    assert not torch.allclose(first_weight, model.last_visit_weight)

    changed_global = batch.clone()
    changed_global.x[3] = changed_global.x[3] + 5.0
    changed = model(changed_global, epoch=0)
    assert torch.allclose(first_graph, model.last_graph_state)
    assert torch.allclose(first_direct, model.last_direct_state)
    assert not torch.allclose(first_global, model.last_global_state)
    assert not torch.allclose(first.logits, changed.logits)

    model.train()
    model.zero_grad(set_to_none=True)
    output = model(batch, epoch=0)
    (output.logits.sum() + output.auxiliary_loss).backward()
    assert_finite_gradient(model.layers[0].relation_gate.weight)
    assert_finite_gradient(model.alpha_projection[0].weight)
    assert_finite_gradient(model.beta_projection[0].weight)
    assert_finite_gradient(model.global_projection[0].weight)


def test_cli_defaults_reject_adapter_conv_and_bind_recursive_method_sources(tmp_path):
    base = [
        "--artifact", str(tmp_path / "artifact"),
        "--targets", str(tmp_path / "targets.csv"),
        "--output", str(tmp_path / "output"),
    ]
    parser = train.parser()
    legacy = train.normalize_method_args(parser.parse_args(base), parser)
    assert legacy.method == "clinical_gnn"
    assert legacy.conv == "edge_conditioned"
    assert legacy.hidden == 96
    assert legacy.layers == 3
    assert legacy.epochs == 12
    assert not (tmp_path / "output").exists()

    incompatible = parser.parse_args(base + ["--method", "gsat", "--conv", "hgt"])
    with pytest.raises(SystemExit):
        train.normalize_method_args(incompatible, parser)
    assert not (tmp_path / "output").exists()

    native_cases = (
        ("protgnn", ["--protgnn-warm-epochs", "2"], "warm_epochs", "warm_epochs", 2),
        ("gsat", ["--gsat-temperature", "0.7"], "gsat_temperature", "temperature", 0.7),
        ("graphcare", ["--graphcare-decay-rate", "0.2"],
         "graphcare_decay_rate", "recency_decay", 0.2),
    )
    for method, flags, destination, effective_key, expected in native_cases:
        configured = train.normalize_method_args(
            parser.parse_args(base + ["--method", method] + flags), parser)
        assert getattr(configured, destination) == expected
        assert configured._method_overrides == [destination]
        configured_adapter = build_method(
            method,
            num_tokens=NUM_TOKENS,
            node_dim=NODE_DIM,
            edge_dim=EDGE_DIM,
            num_classes=NUM_CLASSES,
            hidden=configured.hidden,
            layers=configured.layers,
            dropout=configured.dropout,
            token_dim=configured.token_dim,
            num_triples=NUM_TRIPLES,
            args=configured,
        )
        method_config = configured_adapter.run_config()
        assert "native_defaults" in method_config
        assert method_config["effective_settings"][effective_key] == expected

    incompatible_native = parser.parse_args(base + ["--gsat-temperature", "0.7"])
    with pytest.raises(SystemExit):
        train.normalize_method_args(incompatible_native, parser)
    assert not (tmp_path / "output").exists()

    sources = recursive_source_hashes(paths.PACKAGE_ROOT)
    for required in (
        "methods/base.py",
        "methods/protgnn/adapter.py",
        "methods/gsat/adapter.py",
        "methods/graphcare/adapter.py",
    ):
        assert required in sources
        assert len(sources[required]) == 64


def test_protgnn_early_stopping_waits_for_first_projection():
    model = adapter("protgnn", proj_epochs=20, patience=10)

    assert train.early_stopping_start_epoch("protgnn", model) == 20
    assert train.early_stopping_start_epoch("gsat", adapter("gsat")) == 0


def test_aggregation_keeps_seed_outcomes_out_of_cohort_identity_and_accepts_budget():
    digest = "a" * 64

    def row(seed, epoch, prediction_hash):
        macro_f1 = 0.1 + seed / 100.0
        return {
            "status": "completed",
            "selected_epoch": epoch,
            "proba_sha256": prediction_hash,
            "metrics": {"macro_f1": macro_f1},
            "binding": {
                "method": "gsat",
                "seed": seed,
                "epochs": 10,
                "parameter_count": 123,
                "counts": {"train": 4, "validation": 2},
                "class_order": ["A", "B"],
                "selection": "validation macro-F1",
                "test_evaluated": False,
                "evaluation_fold": "validation",
                "logic_contract_version": "clinical_graph_logic_v2",
                "evaluation_version": "visit_targets_patient_equal_v2",
                "preprocessing_schema_version": "clinical_inputs_v3",
                "visit_membership_contract_version": "clinical_visit_membership_v1",
                "artifact_graphs_sha256": digest,
                "artifact_visit_membership_sha256": digest,
                "targets_sha256": digest,
                "target_binding_sha256": digest,
                "preprocessing_sha256": digest,
                "source_code": {"train.py": digest, "methods/gsat.py": digest},
                "split_sample_ids_sha256": {
                    "train": digest,
                    "validation": digest,
                },
                "selected_validation": {
                    "epoch": epoch,
                    "epoch_index": epoch - 1,
                    "metric": "macro_f1",
                    "metric_value": macro_f1,
                    "prediction_sha256": prediction_hash,
                    "sample_ids_sha256": digest,
                },
            },
        }

    rows = {
        "arm_seed1": row(1, 3, "b" * 64),
        "arm_seed2": row(2, 7, "c" * 64),
    }
    assert aggregate._contract(rows["arm_seed1"]) == aggregate._contract(rows["arm_seed2"])
    report = aggregate.aggregate_rows(rows, base="arm")
    assert len(report["cohorts"]) == 1
    cohort = report["cohorts"][0]
    assert cohort["missing_bindings"] == []
    assert cohort["arms"]["arm"]["n_runs"] == 2

    missing = copy.deepcopy(rows)
    missing["arm_seed1"]["binding"].pop("selected_validation")
    missing_report = aggregate.aggregate_rows(missing, base="arm")
    assert missing_report["cohorts"][0]["comparison_eligible"] is False
    assert any("selected_validation.missing_or_malformed" in issue
               for issue in missing_report["cohorts"][0]["missing_bindings"])

    mismatched = copy.deepcopy(rows)
    mismatched["arm_seed2"]["binding"]["selected_validation"][
        "prediction_sha256"] = "d" * 64
    mismatch_report = aggregate.aggregate_rows(mismatched, base="arm")
    assert mismatch_report["cohorts"][0]["comparison_eligible"] is False
    assert any("selected_validation.prediction_sha256" in issue
               for issue in mismatch_report["cohorts"][0]["missing_bindings"])
