"""Focused smoke and benchmark-contract tests for GSAT."""

import importlib
import inspect
import os

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import InstanceNorm

from gsat_analysis import train
from gsat_analysis.explainability.graphxai_wrapper import GSATGraphXAIWrapper
from gsat_analysis.models import ExtractorMLP, GIN, GSAT


def _synthetic_batch():
    graph_a = Data(
        x=torch.randn(4, 5),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]],
            dtype=torch.long,
        ),
        y=torch.tensor([0]),
    )
    graph_b = Data(
        x=torch.randn(3, 5),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        ),
        y=torch.tensor([2]),
    )
    return Batch.from_data_list([graph_a, graph_b])


def _build_tiny_gsat():
    classifier = GIN(
        x_dim=5,
        num_class=3,
        hidden_dim=8,
        num_layers=2,
        dropout=0.0,
        readout="mean",
    )
    extractor = ExtractorMLP(8, attention_level="node", dropout=0.0)
    return GSAT(classifier, extractor, attention_level="node")


def test_gsat_explainability_entry_points_import():
    importlib.import_module("gsat_analysis.explainability.explain_gsat")
    importlib.import_module("gsat_analysis.explainability.summarize_gsat")


def test_gsat_main_exposes_standardized_benchmark_signature():
    assert list(inspect.signature(train.main).parameters) == [
        "graph_structure",
        "max_epochs",
        "limit",
        "out_dir",
        "canonical_split",
        "seed",
        "loss_weighting",
    ]


def test_gsat_main_requires_canonical_split(tmp_path):
    with pytest.raises(ValueError, match="canonical split"):
        train.main("star", 1, 1, tmp_path, None, 1234)


def test_gsat_main_requires_output_directory(tmp_path):
    split_path = tmp_path / "split.json"
    split_path.write_text("{}")
    with pytest.raises(ValueError, match="output directory"):
        train.main("star", 1, 1, None, split_path, 1234)


@pytest.mark.parametrize(
    ("max_epochs", "limit", "field"),
    [
        (0, 1, "max_epochs"),
        (-1, 1, "max_epochs"),
        (True, 1, "max_epochs"),
        (1.0, 1, "max_epochs"),
        (1, 0, "limit"),
        (1, -1, "limit"),
        (1, True, "limit"),
        (1, 1.0, "limit"),
    ],
)
def test_gsat_main_rejects_non_positive_or_non_exact_integer_controls(
    tmp_path, max_epochs, limit, field
):
    with pytest.raises(ValueError, match=rf"{field}.*positive exact integer"):
        train.main(
            "star", max_epochs, limit, tmp_path / "out", tmp_path / "split.json", 1234
        )


def test_gsat_main_rejects_nonbenchmark_seed(tmp_path, monkeypatch):
    split_path = tmp_path / "split.json"
    split_path.write_text("{}")
    monkeypatch.setattr(
        train,
        "get_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dataset access must follow seed validation")
        ),
    )
    with pytest.raises(ValueError, match="seed must be one of"):
        train.main("star", 1, 1, tmp_path / "out", split_path, 7)


def test_gsat_main_validates_canonical_split_before_data_access(tmp_path, monkeypatch):
    split_path = tmp_path / "split.json"
    split_path.write_text("{}")
    monkeypatch.setattr(
        train,
        "get_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dataset access must follow split validation")
        ),
    )
    with pytest.raises(ValueError, match="missing required fields"):
        train.main("star", 1, 1, tmp_path / "out", split_path, 1234)


def test_gsat_main_sets_requested_seed_unconditionally(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(train, "load_canonical_split", lambda path: {})
    monkeypatch.setattr(train, "set_seed", calls.append)
    monkeypatch.setattr(
        train,
        "get_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )

    with pytest.raises(RuntimeError, match="stop"):
        train.main("star", 1, 1, tmp_path / "out", tmp_path / "split.json", 1235)

    assert calls == [1235]


@pytest.mark.parametrize("previous", [None, "previous-split.json"])
def test_gsat_main_scopes_canonical_split_environment(tmp_path, monkeypatch, previous):
    split_path = tmp_path / "split.json"
    key = "CANONICAL_SPLIT_JSON"
    if previous is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, previous)
    monkeypatch.setattr(train, "load_canonical_split", lambda path: {})
    monkeypatch.setattr(train, "set_seed", lambda seed: None)

    def stop_after_observing_environment(*args, **kwargs):
        assert os.environ[key] == str(split_path)
        raise RuntimeError("stop")

    monkeypatch.setattr(train, "get_dataset", stop_after_observing_environment)

    with pytest.raises(RuntimeError, match="stop"):
        train.main("star", 1, 1, tmp_path / "out", split_path, 1234)

    assert os.environ.get(key) == previous


def test_gsat_main_passes_requested_seed_to_loader(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(train, "load_canonical_split", lambda path: {})
    monkeypatch.setattr(train, "set_seed", lambda seed: None)
    monkeypatch.setattr(train, "get_dataset", lambda *args, **kwargs: [object()])

    def capture_loader_seed(*args, **kwargs):
        observed.append(kwargs["seed"])
        raise RuntimeError("stop")

    monkeypatch.setattr(train, "get_dataloader", capture_loader_seed)

    with pytest.raises(RuntimeError, match="stop"):
        train.main("star", 1, None, tmp_path / "out", tmp_path / "split.json", 1236)

    assert observed == [1236]


def test_gsat_passes_canonical_split_into_dataset_construction(tmp_path, monkeypatch):
    split_path = tmp_path / "split.json"
    observed = []
    monkeypatch.setattr(train, "load_canonical_split", lambda path: {})
    monkeypatch.setattr(train, "set_seed", lambda seed: None)

    def capture_dataset(*args, **kwargs):
        observed.append(kwargs.get("canonical_split"))
        raise RuntimeError("stop")

    monkeypatch.setattr(train, "get_dataset", capture_dataset)

    with pytest.raises(RuntimeError, match="stop"):
        train.main("star", 1, 1, tmp_path / "out", split_path, 1234)

    assert observed == [split_path]


def test_gsat_applies_limit_after_canonical_split_construction(tmp_path, monkeypatch):
    dataset = [object(), object(), object(), object()]
    monkeypatch.setattr(train, "load_canonical_split", lambda path: {})
    monkeypatch.setattr(train, "set_seed", lambda seed: None)
    monkeypatch.setattr(train, "get_dataset", lambda *args, **kwargs: dataset)

    def observe_dataset(candidate, **kwargs):
        assert candidate is dataset
        raise RuntimeError("stop")

    monkeypatch.setattr(train, "get_dataloader", observe_dataset)

    with pytest.raises(RuntimeError, match="stop"):
        train.main("star", 1, 1, tmp_path / "out", tmp_path / "split.json", 1234)


def test_gsat_limit_caps_each_canonical_fold_without_reassignment():
    graphs = [object() for _ in range(6)]
    loaders = {
        "train": DataLoader([graphs[3], graphs[0]], batch_size=2, shuffle=True),
        "eval": DataLoader([graphs[4], graphs[1]], batch_size=2, shuffle=False),
        "test": DataLoader([graphs[5], graphs[2]], batch_size=2, shuffle=False),
    }

    limited = train._limit_fold_loaders(loaders, 1)

    assert {name: len(loader.dataset) for name, loader in limited.items()} == {
        "train": 1,
        "eval": 1,
        "test": 1,
    }
    assert limited["train"].dataset[0] is graphs[3]
    assert limited["eval"].dataset[0] is graphs[4]
    assert limited["test"].dataset[0] is graphs[5]


def test_gsat_cli_requires_and_forwards_standardized_arguments(tmp_path, monkeypatch):
    split_path = tmp_path / "split.json"
    out_dir = tmp_path / "out"
    calls = []
    monkeypatch.setattr(
        train, "main", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    train.cli(
        [
            "--graph_structure",
            "cooccur",
            "--max_epochs",
            "2",
            "--limit",
            "12",
            "--out_dir",
            str(out_dir),
            "--canonical_split",
            str(split_path),
            "--seed",
            "1236",
        ]
    )

    assert calls == [
        (("cooccur", 2, 12, out_dir, split_path, 1236), {"loss_weighting": "none"})
    ]


def test_gsat_selects_highest_validation_macro_f1_checkpoint(tmp_path, monkeypatch):
    class TinyTrainableGSAT(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, data, epoch=0, training=True):
            loss = -self.weight
            return {
                "loss": loss,
                "pred_loss": loss,
                "info_loss": self.weight * 0,
            }

        def get_r(self, epoch):
            return 0.9

    graph = Data(
        x=torch.ones(3, 5),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        y=torch.tensor([0]),
    )
    loader = DataLoader([graph], batch_size=1)
    test_loader = DataLoader([graph], batch_size=1)
    model = TinyTrainableGSAT()
    validation_weights = []
    validation_scores = iter([0.1, 0.3, 0.2])
    selected_weight = []

    def fake_evaluate(candidate, loader_arg, device, epoch):
        metrics = {
            "accuracy": 0.0,
            "balanced_acc": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
        }
        if loader_arg is loader:
            validation_weights.append(candidate.weight.detach().clone())
            metrics["macro_f1"] = next(validation_scores)
        else:
            selected_weight.append(candidate.weight.detach().clone())
        return metrics

    monkeypatch.setattr(train, "load_canonical_split", lambda path: {})
    monkeypatch.setattr(train.cfg, "device", "cpu")
    legacy_output_dir = tmp_path / "legacy-output"
    monkeypatch.setattr(train.cfg, "OUTPUTS_DIR", legacy_output_dir)
    monkeypatch.setattr(train, "get_dataset", lambda *args, **kwargs: [graph])
    monkeypatch.setattr(
        train,
        "get_dataloader",
        lambda *args, **kwargs: {
            "train": loader,
            "eval": loader,
            "test": test_loader,
        },
    )
    monkeypatch.setattr(train, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(train, "evaluate", fake_evaluate)

    train.main("star", 3, None, tmp_path / "out", tmp_path / "split.json", 1234)

    assert torch.equal(selected_weight[0], validation_weights[1])
    assert {path.name for path in (tmp_path / "out").iterdir()} == {
        "gsat_model.pt",
        "metrics.json",
        "report.txt",
    }
    assert not legacy_output_dir.exists()


def test_extractor_uses_graph_aware_instance_norm_and_is_batch_composition_invariant():
    torch.manual_seed(1234)
    extractor = ExtractorMLP(4, attention_level="node", dropout=0.0).eval()
    graph_emb = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 1.0, 3.0], [5.0, 1.0, 2.0, 0.0]]
    )
    other_emb = torch.tensor(
        [[100.0, -100.0, 50.0, -50.0], [200.0, -200.0, 75.0, -75.0]]
    )
    graph_edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    combined_edges = torch.cat(
        (graph_edges, torch.tensor([[3, 4], [4, 3]], dtype=torch.long)), dim=1
    )

    alone = extractor(
        graph_emb, graph_edges, torch.zeros(3, dtype=torch.long)
    )
    combined = extractor(
        torch.cat((graph_emb, other_emb)),
        combined_edges,
        torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )

    assert any(isinstance(module, InstanceNorm) for module in extractor.modules())
    assert torch.allclose(alone, combined[:3], atol=1e-6)


def test_gsat_synthetic_forward_backward():
    torch.manual_seed(1234)
    model = _build_tiny_gsat()
    batch = _synthetic_batch()

    result = model(batch, epoch=0, training=True)

    assert result["logits"].shape == (2, 3)
    assert result["node_att"].shape == (7,)
    assert result["edge_att"].shape == (14,)
    assert torch.isfinite(result["loss"])

    result["loss"].backward()

    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_graphxai_wrapper_is_deterministically_faithful():
    torch.manual_seed(1234)
    model = _build_tiny_gsat().eval()
    batch = _synthetic_batch()
    wrapper = GSATGraphXAIWrapper(model)

    wrapper.set_context(batch.x, batch.edge_index, batch.batch)
    first = wrapper(batch.x, batch.edge_index)
    second = wrapper(batch.x, batch.edge_index)
    faithful, max_abs_diff = wrapper.verify(
        batch.x, batch.edge_index, batch.batch, atol=0.0
    )

    assert torch.equal(first, second)
    assert faithful
    assert max_abs_diff == 0.0
