"""Behavioral contract tests shared by benchmark method entry points."""

import os
import sys
import types

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graphcare_analysis import run as graphcare_run
from gsat_analysis import train as gsat_train
from protgnn_analysis import train as protgnn_train
from protgnn_analysis.models.GCN import GCNNet
from shared.lib.metrics import multiclass_metrics


def test_all_methods_use_the_shared_multiclass_metrics_implementation():
    assert protgnn_train.multiclass_metrics is multiclass_metrics
    assert gsat_train.multiclass_metrics is multiclass_metrics
    assert graphcare_run.multiclass_metrics is multiclass_metrics
    assert not hasattr(protgnn_train, "_multiclass_metrics")


def _thirty_class_predictions():
    labels = np.arange(6, dtype=np.int64)
    probs = np.zeros((6, 30), dtype=np.float64)
    rankings = [
        [0],
        [2, 1],
        [2],
        [4, 0, 1, 3],
        [4],
        [0, 1, 2, 3, 4, 5],
    ]
    for row, ranking in enumerate(rankings):
        for rank, class_id in enumerate(ranking):
            probs[row, class_id] = 1.0 - rank * 0.1
    preds = probs.argmax(axis=1)
    return labels, preds, probs


def test_shared_multiclass_metrics_has_exact_six_metric_schema_and_values():
    labels, preds, probs = _thirty_class_predictions()

    metrics = multiclass_metrics(labels, preds, probs)

    assert metrics == {
        "accuracy": 0.5,
        "balanced_acc": 0.5,
        "macro_f1": 0.333333,
        "micro_f1": 0.5,
        "top3_acc": 0.666667,
        "top5_acc": 0.833333,
    }


def test_method_evaluators_emit_the_same_six_multiclass_metrics():
    labels, preds, probs = _thirty_class_predictions()
    logits = torch.tensor(np.log(probs + 1e-6), dtype=torch.float32)
    expected = multiclass_metrics(labels, preds, probs)
    contract_keys = set(expected)

    class GraphCareModel(torch.nn.Module):
        def forward(self, *args):
            return logits

    graphcare_batch = {
        "node_ids": torch.zeros(1, dtype=torch.long),
        "rel_ids": torch.zeros(1, dtype=torch.long),
        "edge_index": torch.zeros((2, 1), dtype=torch.long),
        "batch": torch.zeros(1, dtype=torch.long),
        "visit_node": torch.zeros(1),
        "ehr_nodes": torch.zeros(1),
        "y": torch.tensor(labels),
    }
    graphcare = graphcare_run._evaluate(GraphCareModel(), [graphcare_batch], "cpu")

    class GsatModel(torch.nn.Module):
        def forward(self, data, epoch, training):
            return {"logits": logits}

    gsat = gsat_train.evaluate(
        GsatModel(), [Data(x=torch.zeros((6, 1)), y=torch.tensor(labels))], "cpu", 0
    )
    protgnn = protgnn_train._state_from_predictions(
        0.25, labels, preds, probs
    )

    for state in (graphcare, gsat, protgnn):
        assert {key: state[key] for key in contract_keys} == expected
        assert "acc" not in state


def test_graphcare_selects_sub_min_delta_exact_best_checkpoint(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    model_file = upstream / "graphcare_" / "model.py"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("", encoding="utf-8")

    class GraphCare(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, *args):
            return torch.zeros((1, 30)) + self.marker * 0.0

    fake_module = types.ModuleType("graphcare_.model")
    setattr(fake_module, "GraphCare", GraphCare)
    monkeypatch.setitem(sys.modules, "graphcare_.model", fake_module)
    monkeypatch.setattr(graphcare_run.cfg, "UPSTREAM_DIR", upstream)
    monkeypatch.setattr(graphcare_run, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        graphcare_run,
        "_load_kg",
        lambda *a, **k: {"num_nodes": 2, "num_rels": 3},
    )
    val_loader = object()
    test_loader = object()
    monkeypatch.setattr(
        graphcare_run,
        "build_loaders",
        lambda *a, **k: ([], val_loader, test_loader, 30, {}),
    )
    monkeypatch.setattr(graphcare_run, "_write_report", lambda *a, **k: None)
    validation_scores = iter((0.4, 0.403))

    def synthetic_evaluate(model, loader, device):
        if loader is val_loader:
            score = next(validation_scores)
            model.marker.data.fill_(1.0 if score == 0.4 else 2.0)
            return {"macro_f1": score, "accuracy": score}
        if loader is test_loader:
            return {
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "accuracy": 0.0,
                "selected_marker": float(model.marker.detach()),
            }
        raise AssertionError("unexpected loader")

    monkeypatch.setattr(graphcare_run, "_evaluate", synthetic_evaluate)

    result = graphcare_run.main(
        max_epochs=2,
        patience=10,
        min_delta=0.005,
        out_dir=tmp_path / "out",
        graph_structure="star",
    )

    assert result["selected_marker"] == 2.0


def test_protgnn_partial_standardized_arguments_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        protgnn_train,
        "train_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training must not start for a partial standardized invocation")
        ),
    )

    with pytest.raises(ValueError, match="requires all of"):
        protgnn_train.main(
            graph_structure="star",
            canonical_split=tmp_path / "split.json",
            seed=1234,
            output_dir=None,
        )


def test_protgnn_standardized_values_win_and_only_target_supplied_output(
    tmp_path, monkeypatch
):
    split_path = tmp_path / "canonical.json"
    output_dir = tmp_path / "standardized"
    events = []
    original = (
        protgnn_train.data_args.dataset_name,
        protgnn_train.data_args.graph_structure,
        protgnn_train.data_args.seed,
        protgnn_train.model_args.checkpoint,
        protgnn_train.model_args.enable_prot,
        protgnn_train.RESULTS_DIR,
        os.environ.get("CANONICAL_SPLIT_JSON"),
    )
    monkeypatch.setattr(
        protgnn_train,
        "load_canonical_split",
        lambda path: events.append(("validate", path)) or {},
    )
    monkeypatch.setattr(
        protgnn_train,
        "set_seed",
        lambda seed: events.append(("seed", seed)),
    )

    def fake_train_model(*args, **kwargs):
        checkpoint_root = kwargs["checkpoint_root"]
        checkpoint_root.mkdir(parents=True)
        (checkpoint_root / "synthetic-best.pt").write_bytes(b"checkpoint")
        events.append(
            (
                "train",
                protgnn_train.data_args.dataset_name,
                protgnn_train.data_args.graph_structure,
                protgnn_train.data_args.seed,
                kwargs,
            )
        )
        return torch.nn.Linear(1, 1), [], {"macro_f1": 0.5}, 30, [], None, {}

    monkeypatch.setattr(protgnn_train, "train_model", fake_train_model)
    monkeypatch.setattr(
        protgnn_train,
        "explain_test_set",
        lambda *args, **kwargs: events.append(("explain", kwargs)) or [],
    )
    def fake_save_results(*args, **kwargs):
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["output_dir"] / "metrics.json").write_text("{}", encoding="utf-8")
        events.append(("save", kwargs))

    monkeypatch.setattr(protgnn_train, "save_results", fake_save_results)
    monkeypatch.setattr(
        protgnn_train,
        "archive_results",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("standardized mode must not archive into legacy outputs")
        ),
    )

    protgnn_train.main(
        graph_structure="cooccur",
        canonical_split=split_path,
        seed=1235,
        output_dir=output_dir,
    )

    assert events[0] == ("validate", split_path)
    assert events[1] == ("seed", 1235)
    train_event = events[2]
    assert train_event[1:4] == (
        "mimic_intra_patient_disease",
        "cooccur",
        1235,
    )
    assert train_event[4]["canonical_split"] == split_path
    assert train_event[4]["checkpoint_root"] == output_dir / "checkpoints"
    assert train_event[4]["standardized"] is True
    assert events[3][0] == "save"
    assert events[3][1]["output_dir"] == output_dir
    assert events[3][1]["canonical_split"] == split_path
    assert not any(event[0] == "explain" for event in events)
    assert {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file()
    } == {"checkpoints/synthetic-best.pt", "metrics.json"}
    assert (
        protgnn_train.data_args.dataset_name,
        protgnn_train.data_args.graph_structure,
        protgnn_train.data_args.seed,
        protgnn_train.model_args.checkpoint,
        protgnn_train.model_args.enable_prot,
        protgnn_train.RESULTS_DIR,
        os.environ.get("CANONICAL_SPLIT_JSON"),
    ) == original


def test_protgnn_sequential_no_prot_then_prot_constructs_requested_model_modes(
    tmp_path, monkeypatch
):
    split_path = tmp_path / "split.json"
    constructed_modes = []
    monkeypatch.setattr(protgnn_train, "load_canonical_split", lambda path: {})
    monkeypatch.setattr(protgnn_train, "set_seed", lambda seed: None)

    def construct_real_model(*args, **kwargs):
        model = GCNNet(2, 2, protgnn_train.model_args)
        constructed_modes.append(model.enable_prot)
        return model, [], {}, 2, [], None, {}

    monkeypatch.setattr(protgnn_train, "train_model", construct_real_model)
    monkeypatch.setattr(protgnn_train, "explain_test_set", lambda *a, **k: [])
    monkeypatch.setattr(protgnn_train, "save_results", lambda *a, **k: None)
    original_mode = protgnn_train.model_args.enable_prot

    protgnn_train.main("star", split_path, 1234, tmp_path / "plain", no_prot=True)
    assert protgnn_train.model_args.enable_prot is original_mode
    protgnn_train.main("star", split_path, 1234, tmp_path / "prototype")

    assert constructed_modes == [False, True]
    assert protgnn_train.model_args.enable_prot is original_mode


def test_protgnn_rejects_seed_before_split_or_training_access(tmp_path, monkeypatch):
    monkeypatch.setattr(
        protgnn_train,
        "load_canonical_split",
        lambda path: (_ for _ in ()).throw(
            AssertionError("split access must follow exact seed validation")
        ),
    )
    monkeypatch.setattr(
        protgnn_train,
        "train_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training must follow exact seed validation")
        ),
    )

    with pytest.raises(ValueError, match="seed must be one of"):
        protgnn_train.main("star", tmp_path / "split.json", 7, tmp_path / "out")


def test_protgnn_validates_split_before_seeding_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        protgnn_train,
        "load_canonical_split",
        lambda path: (_ for _ in ()).throw(ValueError("invalid canonical split")),
    )
    monkeypatch.setattr(
        protgnn_train,
        "set_seed",
        lambda seed: (_ for _ in ()).throw(
            AssertionError("seeding must follow split validation")
        ),
    )
    monkeypatch.setattr(
        protgnn_train,
        "train_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("training must follow split validation")
        ),
    )

    with pytest.raises(ValueError, match="invalid canonical split"):
        protgnn_train.main(
            "star", tmp_path / "split.json", 1234, tmp_path / "out"
        )


@pytest.mark.parametrize("previous", [None, "previous-split.json"])
def test_protgnn_restores_all_mutable_state_and_environment_after_exception(
    tmp_path, monkeypatch, previous
):
    key = "CANONICAL_SPLIT_JSON"
    split_path = tmp_path / "split.json"
    if previous is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, previous)
    monkeypatch.setattr(protgnn_train.model_args, "enable_prot", False)
    monkeypatch.setattr(protgnn_train, "load_canonical_split", lambda path: {})
    monkeypatch.setattr(protgnn_train, "set_seed", lambda seed: None)
    original = (
        protgnn_train.data_args.dataset_name,
        protgnn_train.data_args.graph_structure,
        protgnn_train.data_args.seed,
        protgnn_train.model_args.checkpoint,
        protgnn_train.model_args.enable_prot,
        protgnn_train.RESULTS_DIR,
        previous,
    )

    def observe_and_stop(*args, **kwargs):
        assert os.environ[key] == str(split_path)
        assert protgnn_train.model_args.enable_prot is True
        protgnn_train.RESULTS_DIR = str(tmp_path / "mutated-results")
        raise RuntimeError("stop")

    monkeypatch.setattr(protgnn_train, "train_model", observe_and_stop)

    with pytest.raises(RuntimeError, match="stop"):
        protgnn_train.main("star", split_path, 1234, tmp_path / "out")

    assert (
        protgnn_train.data_args.dataset_name,
        protgnn_train.data_args.graph_structure,
        protgnn_train.data_args.seed,
        protgnn_train.model_args.checkpoint,
        protgnn_train.model_args.enable_prot,
        protgnn_train.RESULTS_DIR,
        os.environ.get(key),
    ) == original


def test_protgnn_legacy_no_standardized_arguments_still_runs(tmp_path, monkeypatch):
    from shared.lib import graph_structures

    seeds = []
    saves = []
    monkeypatch.setattr(graph_structures, "prompt_for_structure", lambda method: "star")
    monkeypatch.setattr(protgnn_train, "set_seed", seeds.append)
    monkeypatch.setattr(
        protgnn_train,
        "train_model",
        lambda *args, **kwargs: (torch.nn.Linear(1, 1), [], {}, 2, [], None, {}),
    )
    monkeypatch.setattr(protgnn_train, "explain_test_set", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        protgnn_train,
        "save_results",
        lambda *args, **kwargs: saves.append(kwargs),
    )

    protgnn_train.main(no_archive=True)

    assert seeds == [protgnn_train.data_args.seed]
    assert saves == [{"threshold": None, "output_dir": None, "canonical_split": None}]


def test_protgnn_legacy_graph_and_seed_overrides_remain_available(monkeypatch):
    seeds = []
    observed = []
    monkeypatch.setattr(protgnn_train, "set_seed", seeds.append)

    def fake_train(*args, **kwargs):
        observed.append(
            (protgnn_train.data_args.graph_structure, protgnn_train.data_args.seed)
        )
        return torch.nn.Linear(1, 1), [], {}, 2, [], None, {}

    monkeypatch.setattr(protgnn_train, "train_model", fake_train)
    monkeypatch.setattr(protgnn_train, "explain_test_set", lambda *args, **kwargs: [])
    monkeypatch.setattr(protgnn_train, "save_results", lambda *args, **kwargs: None)

    protgnn_train.main(graph_structure="ontology", seed=99, no_archive=True)

    assert seeds == [99]
    assert observed == [("ontology", 99)]


def test_protgnn_cli_accepts_hyphenated_standardized_arguments(tmp_path, monkeypatch):
    calls = []
    split_path = tmp_path / "split.json"
    output_dir = tmp_path / "out"
    monkeypatch.setattr(protgnn_train, "main", lambda **kwargs: calls.append(kwargs))

    protgnn_train.cli(
        [
            "--graph-structure",
            "ontology",
            "--canonical-split",
            str(split_path),
            "--seed",
            "1236",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert calls[0]["graph_structure"] == "ontology"
    assert calls[0]["canonical_split"] == split_path
    assert calls[0]["seed"] == 1236
    assert calls[0]["output_dir"] == output_dir


def test_protgnn_standardized_results_directory_does_not_touch_legacy_pointer(
    tmp_path, monkeypatch
):
    supplied = tmp_path / "supplied"
    legacy = tmp_path / "legacy"
    monkeypatch.setattr(protgnn_train, "RESULTS_BASE", str(legacy))

    protgnn_train._prepare_results_dir(supplied)

    assert supplied.is_dir()
    assert (supplied / "explanations").is_dir()
    assert not legacy.exists()


def test_protgnn_standardized_result_files_and_report_stay_in_supplied_directory(
    tmp_path, monkeypatch
):
    supplied = tmp_path / "supplied"
    legacy = tmp_path / "legacy"
    monkeypatch.setattr(protgnn_train, "RESULTS_BASE", str(legacy))
    monkeypatch.setattr(
        protgnn_train,
        "get_dataset",
        lambda *a, **k: types.SimpleNamespace(
            feature_cols=[], feature_metadata={}, label_mapping={"0": "class-a"}
        ),
    )
    metrics = {
        "accuracy": 0.5,
        "balanced_acc": 0.5,
        "macro_f1": 0.5,
        "micro_f1": 0.5,
        "top3_acc": 1.0,
        "top5_acc": 1.0,
    }
    epoch = {
        "epoch": 0,
        "eval_acc": 0.5,
        "eval_pr_auc": "",
        "eval_macro_f1": 0.5,
    }

    protgnn_train.save_results(
        [epoch],
        list(epoch),
        metrics,
        {},
        [],
        0.1,
        0.1,
        30,
        use_prot=True,
        output_dir=supplied,
        canonical_split=tmp_path / "split.json",
    )

    expected_files = {
        "model_config.json",
        "dataset_metadata.json",
        "training_metrics.csv",
        "test_metrics.json",
        "evaluation_diagnostics.json",
        "explanations/explanations_summary.csv",
        "report.txt",
    }
    assert {
        str(path.relative_to(supplied)) for path in supplied.rglob("*") if path.is_file()
    } == expected_files
    assert not legacy.exists()
    report = (supplied / "report.txt").read_text(encoding="utf-8")
    for relative_path in expected_files:
        assert str(supplied / relative_path) in report
    assert "outputs/results/" not in report


def test_protgnn_multiclass_checkpoint_score_is_validation_macro_f1():
    assert protgnn_train._validation_selection_score(
        {"acc": 0.9, "macro_f1": 0.4}
    ) == 0.4


def test_protgnn_standardized_mode_loads_best_validation_checkpoint():
    selected, label = protgnn_train._checkpoint_to_load(
        best="best.pth",
        latest="latest.pth",
        use_prot=True,
        epoch_rows=[{"epoch": 50}],
        proj_epochs=20,
        standardized=True,
    )

    assert selected == "best.pth"
    assert label == "best validation"
