import hashlib
import json
from pathlib import Path

import pytest

from shared.lib.benchmark_contract import (
    ALLOWED_SEEDS,
    EXPECTED_CLASSES,
    EXPECTED_FOLD_COUNTS,
    BenchmarkSpec,
)
from shared.lib.run_manifest import RunManifest


def _valid_test_metrics(parameter_count=42):
    return {
        "accuracy": 0.1,
        "balanced_acc": 0.2,
        "macro_f1": 0.3,
        "micro_f1": 0.4,
        "top3_acc": 0.5,
        "top5_acc": 0.6,
        "parameter_count": parameter_count,
    }


def _nested_metrics_payload(parameter_count=42):
    return {
        "parameter_count": parameter_count,
        "test": _valid_test_metrics(parameter_count),
    }


def _start_manifest(tmp_path, monkeypatch, method="protgnn"):
    dataset = tmp_path / "merged_ed.csv"
    split = tmp_path / "canonical_split.json"
    dataset.write_text("subject_id,disease_1\n1,A\n", encoding="utf-8")
    split.write_text('{"fold": {}, "classes": []}\n', encoding="utf-8")
    run_dir = tmp_path / "results" / method / "star" / "seed_1234"
    run_dir.mkdir(parents=True)
    replacements = []
    from shared.lib import run_manifest as module

    real_replace = module.os.replace

    def recording_replace(source, destination, **kwargs):
        replacements.append((Path(source), Path(destination), kwargs))
        real_replace(source, destination, **kwargs)

    monkeypatch.setattr(module.os, "replace", recording_replace)
    manifest = RunManifest.start(
        run_dir / "run_manifest.json",
        spec=BenchmarkSpec(method, "star", 1234),
        dataset_path=dataset,
        split_path=split,
        topology_parameters={"patient_hub": True},
        canonical_fold_counts={"train": 59607, "validation": 7448, "test": 7456},
        effective_fold_counts={"train": 59607, "validation": 7448, "test": 7456},
        class_ordering=[f"class-{index}" for index in range(30)],
        parameter_count=None,
        device={"type": "cpu"},
        package_versions={"python": "3.9.0", "torch": "unavailable"},
        git_commit="a" * 40,
        git_dirty=True,
        command=["python3", "-m", "protgnn_analysis.train", "--seed", "1234"],
    )
    return manifest, run_dir, replacements


def test_manifest_lifecycle_writes_complete_audit_record_atomically(tmp_path, monkeypatch):
    manifest, run_dir, replacements = _start_manifest(tmp_path, monkeypatch)

    started = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert started["schema_version"] == 1
    assert (started["method"], started["topology"], started["seed"]) == (
        "protgnn", "star", 1234
    )
    assert started["dataset"]["path"].endswith("merged_ed.csv")
    assert len(started["dataset"]["sha256"]) == 64
    assert started["split"]["path"].endswith("canonical_split.json")
    assert len(started["split"]["sha256"]) == 64
    assert started["topology_parameters"] == {"patient_hub": True}
    canonical_policy = json.dumps(
        started["topology_parameters"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert started["topology_policy_fingerprint"] == hashlib.sha256(
        canonical_policy
    ).hexdigest()
    assert started["canonical_fold_counts"] == {
        "train": 59607,
        "validation": 7448,
        "test": 7456,
    }
    assert started["effective_fold_counts"] == started["canonical_fold_counts"]
    assert "fold_counts" not in started
    assert len(started["class_ordering"]) == 30
    assert started["parameter_count"] is None
    assert started["device"] == {"type": "cpu"}
    assert started["package_versions"]["python"] == "3.9.0"
    assert started["git"] == {"commit": "a" * 40, "dirty": True}
    assert started["command"][-2:] == ["--seed", "1234"]
    assert started["timestamps"]["started_at"].endswith("Z")
    assert started["timestamps"]["ended_at"] is None
    assert started["status"] == "running"
    assert started["metrics_path"] is None
    assert started["checkpoint_path"] is None

    metrics = run_dir / "metrics.json"
    checkpoint = run_dir / "model.pt"
    metrics.write_text(
        json.dumps(_valid_test_metrics()), encoding="utf-8"
    )
    checkpoint.write_bytes(b"checkpoint")
    manifest.complete(metrics, checkpoint)

    completed = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["timestamps"]["ended_at"].endswith("Z")
    assert completed["metrics_path"] == "metrics.json"
    assert completed["checkpoint_path"] == "model.pt"
    assert completed["parameter_count"] == 42
    assert all(
        source.parent == destination.parent == Path(".")
        and details["src_dir_fd"] == details["dst_dir_fd"]
        for source, destination, details in replacements
    )
    assert not list(run_dir.glob("*.tmp"))
    assert len(replacements) == 2


def test_manifest_failure_is_atomic_and_preserves_error(tmp_path, monkeypatch):
    manifest, run_dir, replacements = _start_manifest(tmp_path, monkeypatch)

    manifest.fail("training exited 9")

    failed = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["error"] == "training exited 9"
    assert failed["timestamps"]["ended_at"].endswith("Z")
    assert len(replacements) == 2


def test_manifest_start_retains_exclusive_ownership_if_manifest_is_removed(
    tmp_path, monkeypatch
):
    _, run_dir, _ = _start_manifest(tmp_path, monkeypatch)
    (run_dir / "run_manifest.json").unlink()

    with pytest.raises(FileExistsError, match="reserved"):
        RunManifest.start(
            run_dir / "run_manifest.json",
            spec=BenchmarkSpec("protgnn", "star", 1234),
            dataset_path=tmp_path / "merged_ed.csv",
            split_path=tmp_path / "canonical_split.json",
            topology_parameters={"patient_hub": True},
            canonical_fold_counts={
                "train": 59607, "validation": 7448, "test": 7456,
            },
            effective_fold_counts={
                "train": 59607, "validation": 7448, "test": 7456,
            },
            class_ordering=[f"class-{index}" for index in range(30)],
            parameter_count=None,
            device={"type": "cpu"},
            package_versions={"python": "3.9.0"},
            git_commit="a" * 40,
            git_dirty=False,
            command=["python3", "-m", "protgnn_analysis.train"],
        )


@pytest.mark.parametrize(
    "escape_kind",
    ["missing", "external", "symlink", "component_symlink", "traversal"],
)
def test_manifest_completion_rejects_missing_or_external_artifacts(
    tmp_path, monkeypatch, escape_kind
):
    manifest, run_dir, _ = _start_manifest(tmp_path, monkeypatch)
    metrics = run_dir / "metrics.json"
    metrics.write_text(
        json.dumps(_valid_test_metrics()), encoding="utf-8"
    )
    external = tmp_path / "external.pt"
    external.write_bytes(b"checkpoint")
    if escape_kind == "missing":
        checkpoint = run_dir / "missing.pt"
        expected_error = FileNotFoundError
        match = "checkpoint"
    elif escape_kind == "external":
        checkpoint = external
        expected_error = ValueError
        match = "contained"
    elif escape_kind == "symlink":
        checkpoint = run_dir / "linked.pt"
        checkpoint.symlink_to(external)
        expected_error = ValueError
        match = "contained"
    elif escape_kind == "component_symlink":
        external_dir = tmp_path / "external-checkpoints"
        external_dir.mkdir()
        (external_dir / "model.pt").write_bytes(b"checkpoint")
        (run_dir / "checkpoints").symlink_to(external_dir, target_is_directory=True)
        checkpoint = run_dir / "checkpoints" / "model.pt"
        expected_error = ValueError
        match = "contained"
    else:
        checkpoint = run_dir / "nested" / ".." / "model.pt"
        (run_dir / "model.pt").write_bytes(b"checkpoint")
        expected_error = ValueError
        match = "escapes"

    with pytest.raises(expected_error, match=match):
        manifest.complete(metrics, checkpoint)

    assert json.loads((run_dir / "run_manifest.json").read_text())["status"] == "failed"


def test_manifest_consumes_metrics_from_the_validated_descriptor_on_swap(
    tmp_path, monkeypatch
):
    manifest, run_dir, _ = _start_manifest(tmp_path, monkeypatch)
    metrics = run_dir / "metrics.json"
    metrics.write_text(json.dumps(_valid_test_metrics(42)), encoding="utf-8")
    checkpoint = run_dir / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    replacement = tmp_path / "replacement.json"
    replacement.write_text(json.dumps(_valid_test_metrics(999)), encoding="utf-8")
    from shared.lib import run_manifest as module

    real_validate = module._artifact_relative_to_run
    swapped = False

    def swap_after_validation(path, run_dir_arg, label, **kwargs):
        nonlocal swapped
        result = real_validate(path, run_dir_arg, label, **kwargs)
        if label == "metrics":
            Path(path).unlink()
            Path(path).symlink_to(replacement)
            swapped = True
        return result

    monkeypatch.setattr(module, "_artifact_relative_to_run", swap_after_validation)
    manifest.complete(metrics, checkpoint)

    completed = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert swapped
    assert completed["status"] == "completed"
    assert completed["parameter_count"] == 42


def test_manifest_transition_rejects_replaced_owner_marker(tmp_path, monkeypatch):
    manifest, run_dir, _ = _start_manifest(tmp_path, monkeypatch)
    owner = run_dir / ".run_manifest.owner"
    owner.unlink()
    owner.write_text("replacement\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="ownership marker changed"):
        manifest.fail("must not consume replacement ownership")

    assert json.loads((run_dir / "run_manifest.json").read_text())["status"] == "running"


@pytest.mark.parametrize(
    "payload,error_match",
    [
        ({}, "six standardized metrics"),
        ({**_valid_test_metrics(), "accuracy": True}, "finite numeric"),
        ({**_valid_test_metrics(), "macro_f1": float("nan")}, "finite numeric"),
        ({**_valid_test_metrics(), "top5_acc": 1.01}, r"\[0, 1\]"),
        ({key: value for key, value in _valid_test_metrics().items()
          if key != "parameter_count"}, "parameter_count"),
        ({**_valid_test_metrics(), "parameter_count": None}, "parameter_count"),
        ({**_valid_test_metrics(), "parameter_count": True}, "parameter_count"),
    ],
)
def test_manifest_completion_fails_closed_on_invalid_metrics(
    tmp_path, monkeypatch, payload, error_match
):
    manifest, run_dir, _ = _start_manifest(tmp_path, monkeypatch)
    metrics = run_dir / "metrics.json"
    checkpoint = run_dir / "model.pt"
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match=error_match):
        manifest.complete(metrics, checkpoint)

    failed = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["timestamps"]["ended_at"].endswith("Z")
    assert "Completion validation failed" in failed["error"]


@pytest.mark.parametrize("metrics_kind", ["missing", "empty", "malformed"])
def test_manifest_completion_fails_closed_on_unreadable_metrics(
    tmp_path, monkeypatch, metrics_kind
):
    manifest, run_dir, _ = _start_manifest(tmp_path, monkeypatch)
    metrics = run_dir / "metrics.json"
    checkpoint = run_dir / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    if metrics_kind == "empty":
        metrics.write_text("", encoding="utf-8")
    elif metrics_kind == "malformed":
        metrics.write_text("{not-json", encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError)):
        manifest.complete(metrics, checkpoint)

    failed = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert "Completion validation failed" in failed["error"]


@pytest.mark.parametrize(
    "method,payload",
    [
        ("protgnn", _nested_metrics_payload()),
        ("gsat", _valid_test_metrics()),
        ("graphcare", _valid_test_metrics()),
    ],
)
def test_manifest_rejects_metrics_layout_for_wrong_method(
    tmp_path, monkeypatch, method, payload
):
    manifest, run_dir, _ = _start_manifest(tmp_path, monkeypatch, method=method)
    metrics = run_dir / "metrics.json"
    checkpoint = run_dir / "model.pt"
    metrics.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="metrics schema"):
        manifest.complete(metrics, checkpoint)

    failed = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
