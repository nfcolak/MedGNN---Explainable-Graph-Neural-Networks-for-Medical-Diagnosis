import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from comparison.standardized import run_benchmark as runner
from comparison.standardized.run_benchmark import (
    RunnerOptions,
    build_run_specs,
    command_for,
    load_config,
)
from shared.lib.benchmark_contract import (
    ALLOWED_SEEDS,
    EXPECTED_CLASSES,
    EXPECTED_FOLD_COUNTS,
    BenchmarkSpec,
)
from shared.lib.run_manifest import RunManifest


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "comparison" / "standardized" / "benchmark_config.json"


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


def _global_state_snapshot():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].copy(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch": torch.random.get_rng_state().clone(),
        "environment": dict(os.environ),
        "sys_path": list(sys.path),
    }


def _assert_global_state(snapshot):
    assert random.getstate() == snapshot["python"]
    current_numpy = np.random.get_state()
    assert current_numpy[0] == snapshot["numpy"][0]
    assert np.array_equal(current_numpy[1], snapshot["numpy"][1])
    assert current_numpy[2:] == snapshot["numpy"][2:]
    assert torch.equal(torch.random.get_rng_state(), snapshot["torch"])
    assert dict(os.environ) == snapshot["environment"]
    assert sys.path == snapshot["sys_path"]


def _mutate_process_globals():
    random.random()
    np.random.random()
    torch.rand(1)
    os.environ["TASK6_CALLABLE_LEAK_PROBE"] = "mutated"
    sys.path.insert(0, "/tmp/task6-callable-leak-probe")


def _flag_value(command, flag):
    index = command.index(flag)
    return command[index + 1]


def test_runner_script_is_directly_executable_without_pythonpath():
    result = subprocess.run(
        ["python3", "comparison/standardized/run_benchmark.py", "--dry-run"],
        cwd=REPO,
        shell=False,
        check=True,
        capture_output=True,
        text=True,
    )

    assert len(json.loads(result.stdout)) == 18


def test_default_matrix_rejects_duplicate_cells():
    config = load_config(CONFIG_PATH)
    config["seeds"] = [1234, 1234, 1235, 1236]

    with pytest.raises(ValueError, match="duplicates"):
        build_run_specs(config)


def test_default_config_builds_exact_unique_primary_matrix():
    config = load_config(CONFIG_PATH)

    specs = build_run_specs(config)

    assert len(specs) == 18
    assert len(set(specs)) == 18
    assert {spec.method for spec in specs} == {"protgnn", "gsat", "graphcare"}
    assert {spec.structure for spec in specs} == {"star", "cooccur"}
    assert {spec.seed for spec in specs} == set(ALLOWED_SEEDS)


@pytest.mark.parametrize("method", ["protgnn", "gsat", "graphcare"])
def test_commands_use_real_explicit_cli_flags_and_isolated_output(method):
    config = load_config(CONFIG_PATH)
    spec = BenchmarkSpec(method, "star", 1234)

    command = command_for(spec, RunnerOptions(config=config))

    assert isinstance(command, list)
    assert all(isinstance(part, str) for part in command)
    flags = config["methods"][method]["flags"]
    assert _flag_value(command, flags["structure"]) == "star"
    assert _flag_value(command, flags["split"]) == "comparison/canonical_split.json"
    assert _flag_value(command, flags["seed"]) == "1234"
    assert _flag_value(command, flags["output_dir"]) == (
        "comparison/standardized/results/" + method + "/star/seed_1234"
    )
    assert not any("outputs/" in part for part in command)


def test_commands_forward_optional_smoke_controls_without_shell_interpolation():
    config = load_config(CONFIG_PATH)
    options = RunnerOptions(config=config, limit=7, max_epochs=2)

    for spec in build_run_specs(config):
        command = command_for(spec, options)
        flags = config["methods"][spec.method]["flags"]
        assert _flag_value(command, flags["limit"]) == "7"
        assert _flag_value(command, flags["max_epochs"]) == "2"
        assert not any(part in {"sh", "bash", "-c"} for part in command)


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


def _write_valid_project(tmp_path):
    data_dir = tmp_path / "data"
    comparison_dir = tmp_path / "comparison"
    data_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)
    folds = {}
    rows = ["subject_id,disease_1"]
    subject_number = 0
    for fold_id, count in EXPECTED_FOLD_COUNTS.items():
        for _ in range(count):
            subject = str(subject_number)
            folds[subject] = fold_id
            rows.append(f"{subject},{EXPECTED_CLASSES[subject_number % len(EXPECTED_CLASSES)]}")
            subject_number += 1
    (data_dir / "merged_ed.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (comparison_dir / "canonical_split.json").write_text(
        json.dumps({"fold": folds, "classes": list(EXPECTED_CLASSES), "n_kept": len(folds)}),
        encoding="utf-8",
    )
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config_path = tmp_path / "benchmark_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_dry_run_is_deterministic_json_with_no_writes_or_subprocesses(
    tmp_path, monkeypatch, capsys
):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    def forbidden_subprocess(*args, **kwargs):
        raise AssertionError("dry-run launched a subprocess")

    def forbidden_cache_audit(*args, **kwargs):
        raise AssertionError("dry-run required production caches")

    monkeypatch.setattr(runner.subprocess, "run", forbidden_subprocess)
    monkeypatch.setattr(runner, "audit_standardized_caches", forbidden_cache_audit)
    assert runner.main(["--config", str(config_path), "--dry-run"]) == 0
    first = capsys.readouterr().out
    assert runner.main(["--config", str(config_path), "--dry-run"]) == 0
    second = capsys.readouterr().out

    assert first == second
    payload = json.loads(first)
    assert len(payload) == 18
    assert not (tmp_path / "comparison" / "standardized" / "results").exists()


def test_full_run_requires_cache_parity_before_outputs_or_subprocess(
    tmp_path, monkeypatch
):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    launched = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: launched.append(a))

    def reject_cache_parity(**kwargs):
        raise ValueError("production cache fingerprint mismatch")

    monkeypatch.setattr(runner, "audit_standardized_caches", reject_cache_parity)

    with pytest.raises(ValueError, match="cache fingerprint mismatch"):
        runner.main([
            "--config", str(config_path), "--methods", "gsat",
            "--structures", "star", "--seeds", "1234",
        ])

    assert launched == []
    assert not (tmp_path / "comparison" / "standardized" / "results").exists()


def test_filters_select_only_requested_matrix_cells(tmp_path, monkeypatch, capsys):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)

    assert runner.main([
        "--config", str(config_path), "--methods", "gsat",
        "--structures", "cooccur", "--seeds", "1235", "--dry-run",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [(item["method"], item["structure"], item["seed"]) for item in payload] == [
        ("gsat", "cooccur", 1235)
    ]


def test_runtime_metadata_uses_the_method_interpreter(monkeypatch, tmp_path):
    config = load_config(CONFIG_PATH)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({
                "device": {"type": "cpu", "python_executable": command[0]},
                "package_versions": {"python": "3.9.0", "torch": "1.0"},
            }),
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    device, versions = runner.collect_runtime_metadata(
        BenchmarkSpec("graphcare", "star", 1234),
        config["methods"]["graphcare"],
        tmp_path,
    )

    assert calls[0][0][0] == ".venv-graphcare/bin/python3"
    assert calls[0][1]["shell"] is False
    assert device["type"] == "cpu"
    assert versions["torch"] == "1.0"


def test_live_run_writes_and_completes_manifest_around_subprocess(
    tmp_path, monkeypatch
):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "collect_git_state", lambda root: ("b" * 40, False))
    monkeypatch.setattr(
        runner,
        "collect_runtime_metadata",
        lambda spec, config, root: ({"type": "cpu"}, {"python": "3.9.0"}),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        assert isinstance(command, list)
        assert kwargs["cwd"] == tmp_path
        assert kwargs["shell"] is False
        out_flag = "--out_dir"
        run_dir = tmp_path / command[command.index(out_flag) + 1]
        run_dir.joinpath("metrics.json").write_text(
            json.dumps(_nested_metrics_payload()), encoding="utf-8"
        )
        run_dir.joinpath("gsat_model.pt").write_bytes(b"checkpoint")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.main([
        "--config", str(config_path), "--methods", "gsat",
        "--structures", "star", "--seeds", "1234",
        "--limit", "4", "--max-epochs", "2",
    ])

    assert result == 0
    assert len(calls) == 1
    run_dir = tmp_path / "comparison/standardized/results/gsat/star/seed_1234"
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["metrics_path"] == "metrics.json"
    assert manifest["checkpoint_path"] == "gsat_model.pt"
    assert manifest["class_ordering"] == list(EXPECTED_CLASSES)
    assert manifest["canonical_fold_counts"] == {
        "train": 59607, "validation": 7448, "test": 7456
    }
    assert manifest["effective_fold_counts"] == {
        "train": 4, "validation": 4, "test": 4
    }
    assert "fold_counts" not in manifest


def test_effective_fold_counts_cap_each_fold_without_changing_canonical_counts():
    canonical = {"train": 59607, "validation": 7448, "test": 7456}

    assert runner._effective_fold_counts(canonical, None) == canonical
    assert runner._effective_fold_counts(canonical, 10_000) == {
        "train": 10_000,
        "validation": 7448,
        "test": 7456,
    }
    assert canonical == {"train": 59607, "validation": 7448, "test": 7456}


@pytest.mark.parametrize("populate", [False, True])
def test_any_existing_output_cell_fails_closed_before_subprocess(
    tmp_path, monkeypatch, populate
):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    run_dir = tmp_path / "comparison/standardized/results/gsat/star/seed_1234"
    run_dir.mkdir(parents=True)
    if populate:
        (run_dir / "prior.txt").write_text("do not overwrite", encoding="utf-8")
    launched = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: launched.append(a))

    with pytest.raises(FileExistsError, match="existing"):
        runner.main([
            "--config", str(config_path), "--methods", "gsat",
            "--structures", "star", "--seeds", "1234",
        ])

    assert not launched
    if populate:
        assert (run_dir / "prior.txt").read_text(encoding="utf-8") == "do not overwrite"


def test_run_output_symlink_escape_fails_closed_before_subprocess(tmp_path, monkeypatch):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    results = tmp_path / "comparison/standardized/results"
    results.mkdir(parents=True)
    (results / "gsat").symlink_to(outside, target_is_directory=True)
    launched = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: launched.append(a))

    with pytest.raises(ValueError, match="symlink"):
        runner.main([
            "--config", str(config_path), "--methods", "gsat",
            "--structures", "star", "--seeds", "1234",
        ])

    assert not launched
    assert list(outside.iterdir()) == []


def test_reservation_is_not_redirected_by_parent_symlink_swap(tmp_path, monkeypatch):
    config = load_config(CONFIG_PATH)
    spec = BenchmarkSpec("gsat", "star", 1234)
    results = tmp_path / "comparison/standardized/results"
    method_dir = results / "gsat"
    parent = method_dir / "star"
    parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    (outside / "star").mkdir(parents=True)
    held_method_dir = results / "gsat-held"
    real_mkdir = os.mkdir
    attacked = False

    def swapping_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal attacked
        if not attacked and os.fspath(path).endswith("seed_1234"):
            attacked = True
            method_dir.rename(held_method_dir)
            method_dir.symlink_to(outside, target_is_directory=True)
        if dir_fd is None:
            return real_mkdir(path, mode)
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", swapping_mkdir)

    with pytest.raises(ValueError, match="changed|symlink"):
        runner._reserve_run_outputs([spec], config, tmp_path)

    assert attacked
    assert not (outside / "star" / "seed_1234").exists()


def test_concurrent_exact_cell_reservation_has_one_owner(tmp_path):
    config = load_config(CONFIG_PATH)
    spec = BenchmarkSpec("gsat", "star", 1234)
    parent = tmp_path / "comparison/standardized/results/gsat/star"
    parent.mkdir(parents=True)
    barrier = threading.Barrier(2)

    def reserve():
        barrier.wait()
        try:
            reservation = runner._reserve_run_outputs([spec], config, tmp_path)
        except FileExistsError:
            return "occupied"
        reservation.close()
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(lambda _: reserve(), range(2)))

    assert outcomes == ["occupied", "reserved"]
    run_dir = parent / "seed_1234"
    assert run_dir.is_dir()
    assert not (run_dir / "run_manifest.json").exists()


def _stub_runner_metadata(monkeypatch):
    monkeypatch.setattr(runner, "audit_standardized_caches", lambda **kwargs: {})
    monkeypatch.setattr(runner, "collect_git_state", lambda root: ("e" * 40, False))
    monkeypatch.setattr(
        runner,
        "collect_runtime_metadata",
        lambda spec, config, root: ({"type": "cpu"}, {"python": "3.9.0"}),
    )


def test_manifest_start_failure_releases_only_current_unstarted_cell(
    tmp_path, monkeypatch
):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _stub_runner_metadata(monkeypatch)
    from shared.lib import run_manifest as manifest_module

    monkeypatch.setattr(
        manifest_module,
        "_atomic_json_write",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("start probe")),
    )

    with pytest.raises(RuntimeError, match="start probe"):
        runner.main([
            "--config", str(config_path), "--methods", "gsat",
            "--structures", "star", "--seeds", "1234", "1235",
        ])

    results = tmp_path / "comparison/standardized/results/gsat/star"
    assert not (results / "seed_1234").exists()
    assert not (results / "seed_1235").exists()


def test_fail_fast_leaves_later_cells_nonexistent(tmp_path, monkeypatch):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _stub_runner_metadata(monkeypatch)
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 9),
    )

    assert runner.main([
        "--config", str(config_path), "--methods", "gsat",
        "--structures", "star", "--seeds", "1234", "1235",
    ]) == 1

    first = tmp_path / "comparison/standardized/results/gsat/star/seed_1234"
    assert json.loads((first / "run_manifest.json").read_text())["status"] == "failed"
    assert (first / ".run_manifest.owner").is_file()
    assert not (first.parent / "seed_1235").exists()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt(), SystemExit(7)])
def test_baseexception_marks_active_manifest_failed_before_reraising(
    tmp_path, monkeypatch, interruption
):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    _stub_runner_metadata(monkeypatch)

    def interrupt(command, **kwargs):
        raise interruption

    monkeypatch.setattr(runner.subprocess, "run", interrupt)
    with pytest.raises(type(interruption)):
        runner.main([
            "--config", str(config_path), "--methods", "gsat",
            "--structures", "star", "--seeds", "1234", "1235",
        ])

    first = tmp_path / "comparison/standardized/results/gsat/star/seed_1234"
    manifest = json.loads((first / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert type(interruption).__name__ in manifest["error"]
    assert (first / ".run_manifest.owner").is_file()
    assert not (first.parent / "seed_1235").exists()


def test_unstarted_cleanup_never_removes_a_replacement_inode(tmp_path):
    config = load_config(CONFIG_PATH)
    spec = BenchmarkSpec("gsat", "star", 1234)
    reservation = runner._reserve_run_output(spec, config, tmp_path)
    held = reservation.run_dir.with_name("seed_1234-held")
    reservation.run_dir.rename(held)
    reservation.run_dir.mkdir()

    reservation.release_unstarted()
    reservation.close()

    assert reservation.run_dir.is_dir()
    assert held.is_dir()


def test_failed_subprocess_marks_manifest_and_continue_on_error_runs_next(
    tmp_path, monkeypatch
):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "collect_git_state", lambda root: ("c" * 40, True))
    monkeypatch.setattr(runner, "audit_standardized_caches", lambda **kwargs: {})
    monkeypatch.setattr(
        runner,
        "collect_runtime_metadata",
        lambda spec, config, root: ({"type": "cpu"}, {"python": "3.9.0"}),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        seed = _flag_value(command, "--seed")
        run_dir = tmp_path / _flag_value(command, "--out_dir")
        if seed == "1234":
            return subprocess.CompletedProcess(command, 9)
        run_dir.joinpath("metrics.json").write_text(
            json.dumps(_nested_metrics_payload()), encoding="utf-8"
        )
        run_dir.joinpath("gsat_model.pt").write_bytes(b"ok")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.main([
        "--config", str(config_path), "--methods", "gsat", "--structures", "star",
        "--seeds", "1234", "1235", "--continue-on-error",
    ])

    assert result == 1
    assert len(calls) == 2
    failed = json.loads((tmp_path / (
        "comparison/standardized/results/gsat/star/seed_1234/run_manifest.json"
    )).read_text())
    completed = json.loads((tmp_path / (
        "comparison/standardized/results/gsat/star/seed_1235/run_manifest.json"
    )).read_text())
    assert failed["status"] == "failed"
    assert "exit code 9" in failed["error"]
    assert completed["status"] == "completed"


def test_successful_process_with_invalid_metrics_is_recorded_failed(tmp_path, monkeypatch):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "collect_git_state", lambda root: ("d" * 40, False))
    monkeypatch.setattr(runner, "audit_standardized_caches", lambda **kwargs: {})
    monkeypatch.setattr(
        runner,
        "collect_runtime_metadata",
        lambda spec, config, root: ({"type": "cpu"}, {"python": "3.9.0"}),
    )

    def fake_run(command, **kwargs):
        run_dir = tmp_path / _flag_value(command, "--out_dir")
        run_dir.joinpath("metrics.json").write_text(
            json.dumps({"parameter_count": 42, "test": {}}), encoding="utf-8"
        )
        run_dir.joinpath("gsat_model.pt").write_bytes(b"checkpoint")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main([
        "--config", str(config_path), "--methods", "gsat",
        "--structures", "star", "--seeds", "1234",
    ]) == 1
    manifest_path = tmp_path / (
        "comparison/standardized/results/gsat/star/seed_1234/run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert "six standardized metrics" in manifest["error"]


def test_nested_method_metrics_expose_parameter_count_at_documented_top_level(tmp_path):
    from graphcare_analysis import run as graphcare_run
    from gsat_analysis import train as gsat_train

    metrics = _valid_test_metrics(parameter_count=77)
    graphcare_dir = tmp_path / "graphcare"
    gsat_dir = tmp_path / "gsat"

    graphcare_run._write_report(
        metrics,
        30,
        {"num_nodes": 1, "num_rels": 1},
        1,
        1,
        out_dir=graphcare_dir,
    )
    gsat_train._write_report(metrics, "star", 1, 1, 1, 1, gsat_dir)

    for output_dir in (graphcare_dir, gsat_dir):
        payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
        assert payload["parameter_count"] == 77
        assert payload["test"] == metrics


def test_invalid_dataset_and_legacy_output_root_fail_closed(tmp_path, monkeypatch):
    config_path = _write_valid_project(tmp_path)
    monkeypatch.setattr(runner, "PROJECT_ROOT", tmp_path)
    dataset = tmp_path / "data" / "merged_ed.csv"
    dataset.write_text("wrong,columns\n1,A\n", encoding="utf-8")

    with pytest.raises(ValueError, match="subject_id.*disease_1"):
        runner.main(["--config", str(config_path), "--dry-run"])

    _write_valid_project(tmp_path)
    config = json.loads(config_path.read_text())
    config["results_root"] = "outputs/results"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="results_root"):
        runner.main(["--config", str(config_path), "--dry-run"])


def test_graphcare_standardized_main_validates_seed_and_restores_scoped_config(
    tmp_path, monkeypatch
):
    from graphcare_analysis import run as graphcare_run

    config_path = _write_valid_project(tmp_path)
    del config_path
    split_path = tmp_path / "comparison/canonical_split.json"
    out_dir = tmp_path / "comparison/standardized/results/graphcare/star/seed_1234"
    prior_seed = graphcare_run.cfg.seed
    prior_structure = graphcare_run.cfg.graph_structure
    captured = {}

    def fake_training(**kwargs):
        captured.update(kwargs)
        assert graphcare_run.cfg.seed == 1234
        assert graphcare_run.cfg.graph_structure == "star"
        return {"macro_f1": 0.5}

    monkeypatch.setattr(graphcare_run, "_run_training", fake_training)
    result = graphcare_run.main(
        limit=3,
        max_epochs=2,
        split_json=split_path,
        out_dir=out_dir,
        graph_structure="star",
        seed=1234,
    )

    assert result == {"macro_f1": 0.5}
    assert captured["seed"] == 1234
    assert captured["out_dir"] == out_dir
    assert graphcare_run.cfg.seed == prior_seed
    assert graphcare_run.cfg.graph_structure == prior_structure


def test_graphcare_standardized_main_requires_all_explicit_contract_fields(tmp_path):
    from graphcare_analysis import run as graphcare_run

    with pytest.raises(ValueError, match="requires all"):
        graphcare_run.main(
            split_json=tmp_path / "split.json",
            out_dir=tmp_path / "output",
            graph_structure="star",
            seed=None,
        )
    with pytest.raises(ValueError, match="standardized results"):
        graphcare_run.main(
            split_json=tmp_path / "split.json",
            out_dir=tmp_path / "legacy-output",
            graph_structure="star",
            seed=1234,
        )


@pytest.mark.parametrize("raises", [False, True, "baseexception"])
def test_graphcare_callable_restores_all_global_state(tmp_path, monkeypatch, raises):
    from graphcare_analysis import run as graphcare_run

    _write_valid_project(tmp_path)
    split_path = tmp_path / "comparison/canonical_split.json"
    out_dir = tmp_path / "comparison/standardized/results/graphcare/star/seed_1234"
    prior_dropout = graphcare_run.cfg.dropout
    snapshot = _global_state_snapshot()

    def fake_training(**kwargs):
        _mutate_process_globals()
        graphcare_run.cfg.dropout = 0.99
        if raises == "baseexception":
            raise KeyboardInterrupt("graphcare probe")
        if raises:
            raise RuntimeError("graphcare probe")
        return _valid_test_metrics()

    monkeypatch.setattr(graphcare_run, "_run_training", fake_training)
    if raises:
        expected = KeyboardInterrupt if raises == "baseexception" else RuntimeError
        with pytest.raises(expected, match="graphcare probe"):
            graphcare_run.main(
                split_json=split_path, out_dir=out_dir,
                graph_structure="star", seed=1234,
            )
    else:
        graphcare_run.main(
            split_json=split_path, out_dir=out_dir,
            graph_structure="star", seed=1234,
        )

    assert graphcare_run.cfg.dropout == prior_dropout
    _assert_global_state(snapshot)


@pytest.mark.parametrize("raises", [False, True, "baseexception"])
def test_gsat_callable_restores_all_global_state(tmp_path, monkeypatch, raises):
    from gsat_analysis import train as gsat_train

    _write_valid_project(tmp_path)
    split_path = tmp_path / "comparison/canonical_split.json"
    out_dir = tmp_path / "comparison/standardized/results/gsat/star/seed_1234"
    prior_dropout = gsat_train.cfg.dropout
    snapshot = _global_state_snapshot()

    def fake_training(*args, **kwargs):
        _mutate_process_globals()
        gsat_train.cfg.dropout = 0.99
        if raises == "baseexception":
            raise KeyboardInterrupt("gsat probe")
        if raises:
            raise RuntimeError("gsat probe")
        return _valid_test_metrics()

    monkeypatch.setattr(gsat_train, "_run_training", fake_training)
    if raises:
        expected = KeyboardInterrupt if raises == "baseexception" else RuntimeError
        with pytest.raises(expected, match="gsat probe"):
            gsat_train.main("star", 1, 1, out_dir, split_path, 1234)
    else:
        gsat_train.main("star", 1, 1, out_dir, split_path, 1234)

    assert gsat_train.cfg.dropout == prior_dropout
    _assert_global_state(snapshot)


@pytest.mark.parametrize("raises", [False, True, "baseexception"])
def test_protgnn_callable_restores_all_global_state(tmp_path, monkeypatch, raises):
    from protgnn_analysis import train as protgnn_train

    _write_valid_project(tmp_path)
    split_path = tmp_path / "comparison/canonical_split.json"
    out_dir = tmp_path / "comparison/standardized/results/protgnn/star/seed_1234"
    prior_dropout = protgnn_train.model_args.dropout
    snapshot = _global_state_snapshot()

    def fake_train(*args, **kwargs):
        _mutate_process_globals()
        protgnn_train.model_args.dropout = 0.99
        if raises == "baseexception":
            raise KeyboardInterrupt("protgnn probe")
        if raises:
            raise RuntimeError("protgnn probe")
        return torch.nn.Linear(1, 1), [], _valid_test_metrics(), 30, [], None, {}

    monkeypatch.setattr(protgnn_train, "train_model", fake_train)
    monkeypatch.setattr(protgnn_train, "explain_test_set", lambda *a, **k: [])
    monkeypatch.setattr(protgnn_train, "save_results", lambda *a, **k: None)
    kwargs = {
        "graph_structure": "star",
        "canonical_split": split_path,
        "seed": 1234,
        "output_dir": out_dir,
        "max_epochs": 1,
        "explain_n": 0,
    }
    if raises:
        expected = KeyboardInterrupt if raises == "baseexception" else RuntimeError
        with pytest.raises(expected, match="protgnn probe"):
            protgnn_train.main(**kwargs)
    else:
        protgnn_train.main(**kwargs)

    assert protgnn_train.model_args.dropout == prior_dropout
    _assert_global_state(snapshot)


def test_protgnn_explanation_restores_model_mode_and_devices_on_exception(monkeypatch):
    from protgnn_analysis import train as protgnn_train

    class FakeGNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.device = "original-gnn-device"
            self.model = SimpleNamespace(device="original-model-device")

        def to(self, device):
            self.device = device
            self.model.device = device
            return self

    gnn = FakeGNN()
    gnn.train(True)
    monkeypatch.setattr(protgnn_train, "GRAPHXAI_OK", True)
    monkeypatch.setattr(
        protgnn_train, "get_dataset", lambda *a, **k: SimpleNamespace(feature_cols=[])
    )
    monkeypatch.setattr(
        protgnn_train, "get_dataloader", lambda *a, **k: {"test": []}
    )
    monkeypatch.setattr(
        protgnn_train,
        "GradExplainer",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("explainer probe")),
    )

    with pytest.raises(RuntimeError, match="explainer probe"):
        protgnn_train.explain_test_set(gnn, 30, 1)

    assert gnn.training is True
    assert gnn.device == "original-gnn-device"
    assert gnn.model.device == "original-model-device"


def test_method_clis_accept_every_runner_flag(monkeypatch):
    from graphcare_analysis import run as graphcare_run
    from gsat_analysis import train as gsat_train
    from protgnn_analysis import train as protgnn_train

    graphcare_call = {}
    gsat_call = {}
    protgnn_call = {}
    monkeypatch.setattr(
        graphcare_run, "main", lambda **kwargs: graphcare_call.update(kwargs)
    )
    monkeypatch.setattr(
        gsat_train, "main", lambda *args, **kwargs: gsat_call.update({
            "graph_structure": args[0], "max_epochs": args[1], "limit": args[2],
            "out_dir": args[3], "canonical_split": args[4], "seed": args[5],
            **kwargs,  # e.g. loss_weighting, added by class-weighting-and-hpo
        })
    )
    monkeypatch.setattr(
        protgnn_train, "main", lambda **kwargs: protgnn_call.update(kwargs)
    )

    graphcare_run.cli([
        "--graph_structure", "cooccur", "--canonical_split", "comparison/canonical_split.json",
        "--seed", "1235", "--out_dir",
        "comparison/standardized/results/graphcare/cooccur/seed_1235",
        "--limit", "8", "--max_epochs", "3",
    ])
    gsat_train.cli([
        "--graph_structure", "cooccur", "--canonical_split", "comparison/canonical_split.json",
        "--seed", "1235", "--out_dir",
        "comparison/standardized/results/gsat/cooccur/seed_1235",
        "--limit", "8", "--max_epochs", "3",
    ])
    protgnn_train.cli([
        "--graph-structure", "cooccur", "--canonical-split", "comparison/canonical_split.json",
        "--seed", "1235", "--output-dir",
        "comparison/standardized/results/protgnn/cooccur/seed_1235",
        "--limit", "8", "--max-epochs", "3",
    ])

    assert graphcare_call["seed"] == 1235
    assert graphcare_call["limit"] == 8
    assert graphcare_call["max_epochs"] == 3
    assert gsat_call["seed"] == 1235
    assert gsat_call["limit"] == 8
    assert gsat_call["max_epochs"] == 3
    assert protgnn_call["seed"] == 1235
    assert protgnn_call["limit"] == 8
    assert protgnn_call["max_epochs"] == 3


@pytest.mark.parametrize("method", ["protgnn", "gsat", "graphcare"])
def test_generated_command_parser_accepts_all_flags_in_configured_subprocess(method):
    config = load_config(CONFIG_PATH)
    command = command_for(
        BenchmarkSpec(method, "cooccur", 1235),
        RunnerOptions(config=config, limit=8, max_epochs=3),
    )

    result = subprocess.run(
        [*command, "--help"],
        cwd=REPO,
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    for flag in config["methods"][method]["flags"].values():
        assert flag in result.stdout


def test_protgnn_smoke_controls_are_scoped_and_forward_limit(tmp_path, monkeypatch):
    from protgnn_analysis import train as protgnn_train

    prior_epochs = protgnn_train.train_args.max_epochs
    captured = {}
    monkeypatch.setattr(protgnn_train, "load_canonical_split", lambda path: {})

    def fake_train(*args, **kwargs):
        captured.update(kwargs)
        assert protgnn_train.train_args.max_epochs == 2
        return torch.nn.Linear(1, 1), [], {}, 30, [], None, {}

    monkeypatch.setattr(protgnn_train, "train_model", fake_train)
    monkeypatch.setattr(protgnn_train, "explain_test_set", lambda *a, **k: [])
    monkeypatch.setattr(protgnn_train, "save_results", lambda *a, **k: None)

    protgnn_train.main(
        graph_structure="star",
        canonical_split=tmp_path / "comparison/canonical_split.json",
        seed=1234,
        output_dir=tmp_path / (
            "comparison/standardized/results/protgnn/star/seed_1234"
        ),
        limit=5,
        max_epochs=2,
        explain_n=0,
    )

    assert captured["limit"] == 5
    assert protgnn_train.train_args.max_epochs == prior_epochs


def test_protgnn_callable_fails_if_model_cannot_report_parameter_count(
    tmp_path, monkeypatch
):
    from protgnn_analysis import train as protgnn_train

    _write_valid_project(tmp_path)
    split_path = tmp_path / "comparison/canonical_split.json"
    out_dir = tmp_path / "comparison/standardized/results/protgnn/star/seed_1234"
    monkeypatch.setattr(
        protgnn_train,
        "train_model",
        lambda *a, **k: (object(), [], _valid_test_metrics(), 30, [], None, {}),
    )
    monkeypatch.setattr(protgnn_train, "explain_test_set", lambda *a, **k: [])
    monkeypatch.setattr(protgnn_train, "save_results", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="parameter_count"):
        protgnn_train.main(
            graph_structure="star",
            canonical_split=split_path,
            seed=1234,
            output_dir=out_dir,
            max_epochs=1,
            explain_n=0,
        )
