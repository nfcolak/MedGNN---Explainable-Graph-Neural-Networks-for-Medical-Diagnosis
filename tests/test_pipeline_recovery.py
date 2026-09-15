"""Recovery operates on temporary artifacts, never on production runs."""
import json
from pathlib import Path

import pytest

from comparison.standardized import run_all


def _command():
    return next(c for c in run_all.command_plan()["train"] if c[c.index("--methods") + 1] == "gsat")


def _completed(root):
    from test_benchmark_summary import _write_run
    from shared.lib.benchmark_contract import file_sha256
    data, split = root / "data/merged_ed.csv", root / "comparison/canonical_split.json"
    data.parent.mkdir(parents=True)
    split.parent.mkdir(parents=True)
    data.write_text("fixture dataset")
    split.write_text("fixture split")
    path = _write_run(root / "comparison/standardized/results", dataset_hash=file_sha256(data), split_hash=file_sha256(split))
    manifest = json.loads(path.read_text())
    manifest["checkpoint_path"] = "gsat_model.pt"
    (path.parent / "gsat_model.pt").write_bytes(b"fixture checkpoint, not a trained model")
    path.write_text(json.dumps(manifest))
    return path


def test_recover_completed_child_after_parent_interruption(tmp_path):
    path = _completed(tmp_path)
    before = {p: p.read_bytes() for p in path.parent.iterdir()}
    assert run_all.recover_output("train", _command(), root=tmp_path, resume=True)
    assert {p: p.read_bytes() for p in path.parent.iterdir()} == before
    with pytest.raises(FileExistsError):
        run_all.recover_output("train", _command(), root=tmp_path)
    manifest = json.loads(path.read_text())
    manifest["dataset"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        run_all.recover_output("train", _command(), root=tmp_path, resume=True)


def test_incomplete_retry_is_explicit_and_preserves_attempt(tmp_path):
    from test_benchmark_summary import _write_run
    path = _write_run(tmp_path / "comparison/standardized/results", status="failed")
    evidence = path.read_bytes()
    with pytest.raises(FileExistsError, match="archive-incomplete"):
        run_all.recover_output("train", _command(), root=tmp_path, resume=True)
    assert path.read_bytes() == evidence
    assert not run_all.recover_output("train", _command(), root=tmp_path, resume=True, archive_incomplete=True)
    assert not path.parent.exists()
    attempts = list((tmp_path / "comparison/standardized/attempts").rglob("run_manifest.json"))
    assert len(attempts) == 1 and attempts[0].read_bytes() == evidence


def test_orchestrator_recovers_stale_completed_cell_without_child_launch(tmp_path, monkeypatch):
    _completed(tmp_path)
    command = _command()
    plan = {phase: [] for phase in run_all.PHASES}
    plan["train"] = [command]
    monkeypatch.setattr(run_all, "ROOT", tmp_path)
    monkeypatch.setattr(run_all, "command_plan", lambda **kwargs: plan)
    def forbidden(*args, **kwargs):
        pytest.fail("Completed output must not launch training")
    monkeypatch.setattr(run_all.subprocess, "run", forbidden)
    state, events = tmp_path / "state.json", tmp_path / "events.jsonl"
    cp = run_all.Checkpoint(state, events)
    cp.start("train:0")
    assert run_all.main(["--execute", "--resume", "--state", str(state), "--events", str(events)]) == 0
    assert run_all.Checkpoint(state, events).status("train:0") == "completed"
    assert "stale_running_recovered" in events.read_text()


def test_completed_explanations_resume_validates_current_checkpoint_hashes(tmp_path):
    from test_standardized_explanations import _base_record
    from shared.lib.explanation_contract import write_standardized_explanations, load_explanation_cohort
    from comparison.standardized.run_explanations import checkpoint_for, validate_cross_method_outputs
    repo = Path(__file__).resolve().parents[1]
    for relative in ("data/merged_ed.csv", "comparison/canonical_split.json", "comparison/standardized/explanation_subjects.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(repo / relative)  # Read-only production inputs, outputs remain temporary.
    dataset = tmp_path / "data/merged_ed.csv"
    split = tmp_path / "comparison/canonical_split.json"
    cohort = tmp_path / "comparison/standardized/explanation_subjects.json"
    subjects = load_explanation_cohort(cohort, split_path=split, dataset_path=dataset, expected_count=50).subject_ids
    output = tmp_path / "comparison/standardized/explanations/star/seed_1234"
    for method in ("protgnn", "gsat", "graphcare"):
        checkpoint = tmp_path / checkpoint_for(method, "star", 1234)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"synthetic checkpoint provenance only")
        write_standardized_explanations(output_dir=output / method,
            records=[_base_record(method, subject) for subject in subjects],
            method=method, topology="star", seed=1234, cohort_path=cohort,
            split_path=split, dataset_path=dataset, checkpoint_path=checkpoint, expected_count=50)
    validation = validate_cross_method_outputs(output, topology="star", seed=1234,
        cohort_path=cohort, split_path=split, dataset_path=dataset, require_checkpoints=True, cohort_size=50)
    (output / "validation.json").write_text(json.dumps(validation))
    command = run_all.command_plan(cohort_size=50)["explain"][0]
    with pytest.raises(ValueError, match="500 subjects"):
        run_all.recover_output("explain", run_all.command_plan()["explain"][0], root=tmp_path, resume=True)
    assert run_all.recover_output("explain", command, root=tmp_path, resume=True)
    checkpoint.write_bytes(b"changed fixture checkpoint")
    with pytest.raises(ValueError, match="provenance"):
        run_all.recover_output("explain", command, root=tmp_path, resume=True)


def test_incomplete_explanation_set_is_preserved_on_explicit_retry(tmp_path):
    command = run_all.command_plan()["explain"][0]
    output = tmp_path / "comparison/standardized/explanations/star/seed_1234/protgnn"
    output.mkdir(parents=True)
    (output / "partial.json").write_text("fixture evidence")
    with pytest.raises(FileExistsError):
        run_all.recover_output("explain", command, root=tmp_path, resume=True)
    assert not run_all.recover_output("explain", command, root=tmp_path, resume=True, archive_incomplete=True)
    assert len(list((tmp_path / "comparison/standardized/attempts").rglob("partial.json"))) == 1
