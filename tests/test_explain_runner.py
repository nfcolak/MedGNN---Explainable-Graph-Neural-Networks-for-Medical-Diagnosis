"""End-to-end tests for the item-#1 explanation runner (explain.py).

Unlike every other test file touched this session, this one does NOT stop at
synthetic in-memory tensors: it actually invokes train.run() on tiny synthetic
data, through the real CLI-equivalent pipeline, producing a real binding.json/
preprocessing.json/best.pt/validation.npz on disk -- then points the runner at
that real output directory. This is the only way to prove the reload/replay
logic (which exists specifically to catch a mismatched reconstruction) is
actually correct, not just plausible.
"""
import copy
import json

import pytest
import torch

from core import train
from core.explain import runner as explain
from tests.test_gchm_v2 import artifact, run_args


def train_one(tmp_path, out_name, **overrides):
    root, targets, canonical = artifact(tmp_path, count=24, test_rows=2)
    out = tmp_path / out_name
    args = run_args(root, targets, canonical, out, **overrides)
    train.run(args)
    return out


def test_gchm_v2_runner_replays_and_explains(tmp_path):
    run_dir = train_one(tmp_path, "run_gchm")
    output = tmp_path / "explanations_gchm"
    manifest = explain.run_explanations(
        run_dir, output, max_subjects=2, steps=2, epochs=3
    )
    assert manifest["method"] == "clinical_gnn"
    assert manifest["conv"] == "gchm_v2"
    assert manifest["replay_verified"] is True
    assert manifest["explained_count"] >= 1
    assert manifest["failed_count"] == 0, manifest["failures"]

    for filename in manifest["record_files"]:
        record = json.loads((output / filename).read_text())
        assert record["schema"] == "medgnn.clinical_graph_v2_explanation_record"
        assert record["method"] == "clinical_gnn"
        assert record["conv"] == "gchm_v2"
        for algo in ("GradExplainer", "IntegratedGradExplainer", "GNNExplainer"):
            assert record["graphxai"][algo]["status"] == "success", record["graphxai"][algo]
        # GCHM has no .explain(): builtin readout is correctly absent, not fabricated.
        assert record["builtin_available"] is False
        assert record["builtin_node_importance"] is None


def test_gsat_runner_replays_and_explains(tmp_path):
    run_dir = train_one(tmp_path, "run_gsat", method="gsat", conv=None)
    output = tmp_path / "explanations_gsat"
    manifest = explain.run_explanations(
        run_dir, output, max_subjects=2, steps=2, epochs=3
    )
    assert manifest["method"] == "gsat"
    assert manifest["replay_verified"] is True
    assert manifest["explained_count"] >= 1
    assert manifest["failed_count"] == 0, manifest["failures"]

    for filename in manifest["record_files"]:
        record = json.loads((output / filename).read_text())
        assert record["method"] == "gsat"
        for algo in ("GradExplainer", "IntegratedGradExplainer", "GNNExplainer"):
            assert record["graphxai"][algo]["status"] == "success", record["graphxai"][algo]
        # GSAT has .explain(): its own deterministic attention should be present.
        assert record["builtin_available"] is True
        assert len(record["builtin_node_importance"]) > 0


def test_graphcare_recency_decay_is_recovered_not_defaulted(tmp_path):
    """The one confirmed case where a non-shape hyperparameter is baked
    directly into the forward pass (not just the training loss). If this
    runner silently defaulted recency_decay instead of reading it back from
    method_config.effective_settings, the replay check below would (correctly)
    fail -- this test is the proof it doesn't."""
    run_dir = train_one(tmp_path, "run_graphcare", method="graphcare", conv=None)
    output = tmp_path / "explanations_graphcare"
    manifest = explain.run_explanations(
        run_dir, output, max_subjects=2, steps=2, epochs=3
    )
    assert manifest["method"] == "graphcare"
    assert manifest["replay_verified"] is True
    assert manifest["failed_count"] == 0, manifest["failures"]


def test_replay_check_refuses_a_tampered_prediction_hash(tmp_path):
    """If binding.json's recorded hash doesn't match what reloading + replaying
    best.pt actually produces, the runner must refuse outright, before writing
    any output -- never silently explain a model it can't verify."""
    run_dir = train_one(tmp_path, "run_tamper")
    binding_path = run_dir / "binding.json"
    binding = json.loads(binding_path.read_text())
    tampered = copy.deepcopy(binding)
    tampered["selected_validation"]["prediction_sha256"] = "0" * 64
    binding_path.write_text(json.dumps(tampered))

    output = tmp_path / "explanations_tampered"
    with pytest.raises(ValueError, match="REPLAY CHECK FAILED"):
        explain.run_explanations(run_dir, output, max_subjects=2, steps=2, epochs=3)
    assert not output.exists() or not any(output.iterdir())


def test_refuses_to_write_into_an_occupied_output_directory(tmp_path):
    run_dir = train_one(tmp_path, "run_occupied")
    output = tmp_path / "explanations_occupied"
    output.mkdir()
    (output / "leftover.txt").write_text("from a previous run")
    with pytest.raises(FileExistsError):
        explain.run_explanations(run_dir, output, max_subjects=1, steps=2, epochs=3)


def test_unsupported_conv_is_refused_clearly(tmp_path):
    run_dir = train_one(tmp_path, "run_hgt", conv="hgt", heads=2)
    output = tmp_path / "explanations_hgt"
    with pytest.raises(NotImplementedError, match="not yet supported"):
        explain.run_explanations(run_dir, output, max_subjects=1, steps=2, epochs=3)


def test_replay_check_uses_the_trained_batch_size_not_a_hardcoded_default(tmp_path):
    """Regression for a real bug this session found running the real pipeline
    end to end: replay_check() used to default to a hardcoded batch_size
    (64), independent of what training actually used. On a small validation
    fold that difference was invisible (everything fit in one batch either
    way) -- it only surfaced on a larger real run where batch count actually
    differed between the two sizes, and GSAT's InstanceNorm layers turned out
    to be sensitive to how graphs are grouped into a batch, not just to each
    graph's own content. Force a tiny training batch_size (3) against the
    existing 4-graph validation fold, which already spans multiple batches
    at size 3 but not at the old hardcoded 64 -- proving the fix reads
    binding['batch_size'] rather than reintroducing the mismatch.
    """
    run_dir = train_one(tmp_path, "run_batchsize", method="gsat", conv=None, batch_size=3)
    binding = json.loads((run_dir / "binding.json").read_text())
    assert binding["batch_size"] == 3
    output = tmp_path / "explanations_batchsize"
    manifest = explain.run_explanations(run_dir, output, max_subjects=4, steps=2, epochs=3)
    assert manifest["replay_verified"] is True
    assert manifest["failed_count"] == 0, manifest["failures"]
