"""Regression checks for maintained root/module entry points."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", [
    "protgnn_analysis.scripts.confusion_analysis",
    "protgnn_analysis.scripts.explain_checkpoint",
])
def test_help_uses_current_package_imports(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_hpo_import_does_not_create_output_directory(monkeypatch):
    from protgnn_analysis import config
    import protgnn_analysis.scripts.hpo_disease as hpo
    calls = []
    monkeypatch.setattr(hpo.os, "makedirs", lambda *a, **k: calls.append(a))
    importlib.reload(hpo)
    assert calls == []


def test_legacy_multi_seed_keeps_current_multiclass_metrics():
    from protgnn_analysis.scripts.run_multi_seed import aggregate
    result = aggregate({"1": {"macro_f1": 0.25, "top3_acc": 0.5}, "2": {"macro_f1": 0.75, "top3_acc": 1.0}})
    assert result["aggregate"]["macro_f1"]["mean"] == 0.5
    assert result["aggregate"]["top3_acc"]["n"] == 2


def test_legacy_multi_seed_uses_real_trainer_and_latest_metrics(tmp_path, monkeypatch):
    from protgnn_analysis.scripts import run_multi_seed as runner
    results = tmp_path / "results"
    results.mkdir()
    run = results / "new-run"
    run.mkdir()
    metrics = {"macro_f1": 0.5}
    (run / "test_metrics.json").write_text(json.dumps(metrics))
    monkeypatch.setattr(runner, "RESULTS_DIR", results)
    calls = []
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        (results / "latest_run.txt").write_text(str(run))
    monkeypatch.setattr(runner.subprocess, "run", launch)
    assert runner.run_seed(1234, "fixture", ["--explain_n", "0"]) == metrics
    assert calls[0][0][1:3] == ["-m", "protgnn_analysis.train"]
    assert Path(calls[0][1]["cwd"]) == ROOT
