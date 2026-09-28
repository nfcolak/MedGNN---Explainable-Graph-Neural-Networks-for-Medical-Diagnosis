"""Real-replay check that non-persisting replay (print-only path) never writes a proof.

Closes the supervisor-found gap in SF2-B: the print-only integration test mocks
``replay_v2_stage``, so removing the ``persist=False`` guard inside the real replay
went undetected. This test drives the REAL replay on the synthetic fixture of
``test_replay_reconstructs_v2_model_and_rejects_tampering``.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

STUDY_TESTS = Path(__file__).parent / "test_cei_v2_study.py"


def _study_tests():
    spec = importlib.util.spec_from_file_location("cei_v2_study_tests_for_replay", STUDY_TESTS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def test_real_replay_without_persist_never_creates_proof(tmp_path, monkeypatch):
    tests = _study_tests()
    captured = {}
    real_module_factory = tests._module

    def capturing_module():
        module = real_module_factory()
        real_replay = module.replay_v2_stage

        def recording(output_dir, binding, result, **kwargs):
            captured.setdefault("call", (module, real_replay, Path(output_dir), binding, result))
            return real_replay(output_dir, binding, result, **kwargs)

        module.replay_v2_stage = recording
        return module

    monkeypatch.setattr(tests, "_module", capturing_module)
    tests.test_replay_reconstructs_v2_model_and_rejects_tampering(tmp_path, monkeypatch)
    module, real_replay, output, binding, result = captured["call"]
    proof_path = output / "replay.json"
    assert proof_path.is_file()
    persisted = proof_path.read_bytes()

    # Existing matching proof: non-persisting replay verifies it and leaves it byte-identical.
    proof = real_replay(output, binding, result, persist=False)
    assert proof["exact_probabilities"] is True
    assert proof_path.read_bytes() == persisted

    # Missing proof: non-persisting replay must refuse and must not create the file.
    proof_path.unlink()
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    with pytest.raises(ValueError, match="existing replay proof is required"):
        real_replay(output, binding, result, persist=False)
    assert not proof_path.exists()
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    assert after == before
