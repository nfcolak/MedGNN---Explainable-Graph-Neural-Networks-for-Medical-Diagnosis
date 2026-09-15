"""Tiny real preprocessing integration; never uses repository data."""
import json

import pytest


def test_cache_builder_generates_and_reuses_real_fixture(tmp_path, monkeypatch):
    from comparison.standardized import build_caches
    from comparison.standardized.audit_caches import audit_standardized_caches
    from test_protgnn_preprocessing import _write_dataset, _write_split
    data = tmp_path / "data"
    data.mkdir()
    frame = _write_dataset(data / "merged_ed.csv")
    frame.loc[2, "med_alpha"] = 1  # Every canonical record needs a retained concept.
    frame.to_csv(data / "merged_ed.csv", index=False)
    split = _write_split(tmp_path / "canonical_split.json", monkeypatch,
                         {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2})
    args = ["--dataset-dir", str(data), "--canonical-split", str(split)]
    assert build_caches.main(args) == 0
    assert not (data / "graphs").exists()
    assert build_caches.main(args + ["--execute"]) == 0
    files = {p: p.read_bytes() for p in data.rglob("*") if p.is_file()}
    assert build_caches.main(args + ["--execute"]) == 0
    assert all(p.read_bytes() == content for p, content in files.items())
    report = audit_standardized_caches(project_root=tmp_path, split_path=split)
    assert set(report) == {"star", "cooccur"}
    assert all(row["subject_count"] == 6 for row in report.values())


def test_builder_refuses_existing_stale_kg_before_writes(tmp_path, monkeypatch):
    from comparison.standardized import build_caches
    from test_protgnn_preprocessing import _write_dataset, _write_split
    data = tmp_path / "data"
    data.mkdir()
    _write_dataset(data / "merged_ed.csv")
    split = _write_split(tmp_path / "split.json", monkeypatch,
                         {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2})
    kg = data / "graphs/star/graphcare/kg.pt"
    kg.parent.mkdir(parents=True)
    kg.write_bytes(b"existing legacy evidence")
    with pytest.raises(ValueError, match="Preserve"):
        build_caches.main(["--execute", "--dataset-dir", str(data), "--canonical-split", str(split)])
    assert kg.read_bytes() == b"existing legacy evidence"
    assert not (data / "graphs/star/protgnn").exists()
