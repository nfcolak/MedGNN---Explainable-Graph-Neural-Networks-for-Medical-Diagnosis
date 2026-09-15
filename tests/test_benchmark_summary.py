import csv
import json
from pathlib import Path

import pytest

from shared.lib.benchmark_contract import EXPECTED_CLASSES
from shared.lib.run_manifest import STANDARDIZED_METRICS, topology_policy_fingerprint


FOLD_COUNTS = {"train": 59607, "validation": 7448, "test": 7456}
TOPOLOGY_PARAMETERS = {
    "star": {
        "patient_hub": True,
        "patient_concept_edges": "bidirectional",
        "concept_concept_edges": "none",
    },
    "cooccur": {
        "patient_hub": True,
        "patient_concept_edges": "bidirectional",
        "concept_concept_edges": "training_fold_pmi",
        "pmi_threshold": 2.0,
    },
    "ontology": {
        "patient_hub": True,
        "patient_concept_edges": "bidirectional",
        "concept_concept_edges": "ontology",
    },
    "full": {
        "patient_hub": True,
        "patient_concept_edges": "bidirectional",
        "concept_concept_edges": "training_fold_pmi_and_ontology",
    },
    "full_kg_expanded": {
        "patient_hub": True,
        "patient_concept_edges": "bidirectional",
        "concept_concept_edges": "training_fold_pmi_and_ontology",
        "record_external_nodes": "one_hop_kg_neighbours",
    },
}


def _metrics(seed, parameter_count=42):
    offset = (seed - 1234) / 10.0
    return {
        name: (index + 1) / 10.0 + offset
        for index, name in enumerate(STANDARDIZED_METRICS)
    } | {"parameter_count": parameter_count}


def _write_run(
    results_dir: Path,
    method="gsat",
    topology="star",
    seed=1234,
    *,
    status="completed",
    dataset_hash="a" * 64,
    split_hash="b" * 64,
    parameter_count=42,
    topology_parameters=None,
):
    run_dir = results_dir / method / topology / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    parameters = topology_parameters or TOPOLOGY_PARAMETERS[topology]
    metrics_path = run_dir / "metrics.json"
    metrics = _metrics(seed, parameter_count)
    payload = metrics if method == "protgnn" else {
        "parameter_count": parameter_count,
        "test": metrics,
    }
    if status == "completed":
        metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "method": method,
        "topology": topology,
        "seed": seed,
        "dataset": {"path": "data/merged_ed.csv", "sha256": dataset_hash},
        "split": {"path": "comparison/canonical_split.json", "sha256": split_hash},
        "topology_parameters": parameters,
        "topology_policy_fingerprint": topology_policy_fingerprint(parameters),
        "canonical_fold_counts": FOLD_COUNTS,
        "effective_fold_counts": FOLD_COUNTS,
        "class_ordering": list(EXPECTED_CLASSES),
        "parameter_count": parameter_count if status == "completed" else None,
        "status": status,
        "metrics_path": "metrics.json" if status == "completed" else None,
        "timestamps": {
            "started_at": "2026-09-12T10:00:00.000000Z",
            "ended_at": "2026-09-12T11:00:00.000000Z" if status != "running" else None,
        },
        "error": "training failed" if status == "failed" else None,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return run_dir / "run_manifest.json"


def test_loads_completed_manifests_and_common_secondary_runs_in_stable_order(tmp_path):
    from comparison.standardized.summarize import load_compatible_runs

    results = tmp_path / "results"
    _write_run(results, "graphcare", "ontology", 1234)
    _write_run(results, "gsat", "star", 1235)
    _write_run(results, "protgnn", "star", 1234)

    runs = load_compatible_runs(results)

    assert [(run.method, run.topology, run.seed) for run in runs] == [
        ("protgnn", "star", 1234),
        ("gsat", "star", 1235),
        ("graphcare", "ontology", 1234),
    ]
    assert [(run.method, run.topology, run.seed) for run in runs.common_secondary] == [
        ("graphcare", "ontology", 1234)
    ]
    assert runs.method_specific == ()


def test_rejects_incompatible_split_hash_instead_of_averaging(tmp_path):
    from comparison.standardized.summarize import load_compatible_runs

    results = tmp_path / "results"
    _write_run(results, "gsat", "star", 1234)
    _write_run(results, "gsat", "star", 1235, split_hash="c" * 64)

    with pytest.raises(ValueError, match="split SHA-256"):
        load_compatible_runs(results)


def test_rejects_noncanonical_primary_topology_policy_even_when_self_consistent(tmp_path):
    from comparison.standardized.summarize import load_compatible_runs

    results = tmp_path / "results"
    _write_run(
        results,
        "gsat",
        "star",
        1234,
        topology_parameters={
            "patient_hub": True,
            "patient_concept_edges": "bidirectional",
            "concept_concept_edges": "none",
            "unexpected_policy_change": True,
        },
    )

    with pytest.raises(ValueError, match="canonical topology policy"):
        load_compatible_runs(results)


def test_rejects_different_canonical_policies_for_the_same_topology(tmp_path):
    from comparison.standardized.summarize import load_compatible_runs

    results = tmp_path / "results"
    _write_run(results, "graphcare", "ontology", 1234)
    _write_run(
        results,
        "graphcare",
        "ontology",
        1235,
        topology_parameters={
            "patient_hub": True,
            "patient_concept_edges": "bidirectional",
            "concept_concept_edges": "ontology",
            "unexpected_policy_change": True,
        },
    )

    with pytest.raises(ValueError, match="incompatible topology-policy fingerprints"):
        load_compatible_runs(results)


@pytest.mark.parametrize(
    "mutate,error_match",
    [
        (lambda manifest: manifest.__setitem__("schema_version", 2), "schema_version"),
        (lambda manifest: manifest.__setitem__("class_ordering", list(reversed(EXPECTED_CLASSES))), "class_ordering"),
        (lambda manifest: manifest["canonical_fold_counts"].__setitem__("test", 1), "canonical_fold_counts"),
        (lambda manifest: manifest["effective_fold_counts"].__setitem__("test", 1), "effective_fold_counts"),
        (lambda manifest: manifest.__setitem__("topology_policy_fingerprint", "0" * 64), "topology-policy fingerprint"),
        (lambda manifest: manifest.pop("topology_policy_fingerprint"), "topology-policy fingerprint"),
        (lambda manifest: manifest.__setitem__("parameter_count", True), "parameter_count"),
        (lambda manifest: manifest.__setitem__("method", "unknown"), "method"),
        (lambda manifest: manifest.__setitem__("topology", "unknown"), "topology"),
        (lambda manifest: manifest.__setitem__("seed", 9), "seed"),
    ],
)
def test_rejects_manifest_fields_outside_the_exact_contract(tmp_path, mutate, error_match):
    from comparison.standardized.summarize import load_compatible_runs

    results = tmp_path / "results"
    manifest_path = _write_run(results)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((KeyError, TypeError, ValueError), match=error_match):
        load_compatible_runs(results)


@pytest.mark.parametrize(
    "mutate,error_match",
    [
        (lambda payload: payload["test"].pop("accuracy"), "exact six"),
        (lambda payload: payload["test"].__setitem__("unexpected_metric", 0.1), "exact six"),
        (lambda payload: payload["test"].__setitem__("accuracy", True), "finite numeric"),
        (lambda payload: payload["test"].__setitem__("accuracy", float("nan")), "finite numeric"),
        (lambda payload: payload["test"].__setitem__("accuracy", 1.1), r"\[0, 1\]"),
        (lambda payload: payload.__setitem__("parameter_count", 42.0), "parameter_count"),
        (lambda payload: payload["test"].__setitem__("parameter_count", 42.0), "parameter_count"),
        (lambda payload: payload.__setitem__("parameter_count", 43), "parameter_count"),
    ],
)
def test_rejects_metrics_outside_the_exact_contract(tmp_path, mutate, error_match):
    from comparison.standardized.summarize import load_compatible_runs

    results = tmp_path / "results"
    manifest_path = _write_run(results)
    metrics_path = manifest_path.parent / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    mutate(payload)
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((KeyError, TypeError, ValueError), match=error_match):
        load_compatible_runs(results)


def test_retains_failed_and_running_manifests_as_incomplete_without_reading_metrics(tmp_path):
    from comparison.standardized.summarize import load_compatible_runs

    results = tmp_path / "results"
    _write_run(results, "gsat", "star", 1234)
    _write_run(results, "gsat", "star", 1235, status="failed")
    _write_run(results, "gsat", "star", 1236, status="running")

    runs = load_compatible_runs(results)

    assert [(run.method, run.topology, run.seed) for run in runs] == [
        ("gsat", "star", 1234)
    ]
    assert [
        (run.method, run.topology, run.seed, run.status)
        for run in runs.incomplete
    ] == [
        ("gsat", "star", 1235, "failed"),
        ("gsat", "star", 1236, "running"),
    ]


def test_retains_incomplete_common_secondary_and_method_specific_runs(tmp_path):
    from comparison.standardized.summarize import load_compatible_runs, summarize_runs

    results = tmp_path / "results"
    _write_run(results, "graphcare", "ontology", 1236, status="running")
    _write_run(results, "graphcare", "full_kg_expanded", 1234, status="failed")
    _write_run(results, "graphcare", "full_kg_expanded", 1235, status="running")

    summary = summarize_runs(load_compatible_runs(results))

    assert summary.incomplete_run_rows == (
        {
            "category": "common_secondary",
            "method": "graphcare",
            "topology": "ontology",
            "seed": 1236,
            "status": "running",
            "error": "",
        },
        {
            "category": "method_specific",
            "method": "graphcare",
            "topology": "full_kg_expanded",
            "seed": 1234,
            "status": "failed",
            "error": "training failed",
        },
        {
            "category": "method_specific",
            "method": "graphcare",
            "topology": "full_kg_expanded",
            "seed": 1235,
            "status": "running",
            "error": "",
        },
    )


def test_summarizes_only_exact_three_seed_primary_cells_with_sample_sd(tmp_path):
    from comparison.standardized.summarize import load_compatible_runs, summarize_runs

    results = tmp_path / "results"
    for seed in (1234, 1235, 1236):
        _write_run(results, "gsat", "star", seed)
        _write_run(results, "graphcare", "ontology", seed)
    for seed in (1234, 1235):
        _write_run(results, "protgnn", "star", seed)

    summary = summarize_runs(load_compatible_runs(results))

    assert len(summary.per_run_rows) == 8
    assert len(summary.aggregate_rows) == 1
    aggregate = summary.aggregate_rows[0]
    assert (aggregate["method"], aggregate["topology"], aggregate["seed_count"]) == (
        "gsat", "star", 3
    )
    assert aggregate["parameter_count"] == 42
    for index, metric in enumerate(STANDARDIZED_METRICS):
        assert aggregate[f"{metric}_mean"] == pytest.approx((index + 2) / 10.0)
        assert aggregate[f"{metric}_sd"] == pytest.approx(0.1)
    assert summary.incomplete_cells == ({
        "method": "protgnn",
        "topology": "star",
        "completed_seeds": "1234;1235",
        "missing_seeds": "1236",
        "recorded_incomplete": "",
    },)
    assert len(summary.common_secondary_rows) == 3
    assert {row["topology"] for row in summary.common_secondary_rows} == {"ontology"}
    assert summary.method_specific_rows == ()


def test_cli_writes_three_deterministic_explicit_outputs_and_separate_sections(tmp_path):
    from comparison.standardized.summarize import main

    results = tmp_path / "fixture-results"
    for seed in (1234, 1235, 1236):
        _write_run(results, "gsat", "star", seed)
    for seed in (1234, 1235):
        _write_run(results, "protgnn", "star", seed)
    _write_run(results, "protgnn", "star", 1236, status="failed")
    _write_run(results, "graphcare", "ontology", 1234)
    before = {
        path.relative_to(results): path.read_bytes()
        for path in results.rglob("*")
        if path.is_file()
    }
    output_dir = tmp_path / "reports"
    summary_csv = output_dir / "summary.csv"
    aggregate_csv = output_dir / "summary_aggregate.csv"
    summary_md = output_dir / "summary.md"
    argv = [
        "--results-dir", str(results),
        "--summary-csv", str(summary_csv),
        "--summary-aggregate-csv", str(aggregate_csv),
        "--summary-md", str(summary_md),
    ]

    assert main(argv) == 0
    first = tuple(path.read_bytes() for path in (summary_csv, aggregate_csv, summary_md))
    assert main(argv) == 0
    second = tuple(path.read_bytes() for path in (summary_csv, aggregate_csv, summary_md))

    assert first == second
    assert before == {
        path.relative_to(results): path.read_bytes()
        for path in results.rglob("*")
        if path.is_file()
    }
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert rows[2]["status"] == "failed"
    assert rows[2]["category"] == "primary"
    assert rows[-1]["category"] == "common_secondary"
    with aggregate_csv.open(newline="", encoding="utf-8") as handle:
        aggregate = list(csv.DictReader(handle))
    assert [(row["method"], row["topology"]) for row in aggregate] == [
        ("gsat", "star")
    ]
    markdown = summary_md.read_text(encoding="utf-8")
    assert markdown.startswith("# Standardized Benchmark Summary\n\n## Primary comparison\n\n|")
    assert "## Incomplete primary cells" in markdown
    assert "1236:failed" in markdown
    assert "## Common secondary runs" in markdown
    assert "## Method-specific runs" in markdown
    assert "## Incomplete runs" in markdown
    assert "ontology" in markdown


def test_cli_classifies_topologies_and_reports_every_incomplete_run(tmp_path):
    from comparison.standardized.summarize import main

    results = tmp_path / "fixture-results"
    _write_run(results, "protgnn", "full", 1235)
    _write_run(results, "gsat", "ontology", 1236, status="running")
    _write_run(results, "graphcare", "ontology", 1234)
    _write_run(results, "graphcare", "full_kg_expanded", 1234, status="failed")
    _write_run(results, "graphcare", "full_kg_expanded", 1235, status="running")
    _write_run(results, "graphcare", "full_kg_expanded", 1236)
    output_dir = tmp_path / "reports"
    summary_csv = output_dir / "summary.csv"
    aggregate_csv = output_dir / "summary_aggregate.csv"
    summary_md = output_dir / "summary.md"
    argv = [
        "--results-dir", str(results),
        "--summary-csv", str(summary_csv),
        "--summary-aggregate-csv", str(aggregate_csv),
        "--summary-md", str(summary_md),
    ]

    assert main(argv) == 0
    first = tuple(path.read_bytes() for path in (summary_csv, aggregate_csv, summary_md))
    assert main(argv) == 0
    assert first == tuple(path.read_bytes() for path in (summary_csv, aggregate_csv, summary_md))

    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [
        (row["category"], row["method"], row["topology"], row["seed"], row["status"])
        for row in rows
    ] == [
        ("common_secondary", "protgnn", "full", "1235", "completed"),
        ("common_secondary", "gsat", "ontology", "1236", "running"),
        ("common_secondary", "graphcare", "ontology", "1234", "completed"),
        ("method_specific", "graphcare", "full_kg_expanded", "1234", "failed"),
        ("method_specific", "graphcare", "full_kg_expanded", "1235", "running"),
        ("method_specific", "graphcare", "full_kg_expanded", "1236", "completed"),
    ]

    markdown = summary_md.read_text(encoding="utf-8")
    assert "## Common secondary runs" in markdown
    assert "| protgnn | full | 1235 |" in markdown
    assert "| graphcare | ontology | 1234 |" in markdown
    assert "## Method-specific runs" in markdown
    assert "| graphcare | full_kg_expanded | 1236 |" in markdown
    assert "## Incomplete runs" in markdown
    assert "| common_secondary | gsat | ontology | 1236 | running |" in markdown
    assert "| method_specific | graphcare | full_kg_expanded | 1234 | failed | training failed |" in markdown
    assert "| method_specific | graphcare | full_kg_expanded | 1235 | running |  |" in markdown
