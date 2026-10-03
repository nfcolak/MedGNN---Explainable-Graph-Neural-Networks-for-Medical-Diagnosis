"""Focused synthetic checks for source-derived visit membership.

No patient data, artifact generation, or model training is used here.
"""
import copy
import json
import sqlite3

import pytest
import torch
from torch_geometric.data import Batch

from core.contracts import (
    LOGIC_CONTRACT_VERSION,
    VISIT_MEMBERSHIP_CONTRACT_VERSION,
    VISIT_MEMBERSHIP_FILENAME,
    recursive_source_hashes,
    validate_artifact_manifest,
    validate_visit_membership_record,
    verify_visit_membership_file,
)
from core.graph import build_graph_with_visit_membership
from core.schema import sha256, timestamp
from core.store import ClinicalStore
from core.tensorize import (
    ClinicalGraphData,
    load_preprocessing,
)


class OrderedMembershipStore:
    stays = {
        "prior-a-secret": {
            "subject": "subject-secret",
            "start": "2020-01-01T00:00:00",
            "finish": "2020-01-01T12:00:00",
        },
        "prior-b-secret": {
            "subject": "subject-secret",
            "start": "2020-01-02T00:00:00",
            "finish": "2020-01-02T12:00:00",
        },
        "index-secret": {
            "subject": "subject-secret",
            "start": "2020-01-03T00:00:00",
            "finish": "2020-01-03T02:00:00",
        },
    }

    def visit(self, stay):
        return self.stays.get(stay)

    def arrival(self, stay):
        return {}

    def complaints(self, stay):
        return ["pain"]

    def triage_vitals(self, stay):
        return []

    def prior_visits(self, subject, start):
        assert subject == "subject-secret"
        return ["prior-a-secret", "prior-b-secret"]

    @staticmethod
    def _record(identifier, stay, when, value):
        return {
            "id": identifier,
            "source": identifier,
            "token": "lab:synthetic",
            "time": when,
            "available": when,
            "value": value,
            "unit": "mg/dL",
            "timing_basis": "synthetic",
            "stay": stay,
        }

    def measurements(self, stay, cutoff):
        return [self._record(
            "index-event",
            "index-secret",
            "2020-01-03T00:30:00",
            7.0,
        )]

    def analyte_history(self, subject, prior_visits, analyte_token, cutoff):
        assert prior_visits == ["prior-a-secret", "prior-b-secret"]
        return [self._record(
            "prior-event",
            "prior-a-secret",
            "2020-01-01T01:00:00",
            2.0,
        )]


SAMPLE = {
    "stay_id": "index-secret",
    "subject_id": "subject-secret",
    "cutoff": timestamp("2020-01-03T01:00:00"),
    "sample_id": "synthetic-sample",
    "split": "train",
}


def build_membership(store=None):
    return build_graph_with_visit_membership(
        store or OrderedMembershipStore(),
        SAMPLE,
        knowledge=[],
        vocabulary={"pain"},
    )


def test_source_order_empty_visit_multi_membership_and_identifier_exclusion():
    graph, membership = build_membership()
    validate_visit_membership_record(graph, membership)

    assert membership["visit_ordinals"] == [0, 1, 2]
    pair_set = {tuple(pair) for pair in membership["membership_pairs"]}
    prior_node = next(
        index for index, node in enumerate(graph["nodes"])
        if node.get("source_record") == "prior-event"
    )
    index_visit = next(
        index for index, node in enumerate(graph["nodes"])
        if node["kind"] == "visit"
    )
    analyte = next(
        index for index, node in enumerate(graph["nodes"])
        if node["kind"] == "analyte"
    )

    assert (0, prior_node) in pair_set
    assert (2, index_visit) in pair_set
    assert (0, analyte) in pair_set and (2, analyte) in pair_set
    assert not any(ordinal == 1 for ordinal, _ in pair_set)  # retained empty visit
    assert membership["global_node_mask"].count(True) == 1

    encoded = json.dumps(membership, sort_keys=True)
    for raw_identifier in (
        "subject-secret",
        "prior-a-secret",
        "prior-b-secret",
        "index-secret",
    ):
        assert raw_identifier not in encoded
    assert set(membership) == {
        "contract_version",
        "sample_id",
        "visit_ordinals",
        "membership_pairs",
        "global_node_mask",
    }


def test_duplicate_or_missing_source_lineage_fails_closed():
    class DuplicateVisitStore(OrderedMembershipStore):
        def prior_visits(self, subject, start):
            return ["prior-a-secret", "prior-a-secret"]

    with pytest.raises(ValueError, match="Duplicate source stay"):
        build_membership(DuplicateVisitStore())

    class MissingLineageStore(OrderedMembershipStore):
        def measurements(self, stay, cutoff):
            return [self._record(
                "bad-event",
                "outside-lineage-secret",
                "2020-01-03T00:30:00",
                7.0,
            )]

    with pytest.raises(ValueError, match="outside the sample visit lineage"):
        build_membership(MissingLineageStore())

    graph, membership = build_membership()
    malformed = copy.deepcopy(membership)
    complaint_index = next(
        index for index, node in enumerate(graph["nodes"])
        if node["kind"] == "complaint"
    )
    malformed["membership_pairs"] = [
        pair for pair in malformed["membership_pairs"] if pair[1] != complaint_index
    ]
    with pytest.raises(ValueError, match="has no membership"):
        validate_visit_membership_record(graph, malformed)


def test_clinical_store_preserves_measurement_source_stays(tmp_path):
    event_index = tmp_path / "events.sqlite"
    events = sqlite3.connect(event_index)
    events.executescript("""
        CREATE TABLE visits (stay TEXT PRIMARY KEY, subject TEXT, start TEXT, finish TEXT);
        CREATE TABLE events (id TEXT PRIMARY KEY, stay TEXT, subject TEXT, token TEXT,
            value REAL, unit TEXT, time TEXT, available TEXT, source TEXT,
            timing_basis TEXT);
    """)
    events.executemany(
        "INSERT INTO visits VALUES (?,?,?,?)",
        [
            ("prior", "subject", "2020-01-01T00:00:00", "2020-01-01T12:00:00"),
            ("index", "subject", "2020-01-03T00:00:00", "2020-01-03T02:00:00"),
        ],
    )
    events.executemany(
        "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("prior-event", "prior", "subject", "lab:synthetic", 2.0, "mg/dL",
             "2020-01-01T01:00:00", "2020-01-01T01:00:00", "prior-event", "synthetic"),
            ("index-event", "index", "subject", "lab:synthetic", 7.0, "mg/dL",
             "2020-01-03T00:30:00", "2020-01-03T00:30:00", "index-event", "synthetic"),
        ],
    )
    events.commit()
    events.close()

    triage_index = tmp_path / "triage.sqlite"
    triage = sqlite3.connect(triage_index)
    triage.executescript("""
        CREATE TABLE arrival (stay TEXT PRIMARY KEY, subject TEXT, gender TEXT,
            race TEXT, arrival_transport TEXT, age REAL, acuity REAL);
        CREATE TABLE complaints (stay TEXT, ordinal INTEGER, token TEXT);
        CREATE TABLE vitals (stay TEXT, field TEXT, value REAL, unit TEXT, source_unit TEXT);
        INSERT INTO arrival VALUES ('index','subject',NULL,NULL,NULL,NULL,NULL);
    """)
    triage.commit()
    triage.close()

    store = ClinicalStore(event_index, triage_index)
    cutoff = timestamp("2020-01-03T01:00:00")
    assert store.measurements("index", cutoff)[0]["stay"] == "index"
    assert store.analyte_history(
        "subject", ["prior"], "lab:synthetic", cutoff
    )[0]["stay"] == "prior"
    graph, membership = build_graph_with_visit_membership(
        store,
        {"stay_id": "index", "subject_id": "subject", "cutoff": cutoff,
         "sample_id": "store-backed-sample", "split": "train"},
        knowledge=[],
        vocabulary=set(),
    )
    validate_visit_membership_record(graph, membership)
    store.close()


def test_sparse_visit_pairs_and_global_mask_are_sample_local():
    first = ClinicalGraphData(
        x=torch.zeros((3, 4)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        visit_membership_index=torch.tensor([[0, 1, 1], [0, 0, 1]]),
        num_visits=torch.tensor([2]),
        global_node_mask=torch.tensor([False, False, True]),
    )
    second = ClinicalGraphData(
        x=torch.zeros((2, 4)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        visit_membership_index=torch.tensor([[0], [0]]),
        num_visits=torch.tensor([1]),
        global_node_mask=torch.tensor([False, True]),
    )
    batch = Batch.from_data_list([first, second])

    assert batch.visit_membership_index.tolist() == [[0, 1, 1, 2], [0, 0, 1, 3]]
    assert batch.num_visits.tolist() == [2, 1]
    assert batch.global_node_mask.tolist() == [False, False, True, False, True]
    assert batch.global_node_mask.numel() == batch.num_nodes


def test_recursive_source_and_sidecar_hashes_are_bound(tmp_path):
    package = tmp_path / "package"
    (package / "methods").mkdir(parents=True)
    (package / "__pycache__").mkdir()
    (package / "a.py").write_text("VALUE = 1\n")
    (package / "methods" / "b.py").write_text("VALUE = 2\n")
    (package / "__pycache__" / "ignored.py").write_text("VALUE = 3\n")

    first = recursive_source_hashes(package)
    assert list(first) == ["a.py", "methods/b.py"]
    (package / "methods" / "b.py").write_text("VALUE = 4\n")
    second = recursive_source_hashes(package)
    assert first["a.py"] == second["a.py"]
    assert first["methods/b.py"] != second["methods/b.py"]

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    sidecar = artifact / VISIT_MEMBERSHIP_FILENAME
    sidecar.write_text(json.dumps({"synthetic": True}) + "\n")
    manifest = {
        "visit_membership_file": VISIT_MEMBERSHIP_FILENAME,
        "visit_membership_sha256": sha256(sidecar),
        "visit_membership_rows": 1,
    }
    assert verify_visit_membership_file(artifact, manifest) == 1
    sidecar.write_text(json.dumps({"synthetic": False}) + "\n")
    with pytest.raises(ValueError, match="missing or differs"):
        verify_visit_membership_file(artifact, manifest)


def test_legacy_artifact_and_preprocessing_contracts_are_rejected():
    legacy_manifest = {
        "status": "completed",
        "logic_contract_version": LOGIC_CONTRACT_VERSION,
        "temporal_clean": False,
    }
    with pytest.raises(ValueError, match="lacks source-derived visit membership"):
        validate_artifact_manifest(legacy_manifest)

    with pytest.raises(ValueError, match="historical checkpoints"):
        load_preprocessing({"preprocessing_version": "clinical_inputs_v2"})

    current = {
        **legacy_manifest,
        "visit_membership_contract_version": VISIT_MEMBERSHIP_CONTRACT_VERSION,
        "visit_membership_file": VISIT_MEMBERSHIP_FILENAME,
        "visit_membership_rows": 1,
        "visit_membership_sha256": "0" * 64,
    }
    validate_artifact_manifest(current)
