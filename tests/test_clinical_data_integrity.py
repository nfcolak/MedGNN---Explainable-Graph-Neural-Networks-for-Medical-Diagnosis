"""Bounded synthetic regressions; never read or mutate production artifacts."""
import csv
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.s4_graph.diagnosis import DiagnosisIndex, load_icd_map, normalize


def write_csv(path, fields, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return path


def test_history_mapping_matches_frozen_target_policy(tmp_path):
    fields = ['icd9_code', 'icd10_code', 'approximate', 'no_map']
    records = [dict(zip(fields, values)) for values in [
        ('001', 'A009', '1', '0'), ('001', 'B001', '0', '0'),
        ('002', 'C001', '1', '0'), ('002', 'C001', '0', '0'),
        ('003', 'D001', '1', '0'), ('003', 'E001', '1', '0'),
        ('004', 'NoDx', '0', '1'), ('005', 'F001', '1', '0'),
    ]]
    # Enough ties to exercise the existing pandas quicksort policy, not a new
    # stable-sort tie policy which would silently change historical targets.
    records.extend(dict(zip(fields, (str(n), 'G001', '1', '0'))) for n in range(100, 130))
    path = write_csv(tmp_path / 'mapping.csv', fields, records)
    frozen = pd.read_csv(path, dtype=str, keep_default_na=False)
    frozen['approximate'] = pd.to_numeric(frozen.approximate, errors='raise')
    frozen = frozen[frozen.no_map == '0'].sort_values('approximate').drop_duplicates('icd9_code')
    expected = dict(zip(frozen.icd9_code.str.upper(), frozen.icd10_code.str.upper()))
    mapping, approximate = load_icd_map(path)
    assert mapping == expected
    assert approximate == set(frozen.loc[frozen.approximate == 1, 'icd9_code'])
    assert normalize('9', '001', mapping, approximate) == ('dx:icd10:B001', False)
    assert normalize('9', '004', mapping, approximate) == ('dx:icd9:004', False)
    assert normalize('10', ' b001 ', mapping, approximate) == ('dx:icd10:B001', False)
    assert normalize('9', '005', mapping, approximate) == ('dx:icd10:F001', True)
    assert normalize('10', '', mapping, approximate) == (None, False)
    with pytest.raises(ValueError, match='Unsupported ICD version'):
        normalize('11', 'A009', mapping, approximate)


def make_diagnoses(tmp_path):
    fields = ['subject_id', 'stay_id', 'seq_num', 'icd_code', 'icd_version', 'icd_title']
    values = [
        ('1', '10', '1', '001', '9', 'first'),
        ('1', '10', '2', 'B001', '10', 'same normalized code'),
        ('1', '10', '3', '001', '9', 'duplicate'),
        ('1', '11', '1', 'B001', '10', 'second encounter'),
        ('1', '12', '1', 'C001', '10', 'index-only label'),
    ]
    path = write_csv(tmp_path / 'diagnosis.csv', fields,
                     [dict(zip(fields, row)) for row in values])
    return DiagnosisIndex(path, {'10', '11', '12'}, {'001': 'B001'}, {'001'})


def test_recurrence_counts_unique_encounters_not_diagnosis_rows(tmp_path):
    diagnoses = make_diagnoses(tmp_path)
    history = diagnoses.prior_history('12', ['10', '10', '11'])
    assert len(history) == 1
    assert history[0]['occurrences'] == ['10', '11']
    assert history[0]['approximate_mapping'] is True
    assert history[0]['titles'] == {'first', 'same normalized code', 'duplicate', 'second encounter'}
    with pytest.raises(ValueError, match='Index encounter'):
        diagnoses.prior_history('12', ['10', '12'])


def make_clinical_store(tmp_path):
    from data_pipeline.s4_graph.store import ClinicalStore
    events = tmp_path / 'events.sqlite'
    db = sqlite3.connect(events)
    db.executescript('''
        CREATE TABLE visits(stay TEXT PRIMARY KEY, subject TEXT, start TEXT, finish TEXT);
        CREATE TABLE events(id TEXT PRIMARY KEY, subject TEXT, stay TEXT, token TEXT,
            value REAL, unit TEXT, time TEXT, available TEXT, source TEXT, timing_basis TEXT);
        INSERT INTO visits VALUES('10','1','2020-01-01T00:00:00','2020-01-01T12:00:00');
        INSERT INTO visits VALUES('11','1','2020-01-02T00:00:00','2020-01-02T12:00:00');
        INSERT INTO visits VALUES('12','1','2020-01-03T00:00:00','2020-01-03T12:00:00');
        INSERT INTO visits VALUES('13','1','2020-01-02T00:00:00',NULL);
        INSERT INTO visits VALUES('14','1','2020-01-02T00:00:00','2020-01-03T00:00:00');
        INSERT INTO visits VALUES('15','2','2020-01-01T00:00:00','2020-01-01T12:00:00');
        INSERT INTO events VALUES('lab:1','1','12','lab:1',1.0,'u','2020-01-03T00:30:00',
            '2020-01-03T01:00:00','labevents.csv:1','storetime');
    ''')
    db.close()
    triage = tmp_path / 'triage.sqlite'
    db = sqlite3.connect(triage)
    db.executescript('''
        CREATE TABLE arrival(stay TEXT, subject TEXT, acuity REAL);
        CREATE TABLE complaints(stay TEXT, ordinal INTEGER, token TEXT);
        CREATE TABLE vitals(stay TEXT, field TEXT, value REAL, unit TEXT, source_unit TEXT);
        INSERT INTO arrival VALUES('12','1',2.0);
        INSERT INTO complaints VALUES('12',0,'pain');
        INSERT INTO vitals VALUES('12','heartrate',80,'beats/min','beats/min');
    ''')
    db.close()
    return ClinicalStore(events, triage)


def test_graph_temporal_claims_and_unique_history(tmp_path):
    from datetime import datetime
    from data_pipeline.s4_graph.graph import build_graph
    from data_pipeline.s4_graph.build import verify_artifact
    store = make_clinical_store(tmp_path)
    diagnoses = make_diagnoses(tmp_path)
    sample = dict(sample_id='sample', subject_id='1', stay_id='12', split='train',
                  cutoff=datetime.fromisoformat('2020-01-03T01:00:00'))
    try:
        assert store.prior_visits('1', '2020-01-03T00:00:00') == ['10', '11']
        graph = build_graph(store, sample, [], {'pain'}, diagnoses)
        assert graph.get('temporal_clean') is False
        assert graph['logic_contract_version'] == 'clinical_graph_logic_v2'
        assert graph['temporal_validation'] == 'proxy_bounds_only; availability_assumptions_unverified'
        dx = next(n for n in graph['nodes'] if n['kind'] == 'diagnosis')
        assert dx['prior_encounters'] == 2
        assert dx['timestamp_source'] == 'edstays.csv:outtime'
        assert dx['availability_assumed'] is True
        # Retain the old missing timestamp semantics; end time is only an
        # explicitly named proxy, not a newly invented diagnosis observation.
        assert dx['time_hours'] is None and dx['available_hours'] is None
        assert dx['assumed_available_hours'] < 0
        for node in graph['nodes']:
            if node['kind'] in ('complaint', 'vital'):
                assert node['availability_verified'] is False
                assert node['availability_assumed'] is True
                assert node['timestamp_source'] == 'edstays.csv:intime'
            if node['kind'] == 'measurement':
                assert node['availability_verified'] is False
                assert node['availability_basis'] == 'database_storetime_proxy'
        recurrence = next(e for e in graph['edges'] if e['relation'] == 'recurrence_of')
        assert recurrence['prior_encounters'] == 2
        path = tmp_path / 'graphs.jsonl'
        path.write_text(json.dumps(graph) + '\n')
        verification = verify_artifact(tmp_path, [sample], store, {'pain'}, diagnoses)
        assert verification['temporal_clean'] is False
        assert verification['index_diagnosis_leaks'] == 0
        dx['prior_encounters'] = 4
        path.write_text(json.dumps(graph) + '\n')
        with pytest.raises(ValueError, match='encounter count'):
            verify_artifact(tmp_path, [sample], store, {'pain'}, diagnoses)
    finally:
        store.close()


def event_artifact(root, subjects):
    import hashlib
    root.mkdir()
    canonical = {'fold': {'1': 0, '2': 1}, 'classes': ['Second', 'First']}
    (root / 'canonical_split.json').write_text(json.dumps(canonical))
    db = sqlite3.connect(root / 'events.sqlite')
    db.executescript('''
        CREATE TABLE visits(stay TEXT PRIMARY KEY, subject TEXT, start TEXT, finish TEXT);
        CREATE TABLE first_results(stay TEXT PRIMARY KEY, available TEXT);
        CREATE TABLE lab_presence(stay TEXT PRIMARY KEY);
        CREATE TABLE excluded_visits(stay TEXT PRIMARY KEY, subject TEXT, raw_start TEXT,
            raw_finish TEXT, reason TEXT);
        CREATE TABLE events(id TEXT PRIMARY KEY, subject TEXT, stay TEXT, time TEXT,
            available TEXT, token TEXT, value REAL, unit TEXT, source TEXT, timing_basis TEXT);
    ''')
    cohort = []
    for subject in subjects:
        for n in range(7 if subject == '1' else 1):
            stay = str(int(subject) * 100 + n)
            start, cutoff, finish = [f'2020-01-{n + 1:02d}T{h}:00:00' for h in ('00', '01', '12')]
            db.execute('INSERT INTO visits VALUES(?,?,?,?)', (stay, subject, start, finish))
            db.execute('INSERT INTO first_results VALUES(?,?)', (stay, cutoff))
            db.execute('INSERT INTO lab_presence VALUES(?)', (stay,))
            db.execute('INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?,?)',
                       ('lab:' + stay, subject, stay, start, cutoff, 'lab:1', 1, 'u', 'fixture', 'storetime'))
            cohort.append(dict(sample_id=hashlib.sha256(('event-first-lab-v1:' + subject + ':' + stay).encode()).hexdigest(),
                               subject_id=subject, stay_id=stay, cutoff=cutoff,
                               split='train' if subject == '1' else 'validation'))
    db.commit()
    db.close()
    write_csv(root / 'cohort.csv', ['sample_id', 'subject_id', 'stay_id', 'cutoff', 'split'], cohort)
    from core.schema import sha256
    manifest = {'schema_version': 'event_graph_v1', 'status': 'completed',
                'cohort_sha256': 'stale-cohort', 'event_index_sha256': 'stale-index',
                'cohort_policy': {'samples': 999}, 'counts': {'graphs': 999},
                'graphs_sha256': 'phantom', 'graphs_bytes': 999, 'coverage': {'train': {'graphs': 999}},
                'verification_sha256': 'phantom', 'visit_filter': {'max_visits_per_subject': 6},
                'source_bindings': {'canonical_split': {'sha256': sha256(root / 'canonical_split.json')},
                                    'labevents': {'path': '/DO_NOT_READ_RAW.csv', 'sha256': 'historical-raw-hash'}},
                'source_code': {'old.py': 'historical-code-hash'},
                'reference_class_order_only': canonical['classes']}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'prepared_index.json').write_text(json.dumps({'status': 'prepared', 'cohort_policy': {'samples': 999}}))
    (root / 'verification.json').write_text(json.dumps({'status': 'verified', 'counts': {'graphs': 999}}))
    (root / 'delivery_report.json').write_text(json.dumps({'status': 'completed', 'graphs_sha256': 'phantom'}))
    return root


def snapshot_tree(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_metadata_repair_separates_hardlinks_without_mutating_data(tmp_path):
    from data_pipeline.s2_events import repair_metadata
    full = event_artifact(tmp_path / 'full', ['1', '2'])
    filtered = event_artifact(tmp_path / 'max6', ['2'])
    for name in ('manifest.json', 'prepared_index.json', 'delivery_report.json', 'verification.json'):
        (filtered / name).unlink()
        os.link(full / name, filtered / name)
    initial = snapshot_tree(tmp_path)
    preview = repair_metadata.repair(full)
    assert preview['status'] == 'dry_run'
    assert snapshot_tree(tmp_path) == initial
    result = repair_metadata.repair(full, execute=True, offline=True)
    assert result['status'] == 'repaired'
    assert (filtered / 'manifest.json').read_bytes() == initial['max6/manifest.json']
    repair_metadata.repair(filtered, source_artifact=full, max_visits_per_subject=6,
                           execute=True, offline=True)
    for root, expected in ((full, 8), (filtered, 1)):
        manifest = json.loads((root / 'manifest.json').read_text())
        prepared = json.loads((root / 'prepared_index.json').read_text())
        assert manifest['status'] == prepared['status'] == 'index_ready'
        assert manifest['cohort_policy']['samples'] == prepared['cohort_policy']['samples'] == expected
        assert manifest['temporal_clean'] is False
        assert manifest['reference_class_order_only'] == ['Second', 'First']
        assert manifest['source_code'] == {'old.py': 'historical-code-hash'}
        assert manifest['source_bindings']['labevents']['sha256'] == 'historical-raw-hash'
        for name in ('manifest.json', 'prepared_index.json', 'verification.json', 'delivery_report.json'):
            assert (root / name).stat().st_nlink == 1
            metadata = json.loads((root / name).read_text())
            assert not {'counts', 'graphs_sha256', 'graphs_bytes', 'coverage', 'verification_sha256'} & metadata.keys()
            backups = list((root / 'metadata_backups').glob(name + '.*'))
            assert len(backups) == 1
            assert backups[0].read_bytes() == initial[root.name + '/' + name]
            assert backups[0].stat().st_mode & 0o222 == 0
        for name in ('cohort.csv', 'events.sqlite', 'canonical_split.json'):
            assert (root / name).read_bytes() == initial[root.name + '/' + name]
        assert not (root / 'graphs.jsonl').exists()
    after = snapshot_tree(tmp_path)
    assert repair_metadata.repair(full, execute=True, offline=True)['status'] == 'already_repaired'
    assert repair_metadata.repair(filtered, source_artifact=full, max_visits_per_subject=6,
                                   execute=True, offline=True)['status'] == 'already_repaired'
    assert snapshot_tree(tmp_path) == after


@pytest.mark.parametrize('fault', ['graph', 'cutoff', 'missing_stay', 'canonical', 'wal', 'symlink', 'pending', 'lock'])
def test_metadata_repair_fails_closed_without_changing_inputs(tmp_path, fault):
    from data_pipeline.s2_events import repair_metadata
    root = event_artifact(tmp_path / 'artifact', ['1', '2'])
    if fault == 'graph':
        (root / 'graphs.jsonl').write_bytes(b'IMMUTABLE EXISTING GRAPH\n')
    elif fault == 'cutoff':
        data = (root / 'cohort.csv').read_text().replace('T01:00:00', 'T02:00:00')
        (root / 'cohort.csv').write_text(data)
    elif fault == 'missing_stay':
        lines = (root / 'cohort.csv').read_text().splitlines()
        (root / 'cohort.csv').write_text('\n'.join(lines[:-1]) + '\n')
    elif fault == 'canonical':
        (root / 'canonical_split.json').write_text('{}')
    elif fault == 'wal':
        (root / 'events.sqlite-wal').write_bytes(b'not-checkpointed')
    elif fault == 'symlink':
        old = (root / 'manifest.json').read_bytes()
        (tmp_path / 'external.json').write_bytes(old)
        (root / 'manifest.json').unlink()
        (root / 'manifest.json').symlink_to(tmp_path / 'external.json')
    elif fault == 'pending':
        (root / repair_metadata.PENDING).write_text('{}')
    elif fault == 'lock':
        (root / repair_metadata.LOCK).write_text('existing-owner')
    before = snapshot_tree(tmp_path)
    with pytest.raises((ValueError, FileExistsError)):
        repair_metadata.repair(root, execute=True, offline=True)
    assert snapshot_tree(tmp_path) == before


def test_metadata_requires_offline_and_explicit_filter_source(tmp_path):
    from data_pipeline.s2_events import repair_metadata
    root = event_artifact(tmp_path / 'artifact', ['2'])
    before = snapshot_tree(tmp_path)
    with pytest.raises(ValueError, match='offline'):
        repair_metadata.repair(root, execute=True)
    with pytest.raises(ValueError, match='source-artifact'):
        repair_metadata.repair(root, max_visits_per_subject=6)
    assert snapshot_tree(tmp_path) == before


def test_metadata_conflicting_backup_is_never_overwritten(tmp_path):
    from data_pipeline.s2_events import repair_metadata
    root = event_artifact(tmp_path / 'artifact', ['1', '2'])
    backup = root / 'metadata_backups'
    backup.mkdir()
    import hashlib
    old = (root / 'manifest.json').read_bytes()
    path = backup / ('manifest.json.' + hashlib.sha256(old).hexdigest() + '.json')
    path.write_bytes(b'conflicting backup')
    path.chmod(0o400)
    before = snapshot_tree(tmp_path)
    with pytest.raises(ValueError, match='backup conflicts'):
        repair_metadata.repair(root, execute=True, offline=True)
    assert snapshot_tree(tmp_path) == before


def test_interrupted_metadata_apply_is_not_certified(tmp_path, monkeypatch):
    from data_pipeline.s2_events import repair_metadata
    root = event_artifact(tmp_path / 'artifact', ['1', '2'])
    before = snapshot_tree(root)
    replace = os.replace
    calls = []

    def fail_second(source, destination):
        calls.append(destination)
        if len(calls) == 2:
            raise OSError('simulated interruption')
        return replace(source, destination)

    monkeypatch.setattr(repair_metadata.os, 'replace', fail_second)
    with pytest.raises(OSError, match='simulated interruption'):
        repair_metadata.repair(root, execute=True, offline=True)
    assert (root / repair_metadata.PENDING).exists()
    assert (root / 'manifest.json').read_bytes() == before['manifest.json']
    assert (root / 'events.sqlite').read_bytes() == before['events.sqlite']
    assert len(list((root / 'metadata_backups').iterdir())) == 4
    with pytest.raises(ValueError, match='Interrupted'):
        repair_metadata.repair(root, execute=True, offline=True)


def test_new_producer_accepts_verified_index_and_marks_assumptions(tmp_path):
    from argparse import Namespace
    from data_pipeline.s4_graph import build
    from data_pipeline.s2_events import repair_metadata
    inherited = event_artifact(tmp_path / 'inherited', ['1', '2'])
    repair_metadata.repair(inherited, execute=True, offline=True)
    raw = tmp_path / 'raw'
    raw.mkdir()
    cohort = list(csv.DictReader((inherited / 'cohort.csv').open()))
    write_csv(raw / 'edstays.csv', ['subject_id', 'stay_id', 'gender', 'race', 'arrival_transport'],
              [dict(subject_id=r['subject_id'], stay_id=r['stay_id'], gender='F', race='unknown', arrival_transport='walk') for r in cohort])
    write_csv(raw / 'patients.csv', ['subject_id', 'anchor_age'],
              [dict(subject_id=s, anchor_age='40') for s in ('1', '2')])
    fields = ['subject_id', 'stay_id', 'chiefcomplaint', 'acuity', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']
    write_csv(raw / 'triage.csv', fields,
              [dict(zip(fields, [r['subject_id'], r['stay_id'], 'pain', '2', '98.6', '80', '20', '99', '120', '70'])) for r in cohort])
    knowledge = write_csv(tmp_path / 'knowledge.csv',
                          ['source_token', 'relation', 'target_token', 'source_uri', 'source_version', 'reviewed_by'],
                          [dict(source_token='lab:1', relation='measures', target_token='knowledge:1',
                                source_uri='urn:fixture', source_version='1', reviewed_by='fixture')])
    args = Namespace(output=tmp_path / 'new-clinical', inherit_from=inherited, raw_root=raw,
                     canonical=inherited / 'canonical_split.json', knowledge=knowledge,
                     with_diagnosis=False, max_nodes=100, max_edges=100,
                     complaint_min_count=1, limit=None)
    build.run(args)
    manifest = json.loads((args.output / 'manifest.json').read_text())
    assert manifest['status'] == 'completed'
    assert manifest['temporal_clean'] is False
    assert manifest['logic_contract_version'] == 'clinical_graph_logic_v2'
    assert manifest['triage_admissibility']['actual_availability_verified'] is False
    assert manifest['reference_class_order_only'] == ['Second', 'First']
    assert manifest['cohort_sha256'] == manifest['inherited_cohort_sha256']
    assert any(k.endswith('/icd_mapping.py') for k in manifest['source_code'])
    with pytest.raises(FileExistsError):
        build.run(args)


def test_target_builder_explicit_graph_root_preserves_class_order(tmp_path, monkeypatch):
    from data_pipeline.s2_events import repair_metadata
    from core.schema import sha256
    from data_pipeline.s3_labels import local_labels_v2 as targets, labels as label_helpers
    graph_root = event_artifact(tmp_path / 'selected-graph', ['1', '2'])
    repair_metadata.repair(graph_root, execute=True, offline=True)
    raw = tmp_path / 'raw'
    raw.mkdir()
    cohort = list(csv.DictReader((graph_root / 'cohort.csv').open()))
    fields = ['subject_id', 'stay_id', 'seq_num', 'icd_code', 'icd_version', 'icd_title']
    write_csv(raw / 'diagnosis.csv', fields,
              [dict(zip(fields, [r['subject_id'], r['stay_id'], '1',
                                 'B001' if r['subject_id'] == '1' else 'C001', '10',
                                 'First' if r['subject_id'] == '1' else 'Second'])) for r in cohort])
    write_csv(raw / 'icd9_to_icd10_mapping.csv', ['icd9_code', 'icd10_code', 'approximate', 'no_map'],
              [dict(icd9_code='001', icd10_code='B001', approximate='0', no_map='0')])
    merge = tmp_path / 'data_pipeline/s1_clean/merge_ed.py'
    merge.parent.mkdir(parents=True)
    merge.write_text('DISEASE_MERGES = {}\n')
    contract = tmp_path / 'labels.json'
    contract.write_text(json.dumps({'labels': ['Second', 'First']}))
    historical = tmp_path / 'first_recorded_lab_all_visits_v2_targets_v1/binding_manifest.json'
    historical.parent.mkdir()
    historical.write_text(json.dumps({'labels': ['Second', 'First'],
        'raw_diagnosis_sha256': sha256(raw / 'diagnosis.csv'),
        'icd_mapping_sha256': sha256(raw / 'icd9_to_icd10_mapping.csv'),
        'merge_policy_sha256': sha256(merge)}))
    monkeypatch.setattr(targets, 'DEFAULT_GRAPH_ROOT', tmp_path / 'wrong-default')
    monkeypatch.setattr(targets, 'DEFAULT_RAW_ROOT', raw)
    monkeypatch.setattr(targets, 'DEFAULT_LABELS', contract)
    monkeypatch.setattr(targets, 'REPO', tmp_path)
    monkeypatch.setattr(label_helpers, 'REPO', tmp_path)
    output = tmp_path / 'new-targets'
    manifest = targets.build(output, graph_root=graph_root)
    assert manifest['graph_artifact'] == str(graph_root)
    assert manifest['cohort_sha256'] == sha256(graph_root / 'cohort.csv')
    assert manifest['graph_manifest_sha256'] == sha256(graph_root / 'manifest.json')
    assert manifest['labels'] == ['Second', 'First']
    assert manifest['icd_mapping_policy'] == 'local_labels_v2_gem_approximate_quicksort_first_v1'
    assert manifest['graph_sha256'] is None
    assert manifest['target_binding_scope'] == 'cohort_and_index_only'
    bound = pd.read_csv(output / 'targets.csv', dtype=str, keep_default_na=False)
    assert bound.loc[bound.subject_id == '1', 'target'].eq('1').all()
    assert bound.loc[bound.subject_id == '2', 'target'].eq('0').all()
    frozen = json.loads((output / 'frozen_category_map.json').read_text())
    assert frozen['category_to_label'] == {'B00': 'First', 'C00': 'Second'}
    assert manifest['artifact_files']['targets.csv'] == sha256(output / 'targets.csv')
    before = snapshot_tree(output)
    with pytest.raises(FileExistsError):
        targets.build(output, graph_root=graph_root)
    assert snapshot_tree(output) == before


def test_metadata_does_not_mistake_filtered_index_for_full(tmp_path):
    from data_pipeline.s2_events import repair_metadata
    filtered = event_artifact(tmp_path / 'max6', ['2'])
    before = snapshot_tree(tmp_path)
    with pytest.raises(ValueError, match='canonical subject'):
        repair_metadata.repair(filtered, execute=True, offline=True)
    assert snapshot_tree(tmp_path) == before


def test_metadata_refuses_an_active_producer_state(tmp_path):
    from data_pipeline.s2_events import repair_metadata
    root = event_artifact(tmp_path / 'artifact', ['1', '2'])
    path = root / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['status'] = 'indexing'
    path.write_text(json.dumps(manifest))
    before = snapshot_tree(tmp_path)
    with pytest.raises(ValueError, match='completed'):
        repair_metadata.repair(root, execute=True, offline=True)
    assert snapshot_tree(tmp_path) == before


@pytest.mark.parametrize('fault', ['clean_claim', 'old_logic', 'triage_verified', 'recurrence_rows'])
def test_graph_readback_rejects_temporal_or_history_laundering(tmp_path, fault):
    from datetime import datetime
    from data_pipeline.s4_graph import build, graph as producer
    store = make_clinical_store(tmp_path)
    diagnoses = make_diagnoses(tmp_path)
    sample = dict(sample_id='sample', subject_id='1', stay_id='12', split='train',
                  cutoff=datetime.fromisoformat('2020-01-03T01:00:00'))
    try:
        graph = producer.build_graph(store, sample, [], {'pain'}, diagnoses)
        if fault == 'clean_claim':
            graph['temporal_clean'] = True
        elif fault == 'old_logic':
            del graph['logic_contract_version']
        elif fault == 'triage_verified':
            next(n for n in graph['nodes'] if n['kind'] == 'complaint')['availability_verified'] = True
        elif fault == 'recurrence_rows':
            next(e for e in graph['edges'] if e['relation'] == 'recurrence_of')['prior_encounters'] = 5
        (tmp_path / 'graphs.jsonl').write_text(json.dumps(graph) + '\n')
        with pytest.raises(ValueError):
            build.verify_artifact(tmp_path, [sample], store, {'pain'}, diagnoses)
    finally:
        store.close()


def test_metadata_cli_preview_apply_and_idempotency(tmp_path):
    import subprocess
    import sys
    root = event_artifact(tmp_path / 'artifact', ['1', '2'])
    command = [sys.executable, '-m', 'data_pipeline.s2_events.repair_metadata',
               '--artifact', str(root)]
    before = snapshot_tree(tmp_path)
    preview = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(preview.stdout)['status'] == 'dry_run'
    assert snapshot_tree(tmp_path) == before
    applied = subprocess.run(command + ['--execute', '--offline'], check=True, capture_output=True, text=True)
    assert json.loads(applied.stdout)['status'] == 'repaired'
    after = snapshot_tree(tmp_path)
    repeated = subprocess.run(command + ['--execute', '--offline'], check=True, capture_output=True, text=True)
    assert json.loads(repeated.stdout)['status'] == 'already_repaired'
    assert snapshot_tree(tmp_path) == after


def test_first_lab_resume_honors_metadata_lock(tmp_path, monkeypatch):
    from argparse import Namespace
    from data_pipeline.s2_events import first_lab
    from data_pipeline.s2_events.repair_metadata import artifact_lock
    root = event_artifact(tmp_path / 'artifact', ['1', '2'])
    called = []
    monkeypatch.setattr(first_lab, '_run', lambda args: called.append(args))
    with artifact_lock(root):
        with pytest.raises(FileExistsError):
            first_lab.run(Namespace(resume_prepared=True, output=root))
    assert not called
