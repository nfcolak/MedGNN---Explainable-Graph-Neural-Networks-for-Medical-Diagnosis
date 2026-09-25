"""Unlabelled all-visit first-recorded-lab production, with reusable prepared index.

This command consumes the canonical SUBJECT population, never legacy visit labels.
Only numeric results with explicit source units become events. Any nonempty result
can establish the cutoff, even when it cannot be represented as a numeric event.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3

from . import SCHEMA_VERSION
from .graph import build_graph
from .ingest import EventStore
from .schema import identifier, numeric, read_cohort, read_knowledge, rows, save_json, sha256, timestamp

FOLDS = {0: 'train', 1: 'validation', 2: 'test'}
UNKNOWN_UNITS = {'', '?', 'unknown', 'unk', 'nan', 'none', 'null', 'n/a', 'na', 'not available'}


def checkpoint(path, data):
    temporary = path.with_suffix('.tmp')
    save_json(temporary, data)
    temporary.replace(path)


def progress(root, stage, **data):
    value = {'stage': stage, 'utc': datetime.utcnow().isoformat() + 'Z', **data}
    with (root / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')
    print(json.dumps(value, sort_keys=True), flush=True)


def prepare(root, raw, canonical, chunk_size):
    store = EventStore(root / 'events.sqlite')
    store.db.executescript('''
        CREATE TABLE first_results(stay TEXT PRIMARY KEY, available TEXT NOT NULL);
        CREATE TABLE lab_presence(stay TEXT PRIMARY KEY);
        CREATE TABLE excluded_visits(stay TEXT PRIMARY KEY, subject TEXT NOT NULL,
            raw_start TEXT, raw_finish TEXT, reason TEXT NOT NULL);
    ''')
    intervals = defaultdict(list)
    for row in rows(raw / 'edstays.csv', {'subject_id', 'stay_id', 'intime', 'outtime'}):
        if row['subject_id'] not in canonical:
            continue
        subject, stay = identifier(row['subject_id']), identifier(row['stay_id'])
        store.counts['visits:raw_canonical'] += 1
        try:
            start = timestamp(row['intime']).isoformat()
            finish = timestamp(row['outtime']).isoformat() if row['outtime'] else None
            reason = 'end_before_start' if finish and finish < start else None
        except ValueError:
            reason = 'invalid_visit_timestamp'
        if reason:
            store.db.execute('INSERT INTO excluded_visits VALUES(?,?,?,?,?)',
                             (stay, subject, row['intime'], row['outtime'], reason))
            store.counts['visits:excluded:' + reason] += 1
            continue
        store.db.execute('INSERT INTO visits VALUES(?,?,?,?)', (stay, subject, start, finish))
        intervals[subject].append((stay, start, finish))
        store.counts['visits'] += 1
        store.counts['visits:missing_finish'] += int(finish is None)
    store.db.commit()
    columns = ['labevent_id', 'subject_id', 'itemid', 'charttime', 'storetime', 'value', 'valuenum', 'valueuom']
    subjects = set(canonical)
    first = {}
    presence = set()
    for frame in store.chunks(raw / 'labevents.csv', columns, subjects, chunk_size):
        for row in frame.to_dict('records'):
            store.counts['labs:canonical_subject_rows'] += 1
            try:
                event_time = timestamp(row['charttime']).isoformat()
            except ValueError:
                store.counts['labs:invalid_or_missing_charttime'] += 1
                continue
            matches = [(stay, start, finish) for stay, start, finish in intervals.get(row['subject_id'], [])
                       if finish is not None and start <= event_time <= finish]
            for stay, _, _ in matches:
                presence.add(stay)
            if len(matches) != 1:
                store.counts['labs:ambiguous_visit' if matches else 'labs:no_matching_visit'] += 1
                continue
            stay, start, finish = matches[0]
            try:
                available = timestamp(row['storetime']).isoformat()
            except ValueError:
                store.counts['labs:invalid_or_missing_storetime'] += 1
                continue
            if available < event_time:
                store.counts['labs:store_before_chart'] += 1
                continue
            # A textual result establishes availability but never a categorical feature.
            if not row['value'].strip() and not row['valuenum'].strip():
                store.counts['labs:empty_result'] += 1
                continue
            if available <= finish:
                first[stay] = min(first.get(stay, available), available)
                store.counts['labs:cutoff_eligible_result_rows'] += 1
            else:
                store.counts['labs:available_after_visit_end'] += 1
            value = numeric(row['valuenum'])
            unit_ok = row['valueuom'].strip().casefold() not in UNKNOWN_UNITS
            if value is None:
                store.counts['labs:missing_non_numeric_or_nonfinite_value'] += 1
            if not unit_ok:
                store.counts['labs:missing_or_unknown_unit'] += 1
            if value is None or not unit_ok:
                continue
            lab_id, item_id = identifier(row['labevent_id']), identifier(row['itemid'])
            store.insert(('lab:' + lab_id, row['subject_id'], stay, event_time, available,
                          'lab:' + item_id, value, row['valueuom'],
                          'labevents.csv:' + str(row['_source_row']), 'storetime'))
        store.db.executemany('INSERT INTO first_results VALUES(?,?) ON CONFLICT(stay) DO UPDATE SET available=MIN(first_results.available,excluded.available)', first.items())
        store.db.executemany('INSERT OR IGNORE INTO lab_presence VALUES(?)', ((x,) for x in presence))
        store.db.commit()
        checkpoint(root / 'ingest_checkpoint.json', {'status': 'scanning', 'counts': dict(store.counts),
                                                    'note': 'Incomplete scan cannot resume; complete prepared index can.'})
        progress(root, 'lab_chunk_committed', **{'counts': dict(store.counts)})
        first.clear()
        presence.clear()
    checkpoint(root / 'ingest_checkpoint.json', {'status': 'scan_completed', 'counts': dict(store.counts)})
    store.close()


def cohort_from_index(root, canonical):
    db = sqlite3.connect(root / 'events.sqlite')
    with (root / 'cohort.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'subject_id', 'stay_id', 'cutoff', 'split'])
        writer.writeheader()
        for stay, subject, cutoff in db.execute(
                'SELECT v.stay,v.subject,f.available FROM visits v JOIN first_results f USING(stay) ORDER BY v.subject,v.start,v.stay'):
            sample_id = hashlib.sha256(('event-first-lab-v1:' + subject + ':' + stay).encode()).hexdigest()
            writer.writerow(dict(sample_id=sample_id, subject_id=subject, stay_id=stay,
                                 cutoff=cutoff, split=FOLDS[canonical[subject]]))
    exclusions = Counter()
    by_fold = {}
    for fold_id, name in FOLDS.items():
        by_fold[name] = Counter()
    for subject, stay, cutoff, present, finish in db.execute(
            'SELECT v.subject,v.stay,f.available,p.stay,v.finish FROM visits v '
            'LEFT JOIN first_results f USING(stay) LEFT JOIN lab_presence p USING(stay)'):
        counts = by_fold[FOLDS[canonical[subject]]]
        counts['raw_visits'] += 1
        reason = ('eligible' if cutoff else 'missing_visit_end' if finish is None else
                  'no_lab_charttime_in_window' if present is None else 'no_eligible_recorded_result_in_window')
        counts[reason] += 1
        exclusions[reason] += 1
    excluded_subjects = set()
    for subject, reason in db.execute('SELECT subject,reason FROM excluded_visits'):
        excluded_subjects.add(subject)
        by_fold[FOLDS[canonical[subject]]]['raw_visits'] += 1
        by_fold[FOLDS[canonical[subject]]][reason] += 1
        exclusions[reason] += 1
    db.close()
    samples = read_cohort(root / 'cohort.csv')
    patient_visits = defaultdict(set)
    folds = {}
    for s in samples:
        patient_visits[s['subject_id']].add(s['stay_id'])
        c = folds.setdefault(s['split'], {'samples': 0, 'subjects': set()})
        c['samples'] += 1
        c['subjects'].add(s['subject_id'])
    policy = {'sampling_unit': 'visit_cutoff', 'multiple_visits_per_patient': True,
              'subject_deduplication': False, 'split_unit': 'subject', 'samples': len(samples),
              'patients': len(patient_visits), 'index_visits': len(samples),
              'patients_with_multiple_index_visits': sum(len(v) > 1 for v in patient_visits.values()),
              'canonical_subjects': len(canonical), 'canonical_subjects_without_eligible_visit': len(set(canonical) - set(patient_visits)),
              'subjects_with_invalid_source_visits': len(excluded_subjects),
              'visit_eligibility': dict(exclusions),
              'visit_eligibility_by_fold': {k: dict(v) for k, v in by_fold.items()},
              'by_fold': {k: {'samples': v['samples'], 'patients': len(v['subjects']), 'index_visits': v['samples']}
                          for k, v in folds.items()}}
    return samples, policy


def verify_artifact(root, samples, canonical):
    """Read the entire emitted artifact and reconcile against the prepared source index."""
    db = sqlite3.connect('file:' + str(root / 'events.sqlite') + '?mode=ro', uri=True)
    expected = {s['sample_id']: s for s in samples}
    seen = set()
    folds = defaultdict(Counter)
    totals = Counter()
    for line in (root / 'graphs.jsonl').open():
        graph = json.loads(line)
        sid = graph['sample_id']
        if sid in seen or sid not in expected or 'target' in graph:
            raise ValueError('Graph identity/target mismatch')
        seen.add(sid)
        sample = expected[sid]
        if graph['split'] != FOLDS[canonical[sample['subject_id']]]:
            raise ValueError('Canonical subject fold mismatch')
        cutoff = sample['cutoff'].isoformat()
        index = db.execute('SELECT subject,start,finish FROM visits WHERE stay=?', (sample['stay_id'],)).fetchone()
        first = db.execute('SELECT available FROM first_results WHERE stay=?', (sample['stay_id'],)).fetchone()[0]
        if index[0] != sample['subject_id'] or not index[1] <= cutoff <= index[2] or first != cutoff:
            raise ValueError('Cutoff or visit lineage mismatch')
        prior = list(db.execute('SELECT stay,start,finish FROM visits WHERE subject=? AND start<? AND finish<? ORDER BY start,stay',
                               (sample['subject_id'], index[1], index[1])))
        visit_rows = prior + [(sample['stay_id'], index[1], index[2])]
        nodes = {n['id']: n for n in graph['nodes']}
        visits = [n for n in graph['nodes'] if n['kind'] == 'visit']
        if len(visits) != len(visit_rows):
            raise ValueError('History visit count mismatch')
        represented_events = {}
        for visit_node, (stay, start, finish) in zip(visits, visit_rows):
            expected_hours = (timestamp(start) - sample['cutoff']).total_seconds() / 3600
            if visit_node['time_hours'] != expected_hours or expected_hours > 0:
                raise ValueError('Future or incorrect visit')
            represented_events[visit_node['id']] = Counter(
                (source, token, value, unit, (timestamp(time) - sample['cutoff']).total_seconds() / 3600,
                 (timestamp(available) - sample['cutoff']).total_seconds() / 3600)
                for source, token, value, unit, time, available in db.execute(
                    'SELECT source,token,value,unit,time,available FROM events WHERE stay=? AND subject=? AND time<=? AND available<=?',
                    (stay, sample['subject_id'], cutoff, cutoff)))
        observed = defaultdict(Counter)
        for edge in graph['edges']:
            if edge['source'] not in nodes or edge['target'] not in nodes:
                raise ValueError('Dangling edge')
            if edge['relation'] == 'contains_event':
                n = nodes[edge['target']]
                observed[edge['source']][(n['source_record'], n['token'], n['value'], n['unit'], n['time_hours'], n['available_hours'])] += 1
        for visit_id, records in represented_events.items():
            if records != observed[visit_id]:
                raise ValueError('Emitted events differ from full cutoff-filtered index')
        events = [n for n in graph['nodes'] if n['kind'] == 'event']
        if any(n['time_hours'] > 0 or n['available_hours'] > 0 or n['timing_basis'] != 'storetime' for n in events):
            raise ValueError('Future or unverified-availability event')
        if (len(events) != sum(sum(c.values()) for c in represented_events.values()) or
                len(events) != graph['coverage']['events'] or graph['coverage']['prior_visits'] != len(prior)):
            raise ValueError('Coverage mismatch')
        c = folds[graph['split']]
        c['graphs'] += 1
        c['event_empty'] += int(not events)
        c['index_event_empty'] += int(not represented_events[visits[-1]['id']])
        c['with_history'] += int(bool(prior))
        c['with_knowledge'] += int(graph['coverage']['knowledge_edges'] > 0)
        c['events'] += len(events)
        totals.update({'graphs': 1, 'nodes': len(nodes), 'edges': len(graph['edges']),
                       'knowledge_edges': graph['coverage']['knowledge_edges']})
        if len(seen) % 1000 == 0:
            progress(root, 'artifact_readback', verified=len(seen), requested=len(samples))
    if seen != set(expected):
        raise ValueError('Missing samples')
    db.close()
    return {'status': 'verified', 'counts': dict(totals), 'coverage': {k: dict(v) for k, v in folds.items()},
            'cutoff_violations': 0, 'future_visit_violations': 0, 'subject_fold_violations': 0,
            'event_payload_mismatches': 0, 'targets_present': 0,
            'method': 'Full JSONL readback; per-visit event multiset equality against SQLite; canonical subject-fold and minimum cutoff reconciliation.'}


def run(args):
    os.umask(0o077)
    root = args.output.resolve()
    prepared = {}
    sources = {'canonical_split': args.canonical.resolve(), 'edstays': (args.raw_root / 'edstays.csv').resolve(),
               'labevents': (args.raw_root / 'labevents.csv').resolve(), 'knowledge': args.knowledge.resolve()}
    code = {str(p.resolve()): p.resolve() for p in Path(__file__).parent.glob('*.py')}
    from comparison.standardized.enriched_input_v1 import spec
    code[str(Path(spec.__file__).resolve())] = Path(spec.__file__).resolve()
    if args.resume_prepared:
        manifest = json.loads((root / 'manifest.json').read_text())
        prepared = json.loads((root / 'prepared_index.json').read_text())
        if prepared['status'] != 'prepared' or manifest['status'] == 'completed':
            raise ValueError('Resume requires an uncompleted prepared artifact')
        if sha256(root / 'events.sqlite') != prepared['event_index_sha256']:
            raise ValueError('Prepared index changed')
    else:
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
        manifest = {'schema_version': SCHEMA_VERSION, 'status': 'binding', 'input_mode': 'event-history-knowledge',
                    'lab_scope': args.lab_scope, 'timing_policy': 'strict-storetime', 'labels': None,
                    'target_status': 'absent; visit-level verified target attachment required separately',
                    'canonical_benchmark_parity': False, 'temporal_clean': False, 'early_triage_eligible': False,
                    'raw_to_model_train_only': False, 'test_evaluated': False,
                    'decision_policy': 'First nonempty recorded lab result uniquely associated by charttime to an ED visit; charttime<=storetime<=outtime and intime<=charttime. Include all representable results available by cutoff, ties inclusive.',
                    'limits': {'max_events_per_graph': args.max_events, 'max_edges_per_graph': args.max_edges, 'overflow_policy': 'fail; never truncate'},
                    'limitations': ['Unlabelled graphs; legacy tables have no verified visit-level target lineage.',
                                    'All numeric lab item IDs, finite valuenum, explicit exact source unit only; not all laboratory result types.',
                                    'Unit known means nonempty non-placeholder source unit; no clinical unit ontology validation or conversion.',
                                    'Nonempty textual/categorical or unknown-unit results may establish cutoff but never become invented features.',
                                    'Vitals, medrecon, triage, diagnoses and pyxis excluded; no unverified charttime-only availability.',
                                    'storetime is a database availability proxy, not proven clinician visibility.',
                                    'Knowledge seed is source-checked, not clinically reviewed; only matching concepts receive relations.',
                                    'Canonical subject population and folds preserved; raw visits expanded, no old graph parity.',
                                    'Intrinsically invalid source visit intervals excluded without repair; original times retained in SQLite excluded_visits.',
                                    'Sensitive local artifacts; identifiers only in cohort/index/provenance, not model covariates.']}
        checkpoint(root / 'manifest.json', manifest)
        manifest['source_bindings'] = {}
        for name, path in sources.items():
            progress(root, 'hash_source', source=name)
            manifest['source_bindings'][name] = {'path': str(path), 'sha256': sha256(path)}
        manifest['source_code'] = {key: sha256(path) for key, path in code.items()}
        snapshot = root / 'source_snapshot'
        snapshot.mkdir(mode=0o700)
        snapshot_index = {}
        for i, (key, path) in enumerate(sorted(code.items())):
            name = str(i) + '_' + path.name
            shutil.copyfile(path, snapshot / name)
            if sha256(snapshot / name) != manifest['source_code'][key]:
                raise ValueError('Code changed during snapshot')
            snapshot_index[name] = key
        save_json(snapshot / 'index.json', snapshot_index)
        shutil.copyfile(args.knowledge, root / 'knowledge.csv')
        shutil.copyfile(args.canonical, root / 'canonical_split.json')
    try:
        for name, path in sources.items():
            if args.resume_prepared and sha256(path) != manifest['source_bindings'][name]['sha256']:
                raise ValueError('Resume source binding changed')
        if any(sha256(path) != manifest['source_code'][key] for key, path in code.items()):
            raise ValueError('Source code binding changed')
        split = json.loads((root / 'canonical_split.json').read_text())
        canonical = split['fold']
        if any(identifier(k) != k or type(v) is not int or v not in FOLDS for k, v in canonical.items()):
            raise ValueError('Invalid canonical subject fold map')
        manifest['reference_class_order_only'] = split['classes']
        if not args.resume_prepared:
            manifest['status'] = 'indexing'
            checkpoint(root / 'manifest.json', manifest)
            prepare(root, args.raw_root, canonical, args.chunk_size)
            counts = json.loads((root / 'ingest_checkpoint.json').read_text())['counts']
            samples, policy = cohort_from_index(root, canonical)
            prepared = {'status': 'prepared', 'event_index_sha256': sha256(root / 'events.sqlite'),
                        'cohort_sha256': sha256(root / 'cohort.csv'), 'cohort_policy': policy, 'ingest_counts': counts}
            checkpoint(root / 'prepared_index.json', prepared)
        else:
            if sha256(root / 'cohort.csv') != prepared['cohort_sha256']:
                raise ValueError('Prepared cohort changed')
            samples = read_cohort(root / 'cohort.csv')
        manifest.update(status='materializing', cohort_policy=prepared['cohort_policy'],
                        ingest_counts=prepared['ingest_counts'], event_index_sha256=prepared['event_index_sha256'],
                        cohort_sha256=prepared['cohort_sha256'])
        checkpoint(root / 'manifest.json', manifest)
        progress(root, 'prepared', cohort_policy=prepared['cohort_policy'])
        store = EventStore.__new__(EventStore)
        store.db = sqlite3.connect('file:' + str(root / 'events.sqlite') + '?mode=ro', uri=True)
        knowledge = read_knowledge(root / 'knowledge.csv')
        graph_path = root / 'graphs.jsonl'
        if graph_path.exists():
            graph_path.rename(root / ('graphs.incomplete.' + datetime.utcnow().strftime('%Y%m%dT%H%M%S') + '.jsonl'))
        with graph_path.open('x') as stream:
            for i, sample in enumerate(samples, 1):
                graph = build_graph(store, sample, knowledge, args.max_events, args.max_edges)
                stream.write(json.dumps(graph, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
                if i % 1000 == 0:
                    stream.flush()
                    progress(root, 'graphs', written=i, requested=len(samples))
        store.close()
        manifest['status'] = 'verifying'
        checkpoint(root / 'manifest.json', manifest)
        verification = verify_artifact(root, samples, canonical)
        if not verification['counts']['knowledge_edges']:
            raise ValueError('Knowledge seed matched no graphs')
        for name, path in sources.items():
            progress(root, 'rehash_source', source=name)
            if sha256(path) != manifest['source_bindings'][name]['sha256']:
                raise ValueError('Raw source changed during production')
        if any(sha256(path) != manifest['source_code'][key] for key, path in code.items()):
            raise ValueError('Code changed during production')
        if sha256(root / 'events.sqlite') != prepared['event_index_sha256'] or sha256(root / 'cohort.csv') != prepared['cohort_sha256']:
            raise ValueError('Prepared artifact changed during materialization')
        save_json(root / 'verification.json', verification)
        manifest.update(status='completed', counts=verification['counts'], coverage=verification['coverage'],
                        graphs_sha256=sha256(graph_path), verification_sha256=sha256(root / 'verification.json'),
                        graphs_bytes=graph_path.stat().st_size)
        checkpoint(root / 'manifest.json', manifest)
        progress(root, 'completed', counts=manifest['counts'], graphs_sha256=manifest['graphs_sha256'])
    except Exception as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        checkpoint(root / 'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--knowledge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lab-scope', choices=['all-numeric-known-unit'], required=True)
    parser.add_argument('--chunk-size', type=int, default=100000)
    parser.add_argument('--max-events', type=int, default=1000000)
    parser.add_argument('--max-edges', type=int, default=20000000)
    parser.add_argument('--resume-prepared', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if min(args.chunk_size, args.max_events, args.max_edges) < 1:
        parser.error('Chunk and budget limits must be positive')
    if args.execute:
        run(args)
    else:
        print(json.dumps({'status': 'not_executed', 'requires': '--execute'}))


if __name__ == '__main__':
    main()
