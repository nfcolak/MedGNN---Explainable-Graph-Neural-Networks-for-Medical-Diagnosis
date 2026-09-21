"""Produce the v2 clinical decision-point graph artifact.

Inherits cohort, folds, cutoffs and sample ids byte-identically from the v1 event
artifact, whose sha256 bindings are verified before any row is read. The expensive
18 GB lab scan is reused, never repeated.

Every claim this artifact makes is re-derived by `verify_artifact` from the source
index after writing, not trusted from the builder.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

from . import INFORMATIVE_RELATIONS, NODE_KINDS, SCHEMA_VERSION, STRUCTURAL_RELATIONS
from .diagnosis import DiagnosisIndex, load_icd_map
from .graph import build_graph
from .schema import (fit_complaint_vocabulary, read_cohort, read_knowledge,
                     save_json, sha256, timestamp)
from .store import ClinicalStore, build_triage_index, complaint_counts

FOLDS = ('train', 'validation', 'test')


def checkpoint(path, data):
    temporary = path.with_suffix('.tmp')
    save_json(temporary, data)
    temporary.replace(path)


def progress(root, stage, **data):
    value = {'stage': stage, 'utc': datetime.utcnow().isoformat() + 'Z', **data}
    with (root / 'progress.jsonl').open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')
    print(json.dumps(value, sort_keys=True), flush=True)


def verify_artifact(root, samples, store, vocabulary, diagnoses=None):
    """Full readback. Re-derives cutoff, lineage and payloads from the source index."""
    expected = {s['sample_id']: s for s in samples}
    seen = set()
    folds = defaultdict(Counter)
    totals = Counter()
    relations = Counter()
    vocabulary = set(vocabulary)
    label_leaks = 0
    checked_diagnoses = 0
    for line in (root / 'graphs.jsonl').open():
        graph = json.loads(line)
        sid = graph['sample_id']
        if sid in seen or sid not in expected:
            raise ValueError('Graph identity mismatch')
        if 'target' in graph:
            raise ValueError('Unlabelled artifact must not carry targets')
        seen.add(sid)
        sample = expected[sid]
        if graph['split'] != sample['split']:
            raise ValueError('Fold mismatch')
        index = store.visit(sample['stay_id'])
        if index['subject'] != sample['subject_id']:
            raise ValueError('Visit lineage mismatch')
        cutoff = sample['cutoff']

        nodes = {n['id']: n for n in graph['nodes']}
        if len(nodes) != len(graph['nodes']):
            raise ValueError('Duplicate node id')
        for n in graph['nodes']:
            if n['kind'] not in NODE_KINDS:
                raise ValueError('Unknown node kind: ' + n['kind'])
            for field in ('time_hours', 'available_hours'):
                if n.get(field) is not None and n[field] > 0:
                    raise ValueError('Node dated after the cutoff')
            if n['kind'] == 'complaint' and n['token'][3:] not in vocabulary:
                raise ValueError('Out-of-vocabulary complaint node emitted')

        # Index-visit measurements must equal the cutoff-filtered source multiset.
        source_index = Counter((r['source'], r['token'], r['value'], r['unit'])
                               for r in store.measurements(sample['stay_id'], cutoff))
        emitted_index = Counter((n['source_record'], n['token'], n['value'], n['unit'])
                                for n in graph['nodes']
                                if n['kind'] == 'measurement' and n['scope'] == 'index')
        if source_index != emitted_index:
            raise ValueError('Index measurements differ from the source index')

        # Complaints must equal the in-vocabulary triage surface forms.
        source_cc = [t for t in store.complaints(sample['stay_id']) if t in vocabulary]
        emitted_cc = sorted(n['token'][3:] for n in graph['nodes'] if n['kind'] == 'complaint')
        if sorted(source_cc) != emitted_cc:
            raise ValueError('Complaint nodes differ from the triage index')

        # LABEL LEAKAGE GATE. Diagnosis nodes must be re-derivable from completed
        # PRIOR encounters, and must contain nothing that is only knowable from the
        # index encounter. The index encounter's own diagnoses ARE the label, so any
        # token unique to it appearing here would make the benchmark a reconstruction.
        emitted_dx = {n['token'] for n in graph['nodes'] if n['kind'] == 'diagnosis'}
        if diagnoses is not None:
            index = store.visit(sample['stay_id'])
            prior_stays = store.prior_visits(sample['subject_id'], index['start'])
            allowed = {r['token'] for r in diagnoses.prior_history(sample['stay_id'], prior_stays)}
            if emitted_dx != allowed:
                raise ValueError('Diagnosis nodes differ from the recomputed prior history')
            index_only = diagnoses.index_tokens(sample['stay_id']) - allowed
            leaked = emitted_dx & index_only
            if leaked:
                label_leaks += len(leaked)
                raise ValueError('Index-encounter diagnosis leaked into the graph: '
                                 + ', '.join(sorted(leaked)))
            checked_diagnoses += len(emitted_dx)
        elif emitted_dx:
            raise ValueError('Diagnosis nodes emitted without a diagnosis index')

        informative = 0
        for e in graph['edges']:
            if e['source'] not in nodes or e['target'] not in nodes:
                raise ValueError('Dangling edge')
            relation = e['relation']
            known = relation in STRUCTURAL_RELATIONS or relation in INFORMATIVE_RELATIONS
            if not known:
                raise ValueError('Undeclared relation: ' + relation)
            if e['informative'] != (relation in INFORMATIVE_RELATIONS):
                raise ValueError('Relation/informative flag disagreement: ' + relation)
            informative += int(e['informative'])
            if relation in ('baseline_of', 'trajectory_of'):
                # The payload that justifies the edge must actually be present.
                if e['interval_hours'] is None or e['interval_hours'] < 0:
                    raise ValueError('Baseline/trajectory edge without a forward interval')
                src, dst = nodes[e['source']], nodes[e['target']]
                if src['token'] != dst['token']:
                    raise ValueError('Baseline/trajectory edge across different analytes')
                if e['comparable_units'] and src['value'] is not None and dst['value'] is not None:
                    if abs((dst['value'] - src['value']) - e['delta']) > 1e-9:
                        raise ValueError('Stored delta disagrees with endpoint values')
            relations[relation] += 1

        coverage = graph['coverage']
        if coverage['informative_edges'] != informative:
            raise ValueError('Informative edge count mismatch')
        if coverage['index_measurements'] != sum(emitted_index.values()):
            raise ValueError('Coverage/measurement mismatch')

        c = folds[graph['split']]
        c['graphs'] += 1
        c['with_complaint'] += int(coverage['complaints'] > 0)
        c['with_baseline'] += int(coverage['baseline_links'] > 0)
        c['with_prior_diagnosis'] += int(coverage.get('prior_diagnoses', 0) > 0)
        c['with_knowledge'] += int(coverage['knowledge_edges'] > 0)
        c['with_vitals'] += int(any(n['kind'] == 'vital' for n in graph['nodes']))
        c['index_measurement_empty'] += int(coverage['index_measurements'] == 0)
        c['informative_edges'] += informative
        totals.update({'graphs': 1, 'nodes': len(nodes), 'edges': len(graph['edges']),
                       'informative_edges': informative,
                       'structural_edges': len(graph['edges']) - informative,
                       'diagnosis_nodes': len(emitted_dx)})
        if len(seen) % 2000 == 0:
            progress(root, 'artifact_readback', verified=len(seen), requested=len(samples))
    if seen != set(expected):
        raise ValueError('Missing samples')
    if not totals['informative_edges']:
        raise ValueError('Artifact contains no informative edges; the graph would be redundant')
    return {'status': 'verified', 'counts': dict(totals),
            'coverage': {k: dict(v) for k, v in folds.items()},
            'relation_counts': dict(relations),
            'cutoff_violations': 0, 'lineage_violations': 0, 'targets_present': 0,
            'index_diagnosis_leaks': label_leaks,
            'diagnosis_nodes_reverified': checked_diagnoses,
            'method': ('Full JSONL readback; per-visit measurement and complaint multiset equality '
                       'against the source indices; every diagnosis node re-derived from completed '
                       'prior encounters and checked against index-encounter-only codes; '
                       'endpoint-recomputed deltas; declared-relation and node-kind closure; '
                       'cutoff bounds on every node.')}


def run(args):
    os.umask(0o077)
    root = args.output.resolve()
    inherited = args.inherit_from.resolve()
    v1_manifest = json.loads((inherited / 'manifest.json').read_text())
    if v1_manifest['status'] != 'completed':
        raise ValueError('Inherited artifact is not completed')

    code = {str(p.resolve()): p.resolve() for p in Path(__file__).parent.glob('*.py')}
    sources = {
        'inherited_cohort': inherited / 'cohort.csv',
        'inherited_event_index': inherited / 'events.sqlite',
        'edstays': args.raw_root / 'edstays.csv',
        'triage': args.raw_root / 'triage.csv',
        'patients': args.raw_root / 'patients.csv',
        'knowledge': args.knowledge.resolve(),
        'canonical_split': args.canonical.resolve(),
    }
    if args.with_diagnosis:
        sources['diagnosis'] = args.raw_root / 'diagnosis.csv'
        sources['icd_map'] = args.raw_root / 'icd9_to_icd10_mapping.csv'

    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'status': 'binding',
        'input_mode': 'decision-point-clinical-graph',
        'inherits_from': str(inherited),
        'inherited_cohort_sha256': v1_manifest['cohort_sha256'],
        'inherited_event_index_sha256': v1_manifest['event_index_sha256'],
        'timing_policy': 'strict-storetime-labs; triage bound to encounter intime',
        'labels': None,
        'target_status': 'absent; visit-level verified target attachment required separately',
        'canonical_benchmark_parity': False,
        'temporal_clean': True,
        'early_triage_eligible': True,
        'test_evaluated': False,
        'decision_policy': ('Inherited v1 cutoff, unchanged: first nonempty recorded lab result of the '
                            'encounter. Triage/arrival evidence is admitted because the producer verifies '
                            'encounter intime <= cutoff for every cohort visit.'),
        'structural_relations': list(STRUCTURAL_RELATIONS),
        'informative_relations': list(INFORMATIVE_RELATIONS),
        'diagnosis_tier': bool(args.with_diagnosis),
        'diagnosis_policy': ('Completed prior encounters only; the index encounter contributes no '
                             'diagnosis. diagnosis.csv has no timestamp, so eligibility is structural '
                             '(prior encounter finished before the index encounter began), not a '
                             'recorded diagnosis time. Every emitted node is re-derived and checked '
                             'against index-encounter-only codes during verification.')
                            if args.with_diagnosis else 'absent',
        'limits': {'max_nodes_per_graph': args.max_nodes, 'max_edges_per_graph': args.max_edges,
                   'overflow_policy': 'fail; never truncate'},
        'limitations': [
            'Complaint vocabulary is a train-fold frequency allowlist of raw surface forms, not a clinical ontology; no synonym merging is performed.',
            'Out-of-vocabulary complaints are dropped and counted, not bucketed into an "other" node.',
            'Triage vitals carry no independent recording timestamp; they are attributed to encounter intime.',
            'storetime is a database availability proxy, not proven clinician visibility.',
            'Historical context is restricted to analytes that recur in the index visit; unmatched history is deliberately absent.',
            'Knowledge relations are source-checked, not clinically reviewed.',
            'anchor_age is a demographic anchor, not the exact age at this encounter.',
            'Unlabelled artifact; targets are bound separately by exact sample_id lineage.',
        ],
    }
    checkpoint(root / 'manifest.json', manifest)

    manifest['source_bindings'] = {}
    for name, path in sources.items():
        progress(root, 'hash_source', source=name)
        manifest['source_bindings'][name] = {'path': str(path), 'sha256': sha256(path)}
    if manifest['source_bindings']['inherited_event_index']['sha256'] != v1_manifest['event_index_sha256']:
        raise ValueError('Inherited event index does not match its manifest')
    if manifest['source_bindings']['inherited_cohort']['sha256'] != v1_manifest['cohort_sha256']:
        raise ValueError('Inherited cohort does not match its manifest')
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
    shutil.copyfile(sources['knowledge'], root / 'knowledge.csv')
    shutil.copyfile(sources['inherited_cohort'], root / 'cohort.csv')

    try:
        samples = read_cohort(root / 'cohort.csv')
        if args.limit is not None:
            samples = samples[:args.limit]
            manifest['scope'] = 'bounded_smoke_not_benchmark'
            manifest['limit'] = args.limit
        stays = {s['stay_id'] for s in samples}

        manifest['status'] = 'indexing_triage'
        checkpoint(root / 'manifest.json', manifest)
        triage_index = root / 'triage.sqlite'
        manifest['triage_counts'] = build_triage_index(triage_index, args.raw_root, stays)
        manifest['triage_index_sha256'] = sha256(triage_index)
        progress(root, 'triage_indexed', counts=manifest['triage_counts'])

        store = ClinicalStore(sources['inherited_event_index'], triage_index)

        # Hard temporal gate: triage evidence is only admissible if arrival precedes
        # the decision point for EVERY sample. Fail closed, never per-sample silently.
        late = 0
        for s in samples:
            index = store.visit(s['stay_id'])
            if index is None or index['subject'] != s['subject_id']:
                raise ValueError('Cohort stay lineage absent from the inherited index')
            if timestamp(index['start']) > s['cutoff']:
                late += 1
        if late:
            raise ValueError(f'{late} encounters begin after their cutoff; triage evidence is inadmissible')
        manifest['triage_admissibility'] = {'samples_checked': len(samples),
                                            'arrival_after_cutoff': 0}

        train_stays = {s['stay_id'] for s in samples if s['split'] == 'train'}
        counts = complaint_counts(triage_index, train_stays)
        vocabulary = fit_complaint_vocabulary(counts, args.complaint_min_count)
        manifest['complaint_vocabulary'] = {
            'fit_scope': 'train fold visits only', 'min_count': args.complaint_min_count,
            'size': len(vocabulary), 'distinct_train_surface_forms': len(counts),
            'dropped_surface_forms': len(counts) - len(vocabulary),
            'sha256': sha256_of_list(vocabulary)}
        save_json(root / 'complaint_vocabulary.json',
                  {'tokens': vocabulary, 'min_count': args.complaint_min_count,
                   'fit_scope': 'train fold visits only'})
        progress(root, 'vocabulary_fitted', **manifest['complaint_vocabulary'])

        knowledge = read_knowledge(root / 'knowledge.csv')

        diagnoses = None
        if args.with_diagnosis:
            # Every cohort stay plus its prior encounters must be indexed, because
            # verification re-derives the index encounter's own codes to prove none
            # of them reached a graph.
            needed = set(stays)
            for s in samples:
                index = store.visit(s['stay_id'])
                needed.update(store.prior_visits(s['subject_id'], index['start']))
            mapping, approximate = load_icd_map(sources['icd_map'])
            diagnoses = DiagnosisIndex(sources['diagnosis'], needed, mapping, approximate)
            manifest['diagnosis_counts'] = {
                **dict(diagnoses.counts),
                'indexed_stays': len(needed),
                'icd9_map_entries': len(mapping),
                'icd9_approximate_entries': len(approximate),
            }
            progress(root, 'diagnosis_indexed', **manifest['diagnosis_counts'])

        manifest['status'] = 'materializing'
        checkpoint(root / 'manifest.json', manifest)
        graph_path = root / 'graphs.jsonl'
        vocabulary_set = set(vocabulary)
        with graph_path.open('x') as stream:
            for i, sample in enumerate(samples, 1):
                graph = build_graph(store, sample, knowledge, vocabulary_set, diagnoses,
                                    args.max_nodes, args.max_edges)
                stream.write(json.dumps(graph, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False) + '\n')
                if i % 2000 == 0:
                    stream.flush()
                    progress(root, 'graphs', written=i, requested=len(samples))

        manifest['status'] = 'verifying'
        checkpoint(root / 'manifest.json', manifest)
        verification = verify_artifact(root, samples, store, vocabulary, diagnoses)
        store.close()

        for name, path in sources.items():
            if sha256(path) != manifest['source_bindings'][name]['sha256']:
                raise ValueError('Source changed during production: ' + name)
        if any(sha256(path) != manifest['source_code'][key] for key, path in code.items()):
            raise ValueError('Code changed during production')

        save_json(root / 'verification.json', verification)
        manifest.update(status='completed', counts=verification['counts'],
                        coverage=verification['coverage'],
                        relation_counts=verification['relation_counts'],
                        graphs_sha256=sha256(graph_path),
                        graphs_bytes=graph_path.stat().st_size,
                        verification_sha256=sha256(root / 'verification.json'))
        checkpoint(root / 'manifest.json', manifest)
        progress(root, 'completed', counts=manifest['counts'],
                 graphs_sha256=manifest['graphs_sha256'])
    except Exception as exc:
        manifest.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        checkpoint(root / 'manifest.json', manifest)
        raise


def sha256_of_list(values):
    import hashlib
    payload = json.dumps(values, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inherit-from', type=Path, required=True,
                        help='Completed v1 event artifact supplying cohort, cutoffs and the lab index')
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--knowledge', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--complaint-min-count', type=int, default=50)
    parser.add_argument('--with-diagnosis', action='store_true',
                        help='Add the prior-encounter diagnosis tier (completed prior visits only; '
                             'the index encounter contributes nothing)')
    parser.add_argument('--max-nodes', type=int, default=20000)
    parser.add_argument('--max-edges', type=int, default=200000)
    parser.add_argument('--limit', type=int, default=None,
                        help='Bounded wiring smoke only; marks the artifact as non-benchmark')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if min(args.complaint_min_count, args.max_nodes, args.max_edges) < 1:
        parser.error('Budgets must be positive')
    if args.execute:
        run(args)
    else:
        print(json.dumps({'status': 'not_executed', 'requires': '--execute'}))


if __name__ == '__main__':
    main()
