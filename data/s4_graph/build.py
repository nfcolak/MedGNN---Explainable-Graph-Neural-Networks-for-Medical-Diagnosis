"""Produce the v2 clinical decision-point graph artifact.

Inherits cohort, folds, cutoffs and sample ids byte-identically from the v1 event
artifact, whose sha256 bindings are verified before any row is read. The expensive
18 GB lab scan is reused, never repeated.

Checkable cutoff, lineage and payload claims are re-derived by `verify_artifact`
from the source index after writing. Clinician-availability assumptions remain
explicitly unverified; passing proxy bounds does not establish temporal cleanliness.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil

from core import INFORMATIVE_RELATIONS, NODE_KINDS, SCHEMA_VERSION, STRUCTURAL_RELATIONS
from core.paths import REPO_ROOT
from data.s4_graph.diagnosis import DiagnosisIndex, load_icd_map
from data.s4_graph.graph import (LOGIC_CONTRACT_VERSION, assumed_timing,
                    build_graph_with_visit_membership, lab_availability,
                    temporal_contract)
from core.contracts import (VISIT_MEMBERSHIP_CONTRACT_VERSION, VISIT_MEMBERSHIP_FILENAME,
                        validate_artifact_manifest, validate_visit_membership_record,
                        verify_visit_membership_file)
from data.s3_labels import icd_mapping
from data.s3_labels.icd_mapping import ICD_MAPPING_POLICY
from core.schema import (fit_complaint_vocabulary, read_cohort, read_knowledge,
                     save_json, sha256, timestamp)
from data.s4_graph.store import ClinicalStore, build_triage_index, complaint_counts

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


def _source_visit_memberships(graph, sample, store, ordered_stays, diagnoses=None):
    """Re-derive each node's visit set from source encounters and event/diagnosis rows."""
    stay_to_ordinal = {stay: ordinal for ordinal, stay in enumerate(ordered_stays)}
    nodes = graph['nodes']
    node_index = {node['id']: index for index, node in enumerate(nodes)}
    expected = {index: set() for index in range(len(nodes))}
    index_ordinal = stay_to_ordinal[sample['stay_id']]
    prior_stays = ordered_stays[:-1]

    for index, node in enumerate(nodes):
        kind = node['kind']
        if kind == 'patient':
            continue
        if kind in ('visit', 'complaint', 'vital'):
            expected[index].add(index_ordinal)

    measurement_nodes = [node for node in nodes if node['kind'] == 'measurement']
    source_records = {}

    def remember_source(record):
        source = record['source']
        if source in source_records:
            raise ValueError('Conflicting source measurement provenance')
        source_records[source] = record

    for record in store.measurements(sample['stay_id'], sample['cutoff']):
        remember_source(record)
    prior_tokens = {node['token'] for node in measurement_nodes if node.get('scope') == 'prior'}
    for token in sorted(prior_tokens):
        for record in store.analyte_history(sample['subject_id'], prior_stays, token,
                                            sample['cutoff']):
            remember_source(record)

    for index, node in enumerate(nodes):
        if node['kind'] != 'measurement':
            continue
        scope = node.get('scope')
        record = source_records.get(node.get('source_record'))
        if record is None or record['token'] != node['token']:
            raise ValueError('Measurement source stay cannot be re-derived')
        if (record['value'], record['unit']) != (node.get('value'), node.get('unit')):
            raise ValueError('Measurement source payload conflicts with its provenance')
        if scope == 'index':
            if record['stay'] != sample['stay_id']:
                raise ValueError('Index measurement has conflicting source stay')
        elif scope == 'prior':
            if record['stay'] not in stay_to_ordinal or record['stay'] == sample['stay_id']:
                raise ValueError('Historical measurement has conflicting source stay')
        else:
            raise ValueError('Measurement has an unknown visit scope')
        expected[index].add(stay_to_ordinal[record['stay']])

    diagnosis_history = {}
    if diagnoses is not None:
        index_visit = store.visit(sample['stay_id'])
        history = diagnoses.prior_history(
            sample['stay_id'], store.prior_visits(sample['subject_id'], index_visit['start']))
        diagnosis_history = {record['token']: record for record in history}
    for index, node in enumerate(nodes):
        if node['kind'] != 'diagnosis':
            continue
        record = diagnosis_history.get(node['token'])
        if record is None:
            raise ValueError('Diagnosis provenance cannot be re-derived from prior encounters')
        for stay in record['occurrences']:
            if stay not in stay_to_ordinal or stay == sample['stay_id']:
                raise ValueError('Diagnosis occurrence is outside source visit lineage')
            expected[index].add(stay_to_ordinal[stay])

    analyte_indices = {node['token']: index for index, node in enumerate(nodes)
                       if node['kind'] == 'analyte'}
    for index, node in enumerate(nodes):
        if node['kind'] != 'measurement':
            continue
        analyte_index = analyte_indices.get(node['token'])
        if analyte_index is None:
            raise ValueError('Measurement has no analyte anchor')
        expected[analyte_index].update(expected[index])

    knowledge_indices = {node['id']: index for index, node in enumerate(nodes)
                         if node['kind'] == 'knowledge'}
    for edge_record in graph['edges']:
        if not edge_record['relation'].startswith('medical:'):
            continue
        target_index = knowledge_indices.get(edge_record['target'])
        source_index = node_index.get(edge_record['source'])
        if target_index is None or source_index is None:
            raise ValueError('Knowledge provenance edge has an invalid endpoint')
        if nodes[source_index]['kind'] not in ('analyte', 'vital'):
            raise ValueError('Knowledge node is attached to a non-clinical anchor')
        expected[target_index].update(expected[source_index])

    for index, node in enumerate(nodes):
        if node['kind'] != 'patient' and not expected[index]:
            raise ValueError('Visit-specific node has no source-derived membership')
    return expected


def _iter_graph_membership_lines(root):
    graph_path = root / 'graphs.jsonl'
    membership_path = root / VISIT_MEMBERSHIP_FILENAME
    if not membership_path.is_file():
        raise ValueError('Missing visit-membership sidecar; historical artifacts are not upgraded')
    with graph_path.open() as graph_stream, membership_path.open() as membership_stream:
        for graph_line, membership_line in zip_longest(graph_stream, membership_stream):
            if graph_line is None or membership_line is None:
                raise ValueError('Graph and visit-membership sample counts differ')
            yield graph_line, membership_line


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
    membership_rows = 0
    for row_index, (line, membership_line) in enumerate(_iter_graph_membership_lines(root), 1):
        graph = json.loads(line)
        membership_record = json.loads(membership_line)
        if graph.get('logic_contract_version') != LOGIC_CONTRACT_VERSION:
            raise ValueError('Graph logic contract mismatch; regenerate into a new artifact')
        if any(graph.get(k) != v for k, v in temporal_contract(diagnoses is not None).items()):
            raise ValueError('Unsupported graph temporal availability claim')
        sid = graph['sample_id']
        if (row_index > len(samples) or sid != samples[row_index - 1]['sample_id']):
            raise ValueError('Graph sample order differs from the inherited cohort')
        if sid in seen or sid not in expected:
            raise ValueError('Graph identity mismatch')
        if 'target' in graph:
            raise ValueError('Unlabelled artifact must not carry targets')
        seen.add(sid)
        sample = expected[sid]
        if graph['split'] != sample['split']:
            raise ValueError('Fold mismatch')
        index = store.visit(sample['stay_id'])
        if index is None or index['subject'] != sample['subject_id']:
            raise ValueError('Visit lineage mismatch')
        cutoff = sample['cutoff']
        prior_stays = store.prior_visits(sample['subject_id'], index['start'])
        ordered_stays = [*prior_stays, sample['stay_id']]
        if len(set(ordered_stays)) != len(ordered_stays):
            raise ValueError('Duplicate source stay in visit order')
        expected_ordinals = list(range(len(ordered_stays)))
        membership_by_node = validate_visit_membership_record(
            graph, membership_record, expected_ordinals)
        source_membership = _source_visit_memberships(
            graph, sample, store, ordered_stays, diagnoses)
        if membership_by_node != source_membership:
            raise ValueError('Visit-membership sidecar conflicts with source provenance')
        membership_rows += 1

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
            timing = {}
            if n['kind'] in ('complaint', 'vital'):
                timing = assumed_timing('edstays.csv:intime', 'encounter_intime_assumption')
            elif n['kind'] == 'diagnosis':
                timing = assumed_timing('edstays.csv:outtime', 'completed_prior_encounter_end_assumption')
            elif n['kind'] == 'measurement':
                timing = lab_availability()
                if n.get('timing_basis') != 'storetime':
                    raise ValueError('Measurement lacks storetime availability proxy')
            elif n['kind'] == 'visit':
                if n.get('acuity_timing') != assumed_timing('edstays.csv:intime', 'encounter_intime_assumption'):
                    raise ValueError('Acuity lacks unverified triage timing assumption')
            if any(n.get(k) != v for k, v in timing.items()):
                raise ValueError('Node lacks explicit temporal availability assumption/proxy')

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
            history = {r['token']: r for r in diagnoses.prior_history(sample['stay_id'], prior_stays)}
            allowed = set(history)
            if emitted_dx != allowed:
                raise ValueError('Diagnosis nodes differ from the recomputed prior history')
            for n in graph['nodes']:
                if n['kind'] == 'diagnosis':
                    count = len(set(history[n['token']]['occurrences']))
                    if n['prior_encounters'] != count:
                        raise ValueError('Diagnosis prior encounter count mismatch')
                    recurrence = [e for e in graph['edges']
                                  if e['relation'] == 'recurrence_of' and e['source'] == n['id']]
                    if len(recurrence) != 1 or recurrence[0]['prior_encounters'] != count:
                        raise ValueError('Recurrence prior encounter count mismatch')
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
    if membership_rows != len(samples):
        raise ValueError('Graph and visit-membership sample counts differ')
    if not totals['informative_edges']:
        raise ValueError('Artifact contains no informative edges; the graph would be redundant')
    return {'status': 'verified', 'counts': dict(totals),
            'logic_contract_version': LOGIC_CONTRACT_VERSION,
            'visit_membership_contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
            'visit_membership_file': VISIT_MEMBERSHIP_FILENAME,
            'visit_membership_rows': membership_rows,
            **temporal_contract(diagnoses is not None),
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
    from data.s2_events.repair_metadata import LOCK, PENDING, validate_index_ready
    if (inherited / PENDING).exists() or (inherited / LOCK).exists():
        raise ValueError('Inherited metadata repair is incomplete or locked')
    if v1_manifest['status'] == 'index_ready':
        validate_index_ready(inherited, v1_manifest)
    elif v1_manifest['status'] != 'completed' or not (inherited / 'graphs.jsonl').is_file():
        raise ValueError('Inherited artifact needs completed graphs or verified index_ready metadata')

    code = {str(p.resolve()): p.resolve() for p in
            (*(REPO_ROOT / 'data/s4_graph').glob('*.py'),
             REPO_ROOT / 'core/schema.py', REPO_ROOT / 'core/contracts.py')}
    shared_icd_code = Path(icd_mapping.__file__).resolve()
    code[str(shared_icd_code)] = shared_icd_code
    sources = {
        'inherited_manifest': inherited / 'manifest.json',
        'inherited_prepared_index': inherited / 'prepared_index.json',
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
        'logic_contract_version': LOGIC_CONTRACT_VERSION,
        'visit_membership_contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
        'visit_membership_file': VISIT_MEMBERSHIP_FILENAME,
        'icd_mapping_policy': ICD_MAPPING_POLICY,
        'status': 'binding',
        'input_mode': 'decision-point-clinical-graph',
        'inherits_from': str(inherited),
        'reference_class_order_only': v1_manifest['reference_class_order_only'],
        'cohort_sha256': v1_manifest['cohort_sha256'],
        'inherited_cohort_sha256': v1_manifest['cohort_sha256'],
        'inherited_event_index_sha256': v1_manifest['event_index_sha256'],
        'timing_policy': 'storetime-proxy-labs; unverified triage-at-intime and prior-diagnosis-at-completion assumptions',
        'labels': None,
        'target_status': 'absent; visit-level verified target attachment required separately',
        'canonical_benchmark_parity': False,
        **temporal_contract(args.with_diagnosis),
        'test_evaluated': False,
        'decision_policy': ('Inherited v1 cutoff, unchanged: first nonempty recorded lab result of the '
                            'encounter. Triage evidence is admitted under an unverified arrival-time '
                            'assumption. Checking intime <= cutoff verifies only proxy bounds.'),
        'structural_relations': list(STRUCTURAL_RELATIONS),
        'informative_relations': list(INFORMATIVE_RELATIONS),
        'diagnosis_tier': bool(args.with_diagnosis),
        'diagnosis_policy': ('Completed prior encounters only; the index encounter contributes no '
                             'diagnosis. diagnosis.csv has no timestamp, so eligibility is structural '
                             '(prior encounter finished before the index encounter began), not a '
                             'recorded diagnosis time. Availability at completion is unverified. '
                             'Every emitted node is re-derived and checked '
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

        # Necessary proxy bound, NOT proof of triage availability: the source
        # contains no independent triage timestamp to verify that stronger claim.
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
                                            'arrival_after_cutoff': 0,
                                            'actual_availability_verified': False,
                                            'policy': 'unverified_encounter_intime_assumption'}

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
        membership_path = root / VISIT_MEMBERSHIP_FILENAME
        vocabulary_set = set(vocabulary)
        with graph_path.open('x') as graph_stream, membership_path.open('x') as membership_stream:
            for i, sample in enumerate(samples, 1):
                graph, membership_record = build_graph_with_visit_membership(
                    store, sample, knowledge, vocabulary_set, diagnoses,
                    args.max_nodes, args.max_edges)
                graph_stream.write(json.dumps(graph, sort_keys=True, separators=(',', ':'),
                                              allow_nan=False) + '\n')
                membership_stream.write(json.dumps(
                    membership_record, sort_keys=True, separators=(',', ':'),
                    allow_nan=False) + '\n')
                if i % 2000 == 0:
                    graph_stream.flush()
                    membership_stream.flush()
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
                        visit_membership_sha256=sha256(membership_path),
                        visit_membership_rows=verification['visit_membership_rows'],
                        verification_sha256=sha256(root / 'verification.json'))
        validate_artifact_manifest(manifest)
        verify_visit_membership_file(root, manifest)
        checkpoint(root / 'manifest.json', manifest)
        progress(root, 'completed', counts=manifest['counts'],
                 graphs_sha256=manifest['graphs_sha256'],
                 visit_membership_sha256=manifest['visit_membership_sha256'])
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
                        help='Completed or metadata-repaired index_ready v1 artifact; supplies cohort, cutoffs and lab index')
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
