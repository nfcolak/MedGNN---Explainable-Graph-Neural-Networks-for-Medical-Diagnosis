"""Build a separate event graph artifact; execution requires explicit input policy.

Default invocation prints the command contract only. It never scans raw data or
creates outputs without --execute. No legacy artifacts or splits are changed.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil

from . import SCHEMA_VERSION
from .graph import build_graph
from .ingest import EventStore
from .schema import read_cohort, read_knowledge, save_json, sha256


def build(args):
    if args.chunk_size < 1 or args.max_events < 1 or args.max_edges < 1:
        raise ValueError('Chunk/event/edge limits must be positive')
    samples = read_cohort(args.cohort)
    patient_visits = {}
    cohort_coverage = {}
    for sample in samples:
        patient_visits.setdefault(sample['subject_id'], set()).add(sample['stay_id'])
        fold = cohort_coverage.setdefault(sample['split'], {'samples': 0, 'subjects': set(), 'visits': set()})
        fold['samples'] += 1
        fold['subjects'].add(sample['subject_id'])
        fold['visits'].add(sample['stay_id'])
    cohort_policy = {
        'sampling_unit': 'visit_cutoff', 'multiple_visits_per_patient': True,
        'subject_deduplication': False, 'split_unit': 'subject',
        'patients': len(patient_visits),
        'patients_with_multiple_index_visits': sum(len(visits) > 1 for visits in patient_visits.values()),
        'index_visits': sum(len(visits) for visits in patient_visits.values()),
        'samples': len(samples),
        'by_fold': {name: {'samples': fold['samples'], 'patients': len(fold['subjects']),
                           'index_visits': len(fold['visits'])}
                    for name, fold in cohort_coverage.items()},
    }
    knowledge = read_knowledge(args.knowledge) if args.knowledge else []
    labels = json.loads(args.labels.read_text()) if args.labels else None
    if labels is not None and (not isinstance(labels, list) or not labels or
                               any(not isinstance(x, str) or not x for x in labels) or
                               len(set(labels)) != len(labels)):
        raise ValueError('Labels must be an ordered list of unique nonempty names')
    if any(sample['target'] is not None for sample in samples):
        if labels is None:
            raise ValueError('Labelled cohorts require an explicit ordered --labels JSON')
        if any(s['target'] is not None and not 0 <= s['target'] < len(labels) for s in samples):
            raise ValueError('Target outside explicit label order')
    sources = {'cohort': args.cohort, 'edstays': args.raw_root / 'edstays.csv',
               'labevents': args.raw_root / 'labevents.csv'}
    if args.timing_policy == 'charttime-proxy':
        sources.update({name: args.raw_root / (name + '.csv') for name in ('vitalsign', 'medrecon')})
    if args.knowledge:
        sources['knowledge'] = args.knowledge
    if args.labels:
        sources['labels'] = args.labels
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    # Bind raw sources and implementation, not only the compact output.
    code = {str(p): p for p in sorted(Path(__file__).parent.glob('*.py'))}
    from comparison.standardized.enriched_input_v1 import spec
    code[str(Path(spec.__file__))] = Path(spec.__file__)
    bindings = {key: {'path': str(path.resolve()), 'sha256': sha256(path)}
                for key, path in sources.items()}
    code_hashes = {key: sha256(path) for key, path in code.items()}
    manifest = {'schema_version': SCHEMA_VERSION, 'status': 'building',
                'input_mode': 'event-only' if args.event_only else 'event-history-knowledge',
                'decision_policy': args.decision_policy,
                'timing_policy': args.timing_policy, 'source_bindings': bindings,
                'source_code': code_hashes, 'labels': labels,
                'cohort_policy': cohort_policy,
                'canonical_benchmark_parity': False, 'test_evaluated': False,
                'temporal_clean': False, 'early_triage_eligible': False,
                'availability_rule': 'event_time <= cutoff AND available_time <= cutoff',
                'limits': {'max_events_per_graph': args.max_events, 'max_edges_per_graph': args.max_edges,
                           'overflow_policy': 'fail; never truncate'},
                'knowledge_provenance': 'caller-reviewed source metadata; not independently clinically validated',
                'limitations': [
                    'New explicit encounter/cutoff contract; no equivalence to legacy cohort is asserted.',
                    'storetime is a source availability proxy, not verified clinician visibility.',
                    'Visit start/end establish event lineage, not documentation availability.',
                    'Prediction-time and label-time clinical eligibility require independent review.',
                    'Untimestamped triage, target diagnosis, aggregate history and pyxis are excluded.',
                    'Medication reconciliation is not administration, dose or treatment response.',
                    'Lab units are preserved exactly, not silently harmonized.',
                    'Graph files and SQLite contain sensitive clinical data; local access only.',
                ]}
    if args.timing_policy == 'charttime-proxy':
        manifest['limitations'].append('Vitals and medication reconciliation use charttime as an unverified availability proxy.')
    previous_umask = os.umask(0o077)
    store = None
    created = False
    try:
        args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        created = True
        save_json(args.output / 'manifest.json', manifest)
        snapshot = args.output / 'source_snapshot'
        snapshot.mkdir(mode=0o700)
        for position, (name, path) in enumerate(code.items()):
            destination = snapshot / (str(position) + '_' + path.name)
            shutil.copyfile(path, destination)
            destination.chmod(0o600)
            if sha256(destination) != code_hashes[name]:
                raise ValueError('Source changed during code snapshot')
        save_json(snapshot / 'index.json', {str(i) + '_' + path.name: name
                                          for i, (name, path) in enumerate(code.items())})
        if args.knowledge:
            shutil.copyfile(args.knowledge, args.output / 'knowledge.csv')
        store = EventStore(args.output / 'events.sqlite')
        store.ingest_visits(sources['edstays'], samples)
        subjects = {sample['subject_id'] for sample in samples}
        store.ingest_labs(sources['labevents'], subjects, args.chunk_size)
        if args.timing_policy == 'charttime-proxy':
            store.ingest_chart_proxy(args.raw_root, subjects, args.chunk_size)
        totals = Counter()
        by_fold = {}
        graph_path = args.output / 'graphs.jsonl'
        with graph_path.open('x') as stream:
            for ordinal, sample in enumerate(samples):
                graph = build_graph(store, sample, knowledge, args.max_events, args.max_edges)
                stream.write(json.dumps(graph, sort_keys=True, allow_nan=False) + '\n')
                totals['graphs'] += 1
                totals['nodes'] += len(graph['nodes'])
                totals['edges'] += len(graph['edges'])
                totals['knowledge_edges'] += graph['coverage']['knowledge_edges']
                counts = by_fold.setdefault(sample['split'], Counter())
                counts['graphs'] += 1
                counts['event_empty'] += int(graph['coverage']['events'] == 0)
                counts['with_history'] += int(graph['coverage']['prior_visits'] > 0)
                counts['with_knowledge'] += int(graph['coverage']['knowledge_edges'] > 0)
                counts['events'] += graph['coverage']['events']
                if (ordinal + 1) % 1000 == 0:
                    print(json.dumps({'stage': 'graphs', 'written': ordinal + 1,
                                      'requested': len(samples)}), flush=True)
        if knowledge and not totals['knowledge_edges']:
            raise ValueError('Knowledge file matched no observed concepts; full-mode build rejected')
        for key, path in sources.items():
            if sha256(path) != bindings[key]['sha256']:
                raise ValueError('Input changed during construction')
        if any(sha256(path) != code_hashes[key] for key, path in code.items()):
            raise ValueError('Builder source changed during construction')
        manifest.update(status='completed', counts=dict(totals),
                        coverage={key: dict(value) for key, value in by_fold.items()},
                        ingest_counts=dict(store.counts), graphs_sha256=sha256(graph_path))
        store.close()
        store = None
        manifest['event_index_sha256'] = sha256(args.output / 'events.sqlite')
        save_json(args.output / 'manifest.json', manifest)
        return manifest
    except Exception as exc:
        if created:
            manifest.update(status='failed', error_type=type(exc).__name__)
            save_json(args.output / 'manifest.json', manifest)
        raise
    finally:
        if store is not None:
            store.close()
        os.umask(previous_umask)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--raw-root', type=Path, required=True)
    result.add_argument('--cohort', type=Path, required=True,
                        help='CSV: sample_id,subject_id,stay_id,cutoff,split[,target]')
    result.add_argument('--labels', type=Path, help='JSON ordered class names; required with targets')
    result.add_argument('--decision-policy', required=True,
                        help='Human-readable prediction task and why each cutoff is eligible')
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument('--knowledge', type=Path, help='Caller-reviewed provenance-bearing relation CSV')
    mode.add_argument('--event-only', action='store_true', help='Explicitly omit external knowledge; not the full three-layer input')
    result.add_argument('--timing-policy', choices=['strict-storetime', 'charttime-proxy'],
                        default='strict-storetime')
    result.add_argument('--chunk-size', type=int, default=100000)
    result.add_argument('--max-events', type=int, default=20000)
    result.add_argument('--max-edges', type=int, default=200000)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--execute', action='store_true')
    return result


def main():
    args = parser().parse_args()
    if not args.execute:
        print(json.dumps({'status': 'not_executed', 'schema_version': SCHEMA_VERSION,
                          'output': str(args.output), 'timing_policy': args.timing_policy,
                          'required_next_step': 'Review explicit cohort/cutoffs, knowledge provenance and labels; --execute scans raw files.'}, indent=2))
        return
    result = build(args)
    print(json.dumps({'status': result['status'], 'counts': result['counts'],
                      'test_evaluated': False}, indent=2))


if __name__ == '__main__':
    main()
