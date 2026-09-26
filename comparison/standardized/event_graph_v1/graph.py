"""Materialize event identity, temporal order, visit history and typed knowledge."""
from collections import Counter, defaultdict
from itertools import groupby

from . import SCHEMA_VERSION
from .schema import timestamp


def build_graph(store, sample, knowledge, max_events=20000, max_edges=200000):
    cutoff = sample['cutoff']
    index = store.db.execute('SELECT stay, subject, start, finish FROM visits WHERE stay=?',
                             (sample['stay_id'],)).fetchone()
    if index is None or index[1] != sample['subject_id']:
        raise ValueError('Index visit lineage mismatch')
    # An overlapping/unfinished earlier visit is not established prior history.
    visits = list(store.db.execute(
        'SELECT stay, subject, start, finish FROM visits '
        'WHERE subject=? AND start<? AND finish<? ORDER BY start, stay',
        (sample['subject_id'], index[2], index[2]))) + [index]
    nodes, edges, lookup, times = [], [], {}, {}

    def hours(time):
        return (timestamp(time) - cutoff).total_seconds() / 3600. if time else None

    def node(key, kind, token, value=None, unit=None, time=None, available=None, **metadata):
        if key not in lookup:
            node_id = 'n' + str(len(nodes))
            lookup[key] = node_id
            times[node_id] = hours(time)
            nodes.append({'id': node_id, 'kind': kind, 'token': token, 'value': value,
                          'unit': unit, 'time_hours': hours(time),
                          'available_hours': hours(available), **metadata})
        return lookup[key]

    def edge(source, target, relation, **metadata):
        if len(edges) >= max_edges:
            raise ValueError('Graph edge budget exceeded; no silent truncation')
        delta = None if times[source] is None or times[target] is None else times[target] - times[source]
        edges.append({'source': source, 'target': target, 'relation': relation,
                      'delta_hours': delta, **metadata})

    patient = node(('patient',), 'patient', 'patient')
    previous_visit = None
    previous_start = None
    event_count = 0
    concept_nodes = {}
    timing = Counter()
    for stay, subject, start, finish in visits:
        visit = node(('visit', stay), 'visit', 'visit:index' if stay == index[0] else 'visit:prior',
                     time=start, available=start)
        edge(patient, visit, 'has_visit')
        if stay == index[0]:
            edge(visit, patient, 'index_visit_of')
        if previous_visit is not None and previous_start < start:
            edge(previous_visit, visit, 'next_visit_by_start')
        previous_visit, previous_start = visit, start
        observed = []
        series = defaultdict(list)
        # SQL filters before graph size checks; no target used for eligibility.
        cursor = store.db.execute(
            'SELECT id, time, available, token, value, unit, source, timing_basis '
            'FROM events WHERE stay=? AND subject=? AND time<=? AND available<=? '
            'ORDER BY time, id', (stay, subject, cutoff.isoformat(), cutoff.isoformat()))
        for event_id, time, available, token, value, unit, source, basis in cursor:
            event_count += 1
            if event_count > max_events:
                raise ValueError('Graph event budget exceeded; no silent truncation')
            event = node(('event', event_id), 'event', token, value=value, unit=unit,
                         time=time, available=available, source_record=source,
                         timing_basis=basis,
                         event_semantics='medication_reconciliation' if token.startswith('med:') else 'measurement')
            concept = node(('concept', token), 'concept', token)
            concept_nodes[token] = concept
            edge(visit, event, 'contains_event')
            edge(event, concept, 'observes_concept')
            observed.append((time, event))
            series[(token, unit)].append((time, event))
            timing[basis] += 1

        def connect_groups(records, relation):
            # Ties are simultaneous, not arbitrary causal/sequential ordering.
            previous = None
            for _, entries in groupby(records, key=lambda entry: entry[0]):
                current = [entry[1] for entry in entries]
                if previous is not None:
                    for source in previous:
                        for target in current:
                            edge(source, target, relation)
                previous = current

        connect_groups(observed, 'next_event_time')
        for (token, unit), records in series.items():
            connect_groups(records, 'next_reconciliation_record' if token.startswith('med:')
                           else 'next_same_measurement')

    knowledge_count = 0
    for triple in knowledge:
        if triple['source_token'] not in concept_nodes:
            continue
        target = node(('knowledge', triple['target_token']), 'knowledge', triple['target_token'])
        edge(concept_nodes[triple['source_token']], target, 'medical:' + triple['relation'],
             provenance={key: triple[key] for key in ('source_uri', 'source_version', 'reviewed_by', 'evidence')
                         if key in triple})
        knowledge_count += 1
    graph = {'schema_version': SCHEMA_VERSION, 'sample_id': sample['sample_id'],
             'split': sample['split'], 'nodes': nodes, 'edges': edges,
             'coverage': {'events': event_count, 'prior_visits': len(visits) - 1,
                          'knowledge_edges': knowledge_count, 'timing_basis': dict(timing)},
             'limitations': ['Recorded event order is not causality.',
                             'No diagnosis history or timestamp-free triage is inferred.']}
    if sample['target'] is not None:
        graph['target'] = sample['target']
    return graph
