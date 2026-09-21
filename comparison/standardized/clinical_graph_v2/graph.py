"""Materialize one decision-point graph per cohort visit.

Design rule, enforced by `audit.py` against the produced artifact: an edge earns its
place only if computing its `informative` payload requires looking at BOTH endpoints.
v1 failed this -- `contains_event` and `observes_concept` are membership facts that
a presence vector already encodes, and `next_event_time` orders nodes that each
carry their own timestamp. Those relations are kept here only where they are needed
to make the graph navigable, and are declared STRUCTURAL so no result may cite them.

The informative relations:

  baseline_of    prior-visit measurement -> index measurement of the SAME analyte,
                 carrying the signed delta, the elapsed interval and the rate. The
                 delta is a two-endpoint quantity; a fixed-width row holding "latest
                 creatinine" cannot express "creatinine rose 1.8 mg/dL in 11 days"
                 because the comparison partner differs per patient.
  trajectory_of  consecutive prior measurements of one analyte, same payload. Gives
                 message passing a real chain to walk instead of a star.
  co_complaint   complaint <-> complaint within one visit. Conjunction is the point:
                 "chest pain" with "dyspnea" is not the sum of the two presences.
  recurrence_of  prior diagnosis -> the visit, carrying how many prior encounters
                 recorded it and how long ago it was last seen. "COPD, 3 admissions,
                 last one 40 days ago" is not the same evidence as a history flag.
  comorbid_with  prior diagnosis <-> prior diagnosis recorded in the SAME encounter.
                 Co-occurring conditions constrain each other; their pair is not the
                 sum of two indicator columns.
  medical:*      concept -> external knowledge node, shared across patients.

Diagnoses come exclusively from encounters that FINISHED before the index encounter
began. The index encounter's own diagnoses are the label and are never admitted --
`diagnosis.DiagnosisIndex.prior_history` refuses the index stay, and `build.py`
re-verifies from the source after writing.

Everything is bounded by the inherited cutoff; no node may be dated after it.
"""
from collections import defaultdict
from itertools import combinations

from . import SCHEMA_VERSION
from .schema import timestamp


def build_graph(store, sample, knowledge, vocabulary, diagnoses=None,
                max_nodes=20000, max_edges=200000):
    """Build one graph for one (stay, cutoff) sample.

    `store` exposes the read-only v1 SQLite event index plus the v2 triage/arrival
    tables. `vocabulary` is the train-fitted complaint allowlist. `diagnoses` is an
    optional `DiagnosisIndex`; when absent the graph carries no diagnosis tier. No
    target is read or attached here: labels are bound in a separate, hashed sidecar.
    """
    cutoff = sample['cutoff']
    index = store.visit(sample['stay_id'])
    if index is None or index['subject'] != sample['subject_id']:
        raise ValueError('Index visit lineage mismatch')
    if not index['start'] <= cutoff.isoformat() <= (index['finish'] or index['start']):
        raise ValueError('Cutoff falls outside the index encounter')

    nodes, edges, lookup, times = [], [], {}, {}
    dropped = defaultdict(int)

    def hours(value):
        return (timestamp(value) - cutoff).total_seconds() / 3600. if value else None

    def node(key, kind, token, **payload):
        if key not in lookup:
            if len(nodes) >= max_nodes:
                raise ValueError('Graph node budget exceeded; no silent truncation')
            node_id = 'n' + str(len(nodes))
            lookup[key] = node_id
            time_hours = hours(payload.pop('time', None))
            available_hours = hours(payload.pop('available', None))
            if time_hours is not None and time_hours > 0:
                raise ValueError('Node dated after the prediction cutoff')
            if available_hours is not None and available_hours > 0:
                raise ValueError('Node available only after the prediction cutoff')
            times[node_id] = time_hours
            nodes.append({'id': node_id, 'kind': kind, 'token': token,
                          'time_hours': time_hours, 'available_hours': available_hours,
                          **payload})
        return lookup[key]

    def edge(source, target, relation, informative, **payload):
        if len(edges) >= max_edges:
            raise ValueError('Graph edge budget exceeded; no silent truncation')
        edges.append({'source': source, 'target': target, 'relation': relation,
                      'informative': informative, **payload})

    # ---------------------------------------------------------------- patient/visit
    arrival = store.arrival(sample['stay_id'])
    patient = node(('patient',), 'patient', 'patient',
                   age=arrival.get('age'), gender=arrival.get('gender'),
                   race=arrival.get('race'), arrival_transport=arrival.get('arrival_transport'))
    visit = node(('visit', sample['stay_id']), 'visit', 'visit:index',
                 time=index['start'], available=index['start'],
                 acuity=arrival.get('acuity'))
    edge(patient, visit, 'has_visit', False)
    edge(visit, patient, 'index_visit_of', False)

    # ------------------------------------------------------------------- complaints
    # Recorded at triage, i.e. at `intime`, which the producer verifies is <= cutoff
    # for every cohort visit. Out-of-vocabulary surface forms are counted, not bucketed.
    complaint_ids = []
    for token in store.complaints(sample['stay_id']):
        if token not in vocabulary:
            dropped['complaint_out_of_vocabulary'] += 1
            continue
        cid = node(('complaint', token), 'complaint', 'cc:' + token,
                   time=index['start'], available=index['start'])
        edge(visit, cid, 'reports_complaint', False)
        complaint_ids.append(cid)
    # Conjunction, not summation: this is the pairwise term a presence vector lacks.
    for left, right in combinations(sorted(complaint_ids, key=lambda i: int(i[1:])), 2):
        edge(left, right, 'co_complaint', True)
        edge(right, left, 'co_complaint', True)

    # ---------------------------------------------------------------- triage vitals
    for field, value, unit, source_unit in store.triage_vitals(sample['stay_id']):
        vid = node(('vital', field), 'vital', 'vital:' + field, value=value, unit=unit,
                   source_unit=source_unit, time=index['start'], available=index['start'])
        edge(visit, vid, 'observed_vital', False)

    # ------------------------------------------------- index-visit lab measurements
    # One node per result, keyed by the source labevent id so repeated measurements
    # of one analyte stay distinct rather than being collapsed into a summary.
    analyte_nodes = {}
    index_by_analyte = defaultdict(list)
    for record in store.measurements(sample['stay_id'], cutoff):
        mid = node(('measure', record['id']), 'measurement', record['token'],
                   value=record['value'], unit=record['unit'],
                   time=record['time'], available=record['available'],
                   source_record=record['source'], timing_basis=record['timing_basis'],
                   scope='index')
        analyte = analyte_nodes.get(record['token'])
        if analyte is None:
            analyte = node(('analyte', record['token']), 'analyte', record['token'])
            analyte_nodes[record['token']] = analyte
        edge(visit, mid, 'measured_in', False)
        edge(mid, analyte, 'instance_of', False)
        index_by_analyte[record['token']].append((record['time'], mid, record))

    # ------------------------------------------- prior-visit baselines (informative)
    # Only analytes that ALSO occur in the index visit are pulled in. This is the
    # whole point: an unmatched historical value is a feature column, while a matched
    # one defines a delta that needs both endpoints. v1 dumped every historical event
    # regardless of match, producing 18 GB and, as measured, no signal.
    prior_visits = store.prior_visits(sample['subject_id'], index['start'])
    baseline_links = 0
    for analyte_token, index_records in index_by_analyte.items():
        history = store.analyte_history(sample['subject_id'], prior_visits, analyte_token, cutoff)
        if not history:
            continue
        previous = None
        history_ids = []
        for record in history:
            hid = node(('measure', record['id']), 'measurement', record['token'],
                       value=record['value'], unit=record['unit'],
                       time=record['time'], available=record['available'],
                       source_record=record['source'], timing_basis=record['timing_basis'],
                       scope='prior')
            edge(hid, analyte_nodes[analyte_token], 'instance_of', False)
            history_ids.append((record, hid))
            if previous is not None:
                _link(edge, previous[1], hid, 'trajectory_of', previous[0], record)
            previous = (record, hid)
        earliest_index = min(index_records, key=lambda entry: entry[0])
        last_record, last_id = history_ids[-1]
        _link(edge, last_id, earliest_index[1], 'baseline_of', last_record, earliest_index[2])
        baseline_links += 1

    # ------------------------------------------ prior diagnosis tier (informative)
    # Strictly from encounters that finished before this one began. The index
    # encounter's diagnoses are the label; `prior_history` refuses the index stay.
    diagnosis_nodes = {}
    recurrence_links = 0
    comorbid_links = 0
    if diagnoses is not None:
        history = diagnoses.prior_history(sample['stay_id'], prior_visits)
        visit_end = {stay: store.visit(stay)['finish'] for stay in prior_visits}
        by_encounter = defaultdict(list)
        for record in history:
            did = node(('diagnosis', record['token']), 'diagnosis', record['token'],
                       source_version=record['source_version'],
                       source_code=record['source_code'],
                       approximate_mapping=record['approximate_mapping'],
                       prior_encounters=len(record['occurrences']))
            diagnosis_nodes[record['token']] = did
            edge(visit, did, 'has_prior_diagnosis', False)
            # Recency is the two-endpoint quantity: a condition coded three times,
            # most recently 40 days ago, is different evidence from a history flag.
            ends = [visit_end[stay] for stay in record['occurrences'] if visit_end.get(stay)]
            last_seen = max(ends) if ends else None
            age_hours = hours(last_seen) if last_seen else None
            edge(did, visit, 'recurrence_of', True,
                 prior_encounters=len(record['occurrences']),
                 last_seen_hours=age_hours)
            recurrence_links += 1
            for stay in record['occurrences']:
                by_encounter[stay].append(record['token'])
        # Conditions coded together in one past encounter constrain each other.
        for stay, tokens in by_encounter.items():
            for left, right in combinations(sorted(set(tokens)), 2):
                edge(diagnosis_nodes[left], diagnosis_nodes[right], 'comorbid_with', True)
                edge(diagnosis_nodes[right], diagnosis_nodes[left], 'comorbid_with', True)
                comorbid_links += 1

    # ------------------------------------------------------- external knowledge tier
    knowledge_edges = 0
    for triple in knowledge:
        anchor = analyte_nodes.get(triple['source_token']) or lookup.get(('vital', triple['source_token'].split(':', 1)[-1]))
        if anchor is None:
            continue
        target = node(('knowledge', triple['target_token']), 'knowledge', triple['target_token'])
        edge(anchor, target, 'medical:' + triple['relation'], True,
             provenance={k: triple[k] for k in ('source_uri', 'source_version', 'reviewed_by')})
        knowledge_edges += 1

    informative = sum(1 for e in edges if e['informative'])
    return {
        'schema_version': SCHEMA_VERSION,
        'sample_id': sample['sample_id'],
        'split': sample['split'],
        'nodes': nodes,
        'edges': edges,
        'coverage': {
            'complaints': len(complaint_ids),
            'index_measurements': sum(len(v) for v in index_by_analyte.values()),
            'prior_visits': len(prior_visits),
            'baseline_links': baseline_links,
            'prior_diagnoses': len(diagnosis_nodes),
            'recurrence_links': recurrence_links,
            'comorbid_links': comorbid_links,
            'knowledge_edges': knowledge_edges,
            'informative_edges': informative,
            'structural_edges': len(edges) - informative,
            'dropped': dict(dropped),
        },
        'limitations': [
            'Recorded order is not causality.',
            'Triage complaint text is a surface-form allowlist, not a clinical ontology.',
            'Triage fields are bound to encounter intime; the producer verifies intime <= cutoff.',
            'Laboratory availability uses storetime, a database proxy for clinician visibility.',
            'Diagnoses carry no source timestamp; eligibility is structural (prior encounter finished before this one began), not a recorded diagnosis time.',
            'ICD-9 codes are mapped to ICD-10 through the repository GEM table; approximate mappings are flagged, unmappable codes keep their ICD-9 namespace.',
            'Knowledge relations are source-checked, not clinically reviewed.',
        ],
    }


def _link(edge, source_id, target_id, relation, source_record, target_record):
    """Attach the two-endpoint payload that justifies the relation existing at all.

    delta / interval / rate are undefined for a single node: they are the reason a
    graph model can see something a per-node feature row cannot.
    """
    start = timestamp(source_record['time'])
    end = timestamp(target_record['time'])
    interval = (end - start).total_seconds() / 3600.
    if interval < 0:
        raise ValueError('Baseline/trajectory link must run forward in time')
    delta = None
    rate = None
    if source_record['value'] is not None and target_record['value'] is not None:
        if source_record['unit'] != target_record['unit']:
            # Refuse to compare across units rather than invent a conversion.
            delta = None
        else:
            delta = target_record['value'] - source_record['value']
            rate = delta / interval if interval > 0 else None
    edge(source_id, target_id, relation, True,
         delta=delta, interval_hours=interval, rate_per_hour=rate,
         comparable_units=source_record['unit'] == target_record['unit'])
