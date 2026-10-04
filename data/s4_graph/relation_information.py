"""Read-only, field-level reconstruction audit of the RAW artifact node view.

Compare directed endpoint AND payload multisets, never just relation flags. An
observed match establishes agreement with the current builder on the inspected
sample, not population conditional entropy, model utility, or feature parity.
The fitted GNN tensors and XGBoost summaries are different, unaudited views here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta
from itertools import combinations, islice
import json
import math
from pathlib import Path

from core import INFORMATIVE_RELATIONS, STRUCTURAL_RELATIONS

NODE_FIELDS = ('id', 'kind', 'token', 'scope', 'value', 'unit', 'time_hours',
               'prior_encounters')
NODE_VIEW = {
    'name': 'raw_artifact_nodes_v1',
    'available_fields': list(NODE_FIELDS),
    'description': 'Unfitted raw node records, including units, times and builder node IDs.',
    'ordering': 'Chronological measurements; numeric n<ID> breaks ties as in builder insertion order.',
    'excluded': ['edges', 'coverage', 'diagnosis encounter identities and end times',
                 'external knowledge table', 'targets'],
    'not_equivalent_to': ['fitted GNN tensors', 'XGBoost summarized feature matrix'],
}
EDGE_METADATA = {'source', 'target', 'relation', 'informative'}
TIME_FIELDS = ('delta', 'interval_hours', 'rate_per_hour', 'comparable_units')
PAYLOAD_FIELDS: dict[str, tuple[str, ...]] = {rel: () for rel in (*STRUCTURAL_RELATIONS, *INFORMATIVE_RELATIONS)}
PAYLOAD_FIELDS.update({
    'baseline_of': TIME_FIELDS, 'trajectory_of': TIME_FIELDS,
    'recurrence_of': ('prior_encounters', 'last_seen_hours'),
    **{r: ('provenance',) for r in INFORMATIVE_RELATIONS if r.startswith('medical:')},
})


def iter_graphs(path, limit=None):
    """Read only a requested JSONL prefix; no tensorizer or ML dependency."""
    if limit is not None and limit < 0:
        raise ValueError('limit must be nonnegative')
    with Path(path).open() as stream:
        for line in islice(stream, limit):
            yield json.loads(line)


def _freeze(value):
    """Hashable JSON values: preserve types/nulls and reject nonfinite numbers."""
    if isinstance(value, dict):
        return ('object', tuple(sorted((k, _freeze(v)) for k, v in value.items())))
    if isinstance(value, list):
        return ('array', tuple(_freeze(v) for v in value))
    if isinstance(value, bool):
        return ('bool', value)
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError('Nonfinite graph payload')
        return ('number', value)
    return (type(value).__name__, value)


def _multiset(edges, fields=()):
    return Counter((e['source'], e['target'],
                    tuple((f, _freeze(e[f]) if f in e else ('missing',)) for f in fields))
                   for e in edges)


def _compare(actual, predicted, fields=()):
    got, want = _multiset(actual, fields), _multiset(predicted, fields)
    return {'tested': True, 'exact': got == want,
            'unpredicted': sum((got - want).values()),
            'missing': sum((want - got).values())}


def _untested(reason):
    return {'tested': False, 'exact': None, 'reason': reason,
            'unpredicted': None, 'missing': None}


def _node_order(node):
    # IDs are an explicitly available part of this raw view, NOT learned features.
    return int(node['id'][1:])


def _predict(nodes, relation, knowledge):
    """Mirror graph.py topology; reuse its numeric _link rule, never real edges."""
    kinds = defaultdict(list)
    for node in nodes:
        kinds[node['kind']].append(node)
    for kind in ('patient', 'visit'):
        if len(kinds[kind]) != 1:
            raise ValueError('Expected one ' + kind + ' node')
    patient, visit = kinds['patient'][0]['id'], kinds['visit'][0]['id']
    out = []

    def emit(source, target, rel=relation, informative=None, **payload):
        out.append({'source': source, 'target': target, 'relation': rel, **payload})

    if relation == 'has_visit':
        emit(patient, visit)
    elif relation == 'index_visit_of':
        emit(visit, patient)
    elif relation in ('reports_complaint', 'observed_vital', 'has_prior_diagnosis'):
        kind = {'reports_complaint': 'complaint', 'observed_vital': 'vital',
                'has_prior_diagnosis': 'diagnosis'}[relation]
        for node in kinds[kind]:
            emit(visit, node['id'])
    elif relation == 'recurrence_of':
        for node in kinds['diagnosis']:
            emit(node['id'], visit, prior_encounters=node['prior_encounters'])
    elif relation == 'co_complaint':
        for left, right in combinations(sorted(kinds['complaint'], key=_node_order), 2):
            emit(left['id'], right['id'])
            emit(right['id'], left['id'])
    elif relation == 'measured_in':
        for node in kinds['measurement']:
            if node['scope'] == 'index':
                emit(visit, node['id'])
    elif relation == 'instance_of':
        analytes = {n['token']: n['id'] for n in kinds['analyte']}
        for node in kinds['measurement']:
            emit(node['id'], analytes[node['token']])
    elif relation in ('baseline_of', 'trajectory_of'):
        from data.s4_graph.graph import _link
        prior, index = defaultdict(list), defaultdict(list)
        for node in kinds['measurement']:
            if node['scope'] not in ('prior', 'index'):
                raise ValueError('Unknown measurement scope')
            (prior if node['scope'] == 'prior' else index)[node['token']].append(node)
        for token, history in prior.items():
            history = sorted(history, key=lambda n: (n['time_hours'], _node_order(n)))
            if relation == 'trajectory_of':
                pairs = zip(history, history[1:])
            elif token in index:
                first = min(index[token], key=lambda n: (n['time_hours'], _node_order(n)))
                pairs = [(history[-1], first)]
            else:
                pairs = []
            for source, target in pairs:
                def record(n):
                    # Builder timestamps have datetime/microsecond precision. Restore
                    # relative timestamps rather than duplicate delta/rate/unit rules.
                    time = datetime(2000, 1, 1) + timedelta(hours=n['time_hours'])
                    return {'time': time.isoformat(), 'value': n['value'], 'unit': n['unit']}
                _link(emit, source['id'], target['id'], relation, record(source), record(target))
    elif relation.startswith('medical:') and relation in PAYLOAD_FIELDS:
        if knowledge is None:
            return None
        analytes = {n['token']: n['id'] for n in kinds['analyte']}
        vitals = {n['token'].split(':', 1)[-1]: n['id'] for n in kinds['vital']}
        targets = {n['token']: n['id'] for n in kinds['knowledge']}
        for row in knowledge:
            if 'medical:' + row['relation'] != relation:
                continue
            source = analytes.get(row['source_token']) or vitals.get(row['source_token'].split(':', 1)[-1])
            if source is not None:
                # Missing knowledge nodes are a reconstruction failure, not a skipped edge.
                target = targets.get(row['target_token'], ('absent_knowledge_node', row['target_token']))
                emit(source, target, provenance={k: row[k] for k in
                                               ('source_uri', 'source_version', 'reviewed_by')})
    else:
        return None
    return out


def audit_graph(graph, knowledge=None):
    """Per-relation results. External-source agreement never becomes node-only success."""
    nodes = [{k: n[k] for k in NODE_FIELDS if k in n} for n in graph['nodes']]
    ids = [n['id'] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate node ID')
    actual = defaultdict(list)
    annotations = 0
    for edge in graph['edges']:
        actual[edge['relation']].append(edge)
        if edge['relation'] in PAYLOAD_FIELDS:
            annotations += edge.get('informative') != (edge['relation'] in INFORMATIVE_RELATIONS)
    relations = {}
    for rel in sorted(set(actual) | set(PAYLOAD_FIELDS)):
        got = actual[rel]
        fields = PAYLOAD_FIELDS.get(rel, ())
        external = rel.startswith('medical:')
        scope = 'raw_nodes_plus_external_knowledge' if external else 'raw_nodes'
        reason = None
        try:
            pred = _predict(nodes, rel, knowledge)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            pred = None
            reason = 'Insufficient node/source view or incompatible builder rule: ' + str(exc)
        if not got and pred == []:
            continue
        if not got and pred is None and reason is None:
            # Unavailable encounter/knowledge evidence cannot establish even an
            # empty edge multiset when its endpoint kinds are present.
            possible = ((rel == 'comorbid_with' and sum(n['kind'] == 'diagnosis' for n in nodes) > 1)
                        or (external and any(n['kind'] == 'knowledge' for n in nodes)))
            if not possible:
                continue
        if pred is None:
            reason = reason or {
                'comorbid_with': 'Encounter identities per diagnosis are absent from nodes; directed multiplicity needs that history.',
            }.get(rel, 'External knowledge rows and provenance required.' if external
                  else 'No audited reconstruction rule for this relation.')
        unknown = Counter(f for e in got for f in e if f not in EDGE_METADATA and f not in fields)
        topology = _compare(got, pred) if pred is not None else _untested(reason)
        payload = {}
        for field in sorted(set(fields) | set(unknown)):
            present = sum(field in e for e in got)
            if field in unknown:
                check = _untested('Unknown field; no reconstruction rule.')
            elif field == 'last_seen_hours' and rel == 'recurrence_of':
                check = _untested('Prior encounter end times are absent from nodes, including when the stored value is null.')
            elif pred is None:
                check = _untested(reason)
            else:
                check = _compare(got, pred, (field,))
            payload[field] = {**check, 'observed_values': present,
                              'absent_values': len(got) - present}
        tested_fields = [f for f, check in payload.items() if check['tested']]
        joint = _compare(got, pred, tested_fields) if pred is not None else _untested(reason)
        full = bool(topology['exact'] and joint['exact'] and not unknown
                    and all(c['tested'] and c['exact'] for c in payload.values()))
        if pred is None:
            status = 'not_tested'
        elif full:
            status = 'verified_with_external_evidence' if external else 'verified_raw_node_rule'
        elif not topology['exact'] or not joint['exact']:
            status = 'mismatch'
        else:
            status = 'partial'
        relations[rel] = {
            'observed_edges': len(got), 'predicted_edges': len(pred) if pred is not None else None,
            'evidence_scope': scope, 'status': status, 'topology': topology,
            'payload_fields': payload, 'joint_known_payload': joint,
            'unknown_fields': dict(unknown), 'full_exact': full,
            'node_only_full_exact': full and not external,
        }
    return {
        'raw_node_view': NODE_VIEW, 'relations': relations,
        'unknown_relations': {r: len(edges) for r, edges in actual.items() if r not in PAYLOAD_FIELDS},
        'annotation_mismatches': annotations,
        'fully_verified_from_raw_node_view': bool(relations) and all(r['node_only_full_exact'] for r in relations.values()),
    }


def summarize_audits(reports):
    """Accumulate explicit test denominators; unaudited fields never inflate success."""
    stats = {}
    unknown_relations, unknown_fields, unresolved_fields = Counter(), Counter(), Counter()
    count = verified = annotations = 0
    for report in reports:
        count += 1
        verified += report['fully_verified_from_raw_node_view']
        annotations += report['annotation_mismatches']
        unknown_relations.update(report['unknown_relations'])
        for rel, result in report['relations'].items():
            if rel not in stats:
                stats[rel] = {'graphs': 0, 'observed_edges': 0, 'topology_tested_graphs': 0,
                              'topology_exact_graphs': 0, 'full_exact_graphs': 0,
                              'node_only_full_exact_graphs': 0, 'statuses': Counter(),
                              'evidence_scope': result['evidence_scope'], 'fields': {}}
            s = stats[rel]
            s['graphs'] += 1
            s['observed_edges'] += result['observed_edges']
            s['topology_tested_graphs'] += result['topology']['tested']
            s['topology_exact_graphs'] += result['topology']['exact'] is True
            s['full_exact_graphs'] += result['full_exact']
            s['node_only_full_exact_graphs'] += result['node_only_full_exact']
            s['statuses'][result['status']] += 1
            unknown_fields.update({rel + '.' + f: n for f, n in result['unknown_fields'].items()})
            for field, check in result['payload_fields'].items():
                f = s['fields'].setdefault(field, {'observed_values': 0, 'absent_values': 0,
                                                  'tested_graphs': 0, 'exact_graphs': 0,
                                                  'untested_graphs': 0, 'reasons': Counter()})
                f['observed_values'] += check['observed_values']
                f['absent_values'] += check['absent_values']
                f['tested_graphs'] += check['tested']
                f['exact_graphs'] += check['exact'] is True
                f['untested_graphs'] += not check['tested']
                if not check['tested']:
                    f['reasons'][check['reason']] += 1
                    unresolved_fields[rel + '.' + field] += check['observed_values']
    return {
        'graphs': count, 'raw_node_view': NODE_VIEW, 'relations': stats,
        'graphs_fully_verified_from_raw_node_view': verified,
        'graphs_not_fully_verified': count - verified,
        'annotation_mismatches': annotations, 'unknown_relations': dict(unknown_relations),
        'unknown_field_values': dict(unknown_fields),
        'nonreconstructed_field_values': dict(unresolved_fields),
        'numeric_comparison': 'Exact finite numeric value equality (bool distinct); no rounding away errors. Null is distinct from missing.',
        'limitations': [
            'Sampled reconstruction is not a proof of population conditional entropy or model utility.',
            'Mismatch can indicate corruption, incomplete view, or builder/version drift; it is not proof of new information.',
            'Diagnosis recurrence recency and comorbidity encounter identities are not available in the declared node view.',
            'Knowledge-assisted matches require external rows/provenance and are not node-only reconstruction.',
            'No fitted GNN or XGBoost feature-view reconstruction was performed.',
            'Rules target the current graph builder, not a historical source snapshot; relative timestamps are restored at microsecond precision.',
        ],
    }


def load_knowledge(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path,
                        default=Path('comparison/standardized/event_inputs/clinical_graph_v3_full'))
    parser.add_argument('--knowledge', type=Path, default=None,
                        help='Optional explicit external evidence, NEVER counted as node-only.')
    parser.add_argument('--limit', type=int, default=3000)
    args = parser.parse_args()
    knowledge = load_knowledge(args.knowledge) if args.knowledge else None
    report = summarize_audits(audit_graph(g, knowledge) for g in
                             iter_graphs(args.artifact / 'graphs.jsonl', args.limit))
    report['requested_limit'] = args.limit
    report['external_knowledge_path'] = str(args.knowledge) if args.knowledge else None
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
