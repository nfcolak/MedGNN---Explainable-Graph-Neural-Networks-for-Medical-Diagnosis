"""Read-only graph-information audit, with explicit reconstruction evidence.

The relation audit uses unfitted RAW node records, not the model tensorizer or
XGBoost matrix. The two summary views below are diagnostic projections only.
A relation's `informative` annotation is not evidence of additional information.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from core import INFORMATIVE_RELATIONS, STRUCTURAL_RELATIONS
from data.s4_graph.relation_information import (NODE_FIELDS, audit_graph, iter_graphs,
                                   summarize_audits)


def tabular_view(graph, coarse=False):
    """Diagnostic presence/summary projection; NOT a fitted baseline input row."""
    analytes = sorted({n['token'] for n in graph['nodes'] if n['kind'] == 'analyte'})
    complaints = sorted(n['token'] for n in graph['nodes'] if n['kind'] == 'complaint')
    if coarse:
        vitals = sorted({n['token'] for n in graph['nodes'] if n['kind'] == 'vital'})
        return {'analytes': analytes, 'complaints': complaints, 'vitals': vitals}
    vitals = sorted((n['token'], n.get('value')) for n in graph['nodes'] if n['kind'] == 'vital')
    index_values = defaultdict(list)
    for n in graph['nodes']:
        if n['kind'] == 'measurement' and n.get('scope') == 'index' and n.get('value') is not None:
            index_values[n['token']].append(n['value'])
    summary = sorted((token, min(v), max(v), sum(v) / len(v))
                     for token, v in index_values.items())
    return {'analytes': analytes, 'complaints': complaints, 'vitals': vitals,
            'index_summary': summary, 'prior_visits': graph.get('coverage', {}).get('prior_visits')}


def informative_signature(graph):
    """Directed multiset of declared informative/unknown edges AND every payload.

    Use semantic raw-node endpoint descriptions, not arbitrary cross-graph IDs.
    This collision diagnostic does not claim a model sees these descriptions.
    """
    nodes = {n['id']: {k: n[k] for k in NODE_FIELDS if k in n and k != 'id'}
             for n in graph['nodes']}
    signature = []
    for edge in graph['edges']:
        if edge['relation'] in STRUCTURAL_RELATIONS:
            continue
        signature.append(json.dumps({
            'relation': edge['relation'],
            'source': nodes[edge['source']], 'target': nodes[edge['target']],
            'payload': {k: v for k, v in edge.items()
                        if k not in ('source', 'target', 'relation', 'informative')},
        }, sort_keys=True, allow_nan=False))
    return tuple(sorted(signature))  # retains duplicates, unlike the former set


def audit(path, limit=None):
    membership_payloads = defaultdict(set)
    coarse_payloads = defaultdict(set)
    relation_counts = Counter()
    delta_by_analyte = defaultdict(list)
    intervals = []

    def reports():
        for graph in iter_graphs(path, limit):
            payload = informative_signature(graph)
            membership_payloads[json.dumps(tabular_view(graph), sort_keys=True)].add(payload)
            coarse_payloads[json.dumps(tabular_view(graph, coarse=True), sort_keys=True)].add(payload)
            nodes = {n['id']: n for n in graph['nodes']}
            for edge in graph['edges']:
                relation_counts[edge['relation']] += 1
                if edge['relation'] in ('baseline_of', 'trajectory_of'):
                    if edge.get('delta') is not None:
                        delta_by_analyte[nodes[edge['source']]['token']].append(edge['delta'])
                    if edge.get('interval_hours') is not None:
                        intervals.append(edge['interval_hours'])
            yield audit_graph(graph)

    reconstruction = summarize_audits(reports())
    dispersed = sorted(((t, len(v), statistics.pstdev(v)) for t, v in delta_by_analyte.items()
                        if len(v) >= 20), key=lambda row: (-row[1], row[0]))[:10]
    return {
        'graphs': reconstruction['graphs'], 'requested_limit': limit,
        'relation_counts': dict(relation_counts),
        'informative_edges': sum(relation_counts[r] for r in INFORMATIVE_RELATIONS),
        'structural_edges': sum(relation_counts[r] for r in STRUCTURAL_RELATIONS),
        'undeclared_relations': sorted(reconstruction['unknown_relations']),
        'annotation_mismatches': reconstruction['annotation_mismatches'],
        'reconstruction': reconstruction,
        'membership_collisions': {
            'rich_view_distinct': len(membership_payloads),
            'rich_view_mapping_to_multiple_payloads': sum(len(v) > 1 for v in membership_payloads.values()),
            'coarse_view_distinct': len(coarse_payloads),
            'coarse_view_mapping_to_multiple_payloads': sum(len(v) > 1 for v in coarse_payloads.values()),
            'interpretation': (
                'Collisions establish differing directed edge/payload multisets under these diagnostic '
                'summary projections only. Neither projection is the fitted GNN/XGBoost view. '
                'No collision is not evidence of reconstruction; continuous rows may be unique.'),
        },
        'payload_dispersion_top_analytes': [
            {'analyte': t, 'n': n, 'delta_pstdev': round(sd, 4)} for t, n, sd in dispersed],
        'interval_hours': {
            'n': len(intervals),
            'median': statistics.median(intervals) if intervals else None,
            'min': min(intervals) if intervals else None,
            'max': max(intervals) if intervals else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.artifact / 'graphs.jsonl', args.limit)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(text)
    if args.out:
        # Audit must not overwrite historical evidence accidentally.
        with args.out.open('x') as stream:
            stream.write(text + '\n')


if __name__ == '__main__':
    main()
