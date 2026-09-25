"""Artifact audit: does this graph carry information a feature row cannot?

This is the check that condemned v1 (docs/new-input-diagnosis.md): if every graph
can be rebuilt from a fixed-width presence/summary vector, message passing only
redelivers what the features already hold, and the topology is decoration.

The audit is read-only and makes no model claim. It reports three things:

1. Reconstruction test -- rebuild each graph's topology from the tabular view a
   GBDT baseline would receive (which analytes/complaints/vitals are present).
   v1 passed this trivially: node membership determined every edge. An artifact
   that FAILS reconstruction is one where the graph holds something extra.
2. Payload dispersion -- for each informative relation, how much its payload varies
   across graphs that share identical node membership. Zero dispersion means the
   payload is a function of membership, i.e. still tabular.
3. Structural/informative accounting, so no result can cite a bookkeeping edge.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from . import INFORMATIVE_RELATIONS, STRUCTURAL_RELATIONS


def tabular_view(graph, coarse=False):
    """The feature row a tabular baseline would get.

    Two views are audited because they answer different questions:

    `coarse=True`  mirrors what the v1-style baseline actually consumed: which
                   analytes/complaints/vitals are present, nothing continuous. Two
                   graphs sharing a coarse view but differing in informative payload
                   are direct proof that the payload is not a function of presence.
    `coarse=False` is deliberately generous -- it adds per-analyte min/max/mean and
                   vital values. Collisions are rare here simply because continuous
                   summaries are near-unique, so this view tests reconstruction
                   rather than collision.
    """
    analytes = sorted({n['token'] for n in graph['nodes'] if n['kind'] == 'analyte'})
    complaints = sorted(n['token'] for n in graph['nodes'] if n['kind'] == 'complaint')
    if coarse:
        vitals = sorted({n['token'] for n in graph['nodes'] if n['kind'] == 'vital'})
        return {'analytes': analytes, 'complaints': complaints, 'vitals': vitals}
    vitals = sorted((n['token'], n.get('value')) for n in graph['nodes'] if n['kind'] == 'vital')
    index_values = defaultdict(list)
    for n in graph['nodes']:
        if n['kind'] == 'measurement' and n['scope'] == 'index' and n['value'] is not None:
            index_values[n['token']].append(n['value'])
    summary = sorted((token, min(v), max(v), round(sum(v) / len(v), 9))
                     for token, v in index_values.items())
    return {'analytes': analytes, 'complaints': complaints, 'vitals': vitals,
            'index_summary': summary, 'prior_visits': graph['coverage']['prior_visits']}


def reconstruct_from_tabular(view):
    """Best possible topology reconstruction from the tabular view alone.

    A star: the visit connects to everything present. This is exactly what v1's
    topology amounted to, which is why v1 reconstructed losslessly.
    """
    edges = set()
    for token in view['analytes']:
        edges.add(('visit', 'analyte', token))
    for token in view['complaints']:
        edges.add(('visit', 'complaint', token))
    for token, _ in view['vitals']:
        edges.add(('visit', 'vital', token))
    return edges


def informative_signature(graph):
    """The informative content a reconstruction would have to reproduce."""
    nodes = {n['id']: n for n in graph['nodes']}
    signature = set()
    for e in graph['edges']:
        if not e['informative']:
            continue
        src, dst = nodes[e['source']], nodes[e['target']]
        if e['relation'] in ('baseline_of', 'trajectory_of'):
            signature.add((e['relation'], src['token'],
                           None if e['delta'] is None else round(e['delta'], 9),
                           round(e['interval_hours'], 6)))
        elif e['relation'] == 'co_complaint':
            signature.add((e['relation'], src['token'], dst['token']))
        else:
            signature.add((e['relation'], src['token'], dst['token']))
    return signature


def audit(path, limit=None):
    graphs = 0
    membership_payloads = defaultdict(set)
    coarse_payloads = defaultdict(set)
    relation_counts = Counter()
    informative_total = 0
    structural_total = 0
    reconstructible = 0
    extra_facts = []
    delta_by_analyte = defaultdict(list)
    interval_values = []

    with Path(path).open() as stream:
        for line in stream:
            if limit is not None and graphs >= limit:
                break
            graph = json.loads(line)
            graphs += 1
            view = tabular_view(graph)
            signature = informative_signature(graph)
            payload = json.dumps(sorted(repr(item) for item in signature), sort_keys=True)
            membership_payloads[json.dumps(view, sort_keys=True)].add(payload)
            coarse_payloads[json.dumps(tabular_view(graph, coarse=True), sort_keys=True)].add(payload)


            # A graph is "reconstructible" only if its informative content is empty,
            # because the star rebuild can never produce a delta or a pair term.
            if not signature:
                reconstructible += 1
            else:
                extra_facts.append(len(signature))

            for e in graph['edges']:
                relation_counts[e['relation']] += 1
                informative_total += int(e['informative'])
                structural_total += int(not e['informative'])
                if e['relation'] in ('baseline_of', 'trajectory_of'):
                    if e.get('delta') is not None:
                        nodes = {n['id']: n for n in graph['nodes']}
                        delta_by_analyte[nodes[e['source']]['token']].append(e['delta'])
                    if e.get('interval_hours') is not None:
                        interval_values.append(e['interval_hours'])

    collisions = {k: v for k, v in membership_payloads.items() if len(v) > 1}
    coarse_collisions = {k: v for k, v in coarse_payloads.items() if len(v) > 1}

    dispersed = sorted(
        ((token, len(values), round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0)
         for token, values in delta_by_analyte.items() if len(values) >= 20),
        key=lambda row: -row[1])[:10]

    return {
        'graphs': graphs,
        'relation_counts': dict(relation_counts),
        'informative_edges': informative_total,
        'structural_edges': structural_total,
        'undeclared_relations': sorted(set(relation_counts) - set(STRUCTURAL_RELATIONS) - set(INFORMATIVE_RELATIONS)),
        'reconstruction': {
            'graphs_fully_reconstructible_from_tabular_view': reconstructible,
            'graphs_with_extra_facts': graphs - reconstructible,
            'mean_extra_facts_per_such_graph': round(sum(extra_facts) / len(extra_facts), 2) if extra_facts else 0.0,
            'interpretation': ('A graph counts as reconstructible only when it carries no informative '
                               'payload at all. v1 was reconstructible for every graph by construction.'),
        },
        'membership_collisions': {
            'rich_view_distinct': len(membership_payloads),
            'rich_view_mapping_to_multiple_payloads': len(collisions),
            'coarse_view_distinct': len(coarse_payloads),
            'coarse_view_mapping_to_multiple_payloads': len(coarse_collisions),
            'interpretation': ('Under the coarse presence-only view -- the one a v1-style tabular '
                               'baseline consumes -- graphs sharing an identical feature row but '
                               'carrying different informative payloads prove the payload is not a '
                               'function of the feature row. The rich view adds continuous summaries '
                               'and is near-unique by construction, so its collision count is '
                               'uninformative and is reported only for completeness.'),
        },
        'payload_dispersion_top_analytes': [
            {'analyte': token, 'n': n, 'delta_pstdev': sd} for token, n, sd in dispersed],
        'interval_hours': {
            'n': len(interval_values),
            'median': round(statistics.median(interval_values), 2) if interval_values else None,
            'min': round(min(interval_values), 2) if interval_values else None,
            'max': round(max(interval_values), 2) if interval_values else None,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.artifact / 'graphs.jsonl', args.limit)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.write_text(text + '\n')


if __name__ == '__main__':
    main()
