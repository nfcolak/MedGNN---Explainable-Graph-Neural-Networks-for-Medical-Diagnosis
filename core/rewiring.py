"""Seeded, degree-preserving topology controls for clinical graphs.

This is a PAYLOAD-BLIND null: callers must disable numeric edge payload in BOTH
rewired and real-edge arms. Same-type swaps need not remain clinically meaningful
(e.g. they may connect different analytes or reverse temporal order). This is not
an estimate of a pure pairing effect for a payload-bearing model.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict

REWIRING_POLICY = 'degree_preserving_payload_blind_v1'
_MAX_PARTNER_PROBES = 64


def _finite(value):
    """Unknown/nonfinite measurements cannot justify a derived payload."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _new_pair_payload(edge, source, target):
    """Rebuild only claims derivable from the NEW endpoints, never old provenance."""
    rebuilt = {k: edge[k] for k in ('source', 'target', 'relation', 'informative') if k in edge}
    relation = edge['relation']
    if (relation in ('baseline_of', 'trajectory_of')
            and source['kind'] == target['kind'] == 'measurement'):
        start, end = _finite(source.get('time_hours')), _finite(target.get('time_hours'))
        interval = _finite(end - start) if start is not None and end is not None else None
        same_token = source.get('token') not in (None, '') and source.get('token') == target.get('token')
        same_unit = source.get('unit') not in (None, '') and source.get('unit') == target.get('unit')
        left, right = _finite(source.get('value')), _finite(target.get('value'))
        delta = (_finite(right - left) if same_token and same_unit
                 and left is not None and right is not None else None)
        # A null edge may run backward: retain the signed elapsed time for audit,
        # but do not assert a forward rate. Callers MUST blank these model inputs.
        rate = (_finite(delta / interval) if delta is not None
                and interval is not None and interval > 0 else None)
        rebuilt.update(delta=delta, interval_hours=interval, rate_per_hour=rate,
                       comparable_units=bool(same_unit))
    elif relation == 'recurrence_of' and source['kind'] == 'diagnosis' and target['kind'] == 'visit':
        prior = _finite(source.get('prior_encounters'))
        if prior is not None and (prior < 0 or not prior.is_integer()):
            prior = None
        # Builder diagnosis nodes normally lack last_seen_hours. Never substitute
        # diagnosis time or carry over the old edge's recency without node evidence.
        recency = _finite(source.get('last_seen_hours'))
        rebuilt.update(prior_encounters=prior,
                       last_seen_hours=recency if recency is not None and recency <= 0 else None)
    return rebuilt


def rewire_edges(edges, nodes, node_kind, relations, sample_id, seed) -> list:
    """Return copied edges after within-meta-relation directed double-edge swaps.

    ``(u, v), (x, y) -> (u, y), (x, v)`` preserves each node's in/out degree
    including multiplicity. Sources/targets never leave their original kind or
    graph. No nodes or nonselected edges are removed, no new self-loop is created,
    and pair multiplicity never exceeds max(1, its original multiplicity).

    Changed pairs lose all old pair-specific fields, including provenance. Lab
    delta/rate require the same analyte and known matching units; elapsed time is
    recomputed from endpoint times. Recurrence counts/recency require node evidence.
    Unchanged rows keep their original payload. All model-side payload must still
    be disabled: reconstructing some fields does not make the null edges factual.

    Randomized disjoint swaps move each edge at most once; this is a bounded null,
    not a uniform sample of all possible degree-preserving graphs. A group may be
    unchanged when constrained or when the bounded partner search finds no swap.
    """
    return rewire_edges_with_summary(edges, nodes, node_kind, relations, sample_id, seed)[0]


def rewire_edges_with_summary(edges, nodes, node_kind, relations, sample_id, seed):
    """Same control with an explicit per-call report; no mutable global counters.

    ``changed_edges`` counts changed edge rows; ``removed_pair_occurrences`` counts
    actual multiset topology changes (rows can move without changing topology).
    ``no_op_groups`` and ``no_op`` use the latter, not the attempted swap count.
    A no-op is an observed outcome, NOT proof that no legal swap exists.
    """
    selected = frozenset(relations)
    lookup = {node['id']: node for node in nodes}
    groups = defaultdict(list)
    out = [dict(e) for e in edges]
    for i, edge in enumerate(edges):
        if edge['relation'] in selected:
            triple = (edge['relation'], node_kind[edge['source']], node_kind[edge['target']])
            groups[triple].append(i)
    summary = {'policy': REWIRING_POLICY, 'selected_edges': sum(map(len, groups.values())),
               'groups': len(groups), 'attempted_swaps': 0, 'accepted_swaps': 0,
               'changed_edges': 0, 'removed_pair_occurrences': 0, 'no_op_groups': 0}
    for triple in sorted(groups):
        indices = groups[triple]
        original = Counter((edges[i]['source'], edges[i]['target']) for i in indices)
        counts = original.copy()
        # JSON framing prevents ambiguous sample/seed concatenation; local RNG
        # avoids touching training RNG state and makes relation set order irrelevant.
        rng = random.Random(json.dumps([REWIRING_POLICY, sample_id, seed, triple]))
        remaining = list(indices)
        rng.shuffle(remaining)
        while len(remaining) > 1:
            i = remaining.pop()
            u, v = out[i]['source'], out[i]['target']
            candidates = rng.sample(range(len(remaining)), min(len(remaining), _MAX_PARTNER_PROBES))
            for position in candidates:
                j = remaining[position]
                x, y = out[j]['source'], out[j]['target']
                summary['attempted_swaps'] += 1
                if u == x or v == y or u == y or x == v:
                    continue
                if (counts[u, y] >= max(1, original[u, y])
                        or counts[x, v] >= max(1, original[x, v])):
                    continue
                counts[u, v] -= 1
                counts[x, y] -= 1
                counts[u, y] += 1
                counts[x, v] += 1
                out[i]['target'], out[j]['target'] = y, v
                summary['accepted_swaps'] += 1
                remaining[position] = remaining[-1]
                remaining.pop()
                break
        removed = sum((original - counts).values())
        summary['removed_pair_occurrences'] += removed
        summary['no_op_groups'] += int(removed == 0)
    for i, (before, after) in enumerate(zip(edges, out)):
        if (before['source'], before['target']) != (after['source'], after['target']):
            summary['changed_edges'] += 1
            out[i] = _new_pair_payload(after, lookup[after['source']], lookup[after['target']])
    summary['no_op'] = summary['removed_pair_occurrences'] == 0
    return out, summary
