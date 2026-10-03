"""Shared, train-fitted clinical input encoding for GNN and tabular controls.

The value key/token identity includes exact node kind, clinical token and unit for
numeric nodes. Frequent finite values use clipped train z-scores; rare/unseen values
retain signed-log magnitude. Invalid/nonfinite values are explicitly missing.

The dense node prefix keeps kind/value/time/availability columns. Appended context
columns expose patient age, gender (source sex field), race, arrival transport and
index-visit acuity. Categorical dictionaries are train-only, with separate missing
and unseen codes. IDs, raw coverage and target fields are never predictor columns.

Type-index tensors support existing heterogeneous layers. Relation one-hot and
numeric edge payload layout are unchanged. Node layout and numeric identity ARE
incompatible with historical checkpoints: versioned state must be loaded explicitly.
Edge filtering/rewiring changes the input graph, not a causal identification claim.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from . import INFORMATIVE_RELATIONS, NODE_KINDS, STRUCTURAL_RELATIONS
from .contracts import iter_graphs_with_membership, validate_visit_membership_record

ALL_RELATIONS = tuple(STRUCTURAL_RELATIONS) + tuple(INFORMATIVE_RELATIONS)
KIND_INDEX = {k: i for i, k in enumerate(NODE_KINDS)}
RELATION_INDEX = {r: i for i, r in enumerate(ALL_RELATIONS)}
# The producer emits every hub->evidence relation in one direction only, so under
# source->target message passing no complaint, vital or laboratory node can reach
# the index visit at any depth. `edge_direction='bidirectional'` appends a typed
# reverse edge for exactly the relations whose inverse the producer does not emit.
# has_visit/index_visit_of and has_prior_diagnosis/recurrence_of are already
# explicit inverse pairs; co_complaint and comorbid_with are already symmetric.
# Reverse edges add routing, not information: same endpoints, same payload.
REVERSIBLE_RELATIONS = ('reports_complaint', 'measured_in', 'instance_of',
                        'observed_vital', 'baseline_of', 'trajectory_of',
                        'medical:member_of', 'medical:assesses', 'medical:measures')
REVERSE_RELATIONS = tuple('rev:' + relation for relation in REVERSIBLE_RELATIONS)
EDGE_DIRECTIONS = ('forward', 'bidirectional')
if not set(REVERSIBLE_RELATIONS) <= set(ALL_RELATIONS):
    raise ImportError('REVERSIBLE_RELATIONS names a relation outside the producer vocabulary')
UNK = 0
# Columns of `edge_attr` after the relation one-hot block: delta, has_delta,
# interval, has_interval, recency, has_recency, log1p(prior_encounters).
PAYLOAD_WIDTH = 7
TRIPLE_UNK = 0
PREPROCESSING_VERSION = 'clinical_inputs_v3'
CONTEXT_CATEGORICAL = ('gender', 'race', 'arrival_transport')
CONTEXT_NUMERIC = ('patient.age', 'visit:index.acuity')
NUMERIC_POLICY = {'min_count': 20, 'fitted': 'zscore_clip_10',
                  'rare_or_unseen': 'signed_log1p_without_clipping',
                  'invalid_or_nonfinite': 'missing_zero_and_has_value_zero',
                  'units': 'exact_kind_token_unit_no_conversion'}


class ClinicalGraphData(Data):
    """Clinical graph tensors with separate node and visit batch offsets."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'visit_membership_index':
            return value.new_tensor([[int(self.num_visits.item())], [self.num_nodes]])
        return super().__inc__(key, value, *args, **kwargs)


def triple_key(source_kind, relation, target_kind):
    """Canonical name of one HGT meta-relation <src type, relation, dst type>."""
    return f'{source_kind}|{relation}|{target_kind}'


def relation_vocabulary(edge_direction='forward'):
    """Ordered relation ids of one edge view; forward ids are never renumbered."""
    if edge_direction == 'forward':
        return ALL_RELATIONS
    if edge_direction == 'bidirectional':
        return ALL_RELATIONS + REVERSE_RELATIONS
    raise ValueError(f'unknown edge_direction {edge_direction!r}')


def triple_count(prep, edge_direction='forward'):
    """Meta-relation ids incl. index 0 for unseen; a reverse edge of fitted triple t
    gets id T + t, so the train-fitted forward state is reused, never refitted."""
    fitted = len(prep['triples'])
    if edge_direction == 'forward':
        return fitted + 1
    if edge_direction == 'bidirectional':
        return 2 * fitted + 1
    raise ValueError(f'unknown edge_direction {edge_direction!r}')


def finite_number(value):
    """Return a finite float or None; invalid/nonfinite inputs are missing."""
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def node_token(node):
    """Value-bearing identities include exact kind/token/unit; no conversions.

    Missing units remain JSON null and never borrow a known unit's statistics.
    Measurement/vital identity is stable even if its value is missing.
    """
    if node['kind'] in ('measurement', 'vital') or 'value' in node:
        return json.dumps([node['kind'], node['token'], node.get('unit')],
                          ensure_ascii=True, separators=(',', ':'))
    return node['token']


def signed_log(value):
    """Compress finite magnitudes while keeping direction; mark invalid missing."""
    value = finite_number(value)
    if value is None:
        return 0.0, 0.0
    return math.copysign(math.log1p(abs(value)), value), 1.0


def context_number(node, field):
    if field == 'patient.age' and node['kind'] == 'patient':
        return node.get('age')
    if (field == 'visit:index.acuity' and node['kind'] == 'visit'
            and node['token'] == 'visit:index'):
        return node.get('acuity')
    return None


def category_value(value):
    # Preserve source strings exactly; only absent/blank/non-string is missing.
    return value if isinstance(value, str) and value.strip() else None


def node_feature_layout(prep):
    layout = list(NODE_KINDS) + ['scaled_value', 'has_value', 'time_signed_log',
                                'has_time', 'available_signed_log', 'has_available']
    for field in CONTEXT_NUMERIC:
        layout.extend([f'context.{field}', f'context.{field}.has_value'])
    for field in CONTEXT_CATEGORICAL:
        layout.extend([f'context.{field}:<missing>', f'context.{field}:<unknown>'])
        layout.extend(f'context.{field}={json.dumps(v, ensure_ascii=True)}'
                      for v in prep['context_categories'][field])
    return layout


class Vocabulary:
    """Train-fitted token vocabulary. Index 0 is reserved for unseen tokens."""

    def __init__(self, tokens=None, min_count=1):
        self.tokens = list(tokens or [])
        self.index = {t: i + 1 for i, t in enumerate(self.tokens)}
        self.min_count = min_count

    @classmethod
    def fit(cls, counter, min_count):
        kept = sorted(t for t, c in counter.items() if c >= min_count)
        return cls(kept, min_count)

    def __len__(self):
        return len(self.tokens) + 1

    def get(self, token):
        return self.index.get(token, UNK)

    def state(self):
        return {'tokens': self.tokens, 'min_count': self.min_count}


class Scaler:
    """Train-only per-identity z-scores with signed-log rare/unseen fallback."""

    def __init__(self, stats=None):
        self.stats = stats or {}

    @classmethod
    def fit(cls, values_by_token, min_count=20):
        stats = {}
        for token, values in values_by_token.items():
            values = [v for value in values if (v := finite_number(value)) is not None]
            if len(values) < min_count:
                continue
            arr = np.asarray(values, dtype=np.float64)
            with np.errstate(over='ignore', invalid='ignore'):
                mean = float(arr.mean())
                scale = float(arr.std())
            if not math.isfinite(mean) or not math.isfinite(scale):
                continue  # even extreme finite values retain signed-log fallback
            if scale <= 0:
                scale = 1.0
            stats[token] = {'mean': mean, 'scale': scale, 'count': len(values)}
        return cls(stats)

    def transform(self, token, value, clip=10.0):
        value = finite_number(value)
        if value is None:
            return 0.0, 0.0
        s = self.stats.get(token)
        if s is None:
            return signed_log(value)
        z = (value - s['mean']) / s['scale']
        return float(np.clip(z, -clip, clip)), 1.0

    def state(self):
        return self.stats


def iter_graphs(path, limit=None):
    with Path(path).open() as stream:
        for i, line in enumerate(stream):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def fit_preprocessing(path, train_ids, token_min_count=20, limit=None, *, membership_path):
    """Fit vocabulary + scalers + meta-relation set on TRAIN graphs only."""
    tokens = Counter()
    values = {}
    triples = Counter()
    categories = {field: set() for field in CONTEXT_CATEGORICAL}
    for graph, _ in iter_graphs_with_membership(path, membership_path, limit):
        if graph['sample_id'] not in train_ids:
            continue
        kind = {}
        for node in graph['nodes']:
            token = node_token(node)
            tokens[token] += 1
            kind[node['id']] = node['kind']
            if node.get('value') is not None:
                values.setdefault(token, []).append(node['value'])
            for field in CONTEXT_NUMERIC:
                value = context_number(node, field)
                if value is not None:
                    values.setdefault('context.' + field, []).append(value)
            if node['kind'] == 'patient':
                for field in CONTEXT_CATEGORICAL:
                    category = category_value(node.get(field))
                    if category is not None:
                        categories[field].add(category)
        for edge in graph['edges']:
            triples[triple_key(kind[edge['source']], edge['relation'],
                               kind[edge['target']])] += 1
    if not tokens:
        raise ValueError('No training graphs found while fitting preprocessing')
    if not triples:
        raise ValueError('No training edges found while fitting meta-relations')
    return {'preprocessing_version': PREPROCESSING_VERSION,
            'context_categories': {f: sorted(v) for f, v in categories.items()},
            'vocabulary': Vocabulary.fit(tokens, token_min_count),
            'scaler': Scaler.fit(values),
            'token_min_count': token_min_count,
            # Index 0 stays free for a meta-relation unseen in training, so a
            # validation-only triple keeps its edge instead of silently changing
            # the graph's topology between folds.
            'triples': sorted(triples),
            'triple_counts': dict(triples)}


def _rewire_edges(edges, nodes, node_kind, relations, sample_id, seed):
    """Delegate the shared degree-preserving control; do not duplicate its policy."""
    from .rewiring import rewire_edges
    return rewire_edges(edges, nodes, node_kind, relations, sample_id, seed)


def encode_graph(graph, prep, visit_membership, edge_mode='all', drop_relations=(),
                 rewire_relations=(), rewire_seed=0, edge_direction='forward'):
    """Encode nodes, validated visit membership, and the selected edge view.

    Drop controls remove edges, never nodes. Rewiring delegates to the shared
    degree-preserving policy; a matched structural comparison requires disabling
    edge payload on BOTH the original and rewired runs. The run entrypoints enforce
    that contract. Neither unchanged nor changed scores alone identify causality.

    `edge_direction='forward'` is byte-identical to the historical encoding. The
    bidirectional view appends, after every forward edge, one typed reverse edge per
    kept edge of a REVERSIBLE relation, carrying the same payload.
    """
    if prep.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('Incompatible preprocessing version; refit with this encoder')
    relations = relation_vocabulary(edge_direction)
    validate_visit_membership_record(graph, visit_membership)
    vocab, scaler = prep['vocabulary'], prep['scaler']
    drop = frozenset(drop_relations)
    rewire = frozenset(rewire_relations)
    triple_index = prep.get('triple_index')
    if triple_index is None:
        triple_index = {t: i + 1 for i, t in enumerate(prep['triples'])}
        prep['triple_index'] = triple_index
    nodes = graph['nodes']
    index = {n['id']: i for i, n in enumerate(nodes)}

    x = np.zeros((len(nodes), len(node_feature_layout(prep))), dtype=np.float32)
    tok = np.zeros(len(nodes), dtype=np.int64)
    kind = np.zeros(len(nodes), dtype=np.int64)
    for i, n in enumerate(nodes):
        x[i, KIND_INDEX[n['kind']]] = 1.0
        token = node_token(n)
        value, has_value = scaler.transform(token, n.get('value'))
        t, has_t = signed_log(n.get('time_hours'))
        a, has_a = signed_log(n.get('available_hours'))
        x[i, len(NODE_KINDS):len(NODE_KINDS) + 6] = (value, has_value, t, has_t, a, has_a)
        offset = len(NODE_KINDS) + 6
        for field in CONTEXT_NUMERIC:
            x[i, offset:offset + 2] = scaler.transform(
                'context.' + field, context_number(n, field))
            offset += 2
        for field in CONTEXT_CATEGORICAL:
            categories = prep['context_categories'][field]
            if n['kind'] == 'patient':
                category = category_value(n.get(field))
                code = (0 if category is None else categories.index(category) + 2
                        if category in categories else 1)
                x[i, offset + code] = 1.0
            offset += len(categories) + 2
        tok[i] = vocab.get(token)
        kind[i] = KIND_INDEX[n['kind']]

    node_kind = {n['id']: n['kind'] for n in nodes}
    keep = []
    for e in graph['edges']:
        if e['relation'] in drop:
            continue
        if edge_mode == 'informative' and not e['informative']:
            continue
        if edge_mode == 'structural' and e['informative']:
            continue
        keep.append(e)

    rewired_edge_count = 0
    if rewire:
        before_pairs = Counter((e['relation'], e['source'], e['target'])
                               for e in keep if e['relation'] in rewire)
        keep = _rewire_edges(keep, nodes, node_kind, rewire,
                             graph['sample_id'], rewire_seed)
        after_pairs = Counter((e['relation'], e['source'], e['target'])
                              for e in keep if e['relation'] in rewire)
        # Count replaced directed endpoint-pair occurrences, not swap attempts or
        # reordered edge-list positions. A clique can correctly report zero.
        rewired_edge_count = sum((before_pairs - after_pairs).values())

    reverse = []
    if edge_direction == 'bidirectional':
        # Appended strictly after the forward block so forward edge ids are stable.
        reverse = [e for e in keep if e['relation'] in REVERSIBLE_RELATIONS]
    fitted_triples = len(prep['triples'])
    relation_index = {r: i for i, r in enumerate(relations)}
    total = len(keep) + len(reverse)
    src = np.fromiter([*(index[e['source']] for e in keep),
                       *(index[e['target']] for e in reverse)], dtype=np.int64, count=total)
    dst = np.fromiter([*(index[e['target']] for e in keep),
                       *(index[e['source']] for e in reverse)], dtype=np.int64, count=total)
    ea = np.zeros((total, len(relations) + PAYLOAD_WIDTH), dtype=np.float32)
    rel = np.zeros(total, dtype=np.int64)
    tri = np.zeros(total, dtype=np.int64)
    for i, (e, is_reverse) in enumerate([*((e, False) for e in keep),
                                         *((e, True) for e in reverse)]):
        relation = 'rev:' + e['relation'] if is_reverse else e['relation']
        ea[i, relation_index[relation]] = 1.0
        d, has_d = signed_log(e.get('delta'))
        iv, has_iv = signed_log(e.get('interval_hours'))
        rc, has_rc = signed_log(e.get('last_seen_hours'))
        prior = finite_number(e.get('prior_encounters', 0))
        if e.get('prior_encounters') is None:
            prior = 0.0
        if prior is None or prior < 0:
            raise ValueError('prior_encounters must be finite and nonnegative')
        ea[i, len(relations):] = (d, has_d, iv, has_iv, rc, has_rc, math.log1p(prior))
        rel[i] = relation_index[relation]
        forward_triple = triple_index.get(
            triple_key(node_kind[e['source']], e['relation'], node_kind[e['target']]),
            TRIPLE_UNK)
        tri[i] = (fitted_triples + forward_triple
                  if is_reverse and forward_triple != TRIPLE_UNK else forward_triple)

    # Node rows preserve graph order, so sidecar node positions map directly.
    pairs = visit_membership['membership_pairs']
    visit_membership_index = (torch.tensor(pairs, dtype=torch.long).t().contiguous()
                              if pairs else torch.zeros((2, 0), dtype=torch.long))
    data = ClinicalGraphData(x=torch.from_numpy(x),
                             edge_index=torch.from_numpy(np.stack([src, dst])) if total
                             else torch.zeros((2, 0), dtype=torch.long),
                             edge_attr=torch.from_numpy(ea))
    data.token = torch.from_numpy(tok)
    # All arms share these primitives; type indices add no private raw features.
    data.node_type = torch.from_numpy(kind)
    data.edge_relation = torch.from_numpy(rel)
    data.edge_triple = torch.from_numpy(tri)
    data.edge_payload = torch.from_numpy(
        np.ascontiguousarray(ea[:, len(relations):]))
    data.visit_membership_index = visit_membership_index
    data.num_visits = torch.tensor([len(visit_membership['visit_ordinals'])], dtype=torch.long)
    data.global_node_mask = torch.tensor(visit_membership['global_node_mask'], dtype=torch.bool)
    data.sample_id = graph['sample_id']
    data.rewired_edge_count = rewired_edge_count
    return data


def degree_histogram(datasets, max_degree=64):
    """In-degree histogram for PNA-style aggregators; train split only."""
    hist = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in datasets:
        if data.edge_index.numel() == 0:
            hist[0] += data.num_nodes
            continue
        deg = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
        deg = deg.clamp(max=max_degree)
        hist += torch.bincount(deg, minlength=max_degree + 1)
    return hist


def preprocessing_state(prep):
    if prep.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('Incompatible preprocessing version; cannot relabel old state')
    return {'preprocessing_version': PREPROCESSING_VERSION,
            'numeric_policy': dict(NUMERIC_POLICY),
            'context_categories': {f: list(prep['context_categories'][f])
                                   for f in CONTEXT_CATEGORICAL},
            'context_policy': {'categorical': 'patient_node_one_hot',
                               'missing_index': 0, 'unknown_index': 1,
                               'numeric_fields': list(CONTEXT_NUMERIC),
                               'sex_source_field': 'patient.gender',
                               'non_context_nodes': 'all_context_columns_zero'},
            'vocabulary': prep['vocabulary'].state(),
            'scaler': prep['scaler'].state(),
            'token_min_count': prep['token_min_count'],
            'triples': prep['triples'],
            'triple_counts': prep.get('triple_counts', {}),
            'node_feature_layout': node_feature_layout(prep),
            'edge_feature_layout': list(ALL_RELATIONS) + ['delta_signed_log', 'has_delta',
                                                          'interval_signed_log', 'has_interval',
                                                          'recency_signed_log', 'has_recency',
                                                          'log1p_prior_encounters'],
            'type_index_layout': {'node_type': list(NODE_KINDS),
                                  'edge_relation': list(ALL_RELATIONS),
                                  'edge_triple': ['<unseen>'] + prep['triples']},
            'fit_scope': 'train fold only'}


def edge_feature_layout(prep, edge_direction='forward'):
    """Realised `edge_attr` columns of one edge view.

    The fitted preprocessing state is direction-free and keeps the forward layout, so
    its hash is unchanged; runs record this realised layout next to it instead.
    """
    payload = preprocessing_state(prep)['edge_feature_layout'][len(ALL_RELATIONS):]
    return list(relation_vocabulary(edge_direction)) + payload


def load_preprocessing(state):
    if state.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('Incompatible preprocessing version; historical checkpoints '
                         'require their original source snapshot, not this encoder')
    prep = {'preprocessing_version': PREPROCESSING_VERSION,
            'context_categories': state['context_categories'],
            'vocabulary': Vocabulary(state['vocabulary']['tokens'],
                                     state['vocabulary']['min_count']),
            'scaler': Scaler(state['scaler']),
            'token_min_count': state['token_min_count'],
            'triples': state['triples'],
            'triple_counts': state.get('triple_counts', {})}
    for values in prep['context_categories'].values():
        if values != sorted(set(values)) or any(category_value(v) is None for v in values):
            raise ValueError('Incompatible context category layout')
    expected = preprocessing_state(prep)
    for key in ('numeric_policy', 'context_policy', 'node_feature_layout',
                'edge_feature_layout', 'type_index_layout', 'fit_scope'):
        if state.get(key) != expected[key]:
            raise ValueError(f'Incompatible preprocessing layout/policy: {key}')
    for stats in prep['scaler'].stats.values():
        if (finite_number(stats.get('mean')) is None
                or finite_number(stats.get('scale')) is None or stats['scale'] <= 0):
            raise ValueError('Invalid numeric scaler state')
    return prep
