"""Tensorize clinical_graph_v2/v3 JSONL into PyG heterogeneous-ish Data objects.

Everything fitted here -- token vocabularies, numeric scalers, degree histogram --
is fitted on the TRAIN fold only and frozen into a manifest. Validation and test
rows are transformed with the train statistics, never re-fitted, so no information
crosses the split boundary.

Node features are deliberately thin and uniform across kinds:

    [one-hot node kind | scaled value | has_value | signed-log time | has_time
     | signed-log availability | has_availability | token embedding index]

The token index is kept as a separate integer column so the model can embed it;
unseen validation/test tokens map to index 0 (UNK) rather than being dropped, which
would silently change a graph's topology between folds.

Edge features carry the payload that justifies the informative relations:

    [one-hot relation | signed-log delta | has_delta | signed-log interval
     | has_interval | signed-log recency | has_recency | prior_encounters]

`informative` edges are NOT filtered out here. Filtering is a model-side ablation
(`--edges informative|all|structural`), so one artifact serves every arm.
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

ALL_RELATIONS = tuple(STRUCTURAL_RELATIONS) + tuple(INFORMATIVE_RELATIONS)
KIND_INDEX = {k: i for i, k in enumerate(NODE_KINDS)}
RELATION_INDEX = {r: i for i, r in enumerate(ALL_RELATIONS)}
UNK = 0


def signed_log(value):
    """Compress heavy-tailed hour/delta magnitudes while keeping direction."""
    if value is None:
        return 0.0, 0.0
    return math.copysign(math.log1p(abs(float(value))), float(value)), 1.0


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
    """Per-token mean/scale for measurement values, fitted on the train fold."""

    def __init__(self, stats=None):
        self.stats = stats or {}

    @classmethod
    def fit(cls, values_by_token, min_count=20):
        stats = {}
        for token, values in values_by_token.items():
            if len(values) < min_count:
                continue
            arr = np.asarray(values, dtype=np.float64)
            mean = float(arr.mean())
            scale = float(arr.std())
            if not math.isfinite(scale) or scale <= 0:
                scale = 1.0
            stats[token] = {'mean': mean, 'scale': scale, 'count': len(values)}
        return cls(stats)

    def transform(self, token, value, clip=10.0):
        if value is None:
            return 0.0, 0.0
        s = self.stats.get(token)
        if s is None:
            # Unseen analyte: keep the observation, drop the unscaled magnitude.
            return 0.0, 1.0
        z = (float(value) - s['mean']) / s['scale']
        return float(np.clip(z, -clip, clip)), 1.0

    def state(self):
        return self.stats


def iter_graphs(path, limit=None):
    with Path(path).open() as stream:
        for i, line in enumerate(stream):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def fit_preprocessing(path, train_ids, token_min_count=20, limit=None):
    """Fit vocabulary + scalers on TRAIN graphs only."""
    tokens = Counter()
    values = {}
    for graph in iter_graphs(path, limit):
        if graph['sample_id'] not in train_ids:
            continue
        for node in graph['nodes']:
            tokens[node['token']] += 1
            if node['kind'] == 'measurement' and node.get('value') is not None:
                values.setdefault(node['token'], []).append(node['value'])
    if not tokens:
        raise ValueError('No training graphs found while fitting preprocessing')
    return {'vocabulary': Vocabulary.fit(tokens, token_min_count),
            'scaler': Scaler.fit(values),
            'token_min_count': token_min_count}


def encode_graph(graph, prep, edge_mode='all'):
    """Return a PyG Data object. `edge_mode` selects the relation ablation arm."""
    vocab, scaler = prep['vocabulary'], prep['scaler']
    nodes = graph['nodes']
    index = {n['id']: i for i, n in enumerate(nodes)}

    x = np.zeros((len(nodes), len(NODE_KINDS) + 6), dtype=np.float32)
    tok = np.zeros(len(nodes), dtype=np.int64)
    for i, n in enumerate(nodes):
        x[i, KIND_INDEX[n['kind']]] = 1.0
        value, has_value = scaler.transform(n['token'], n.get('value'))
        t, has_t = signed_log(n.get('time_hours'))
        a, has_a = signed_log(n.get('available_hours'))
        x[i, len(NODE_KINDS):] = (value, has_value, t, has_t, a, has_a)
        tok[i] = vocab.get(n['token'])

    keep = []
    for e in graph['edges']:
        if edge_mode == 'informative' and not e['informative']:
            continue
        if edge_mode == 'structural' and e['informative']:
            continue
        keep.append(e)

    src = np.fromiter((index[e['source']] for e in keep), dtype=np.int64, count=len(keep))
    dst = np.fromiter((index[e['target']] for e in keep), dtype=np.int64, count=len(keep))
    ea = np.zeros((len(keep), len(ALL_RELATIONS) + 7), dtype=np.float32)
    for i, e in enumerate(keep):
        ea[i, RELATION_INDEX[e['relation']]] = 1.0
        d, has_d = signed_log(e.get('delta'))
        iv, has_iv = signed_log(e.get('interval_hours'))
        rc, has_rc = signed_log(e.get('last_seen_hours'))
        prior = float(e.get('prior_encounters') or 0.0)
        ea[i, len(ALL_RELATIONS):] = (d, has_d, iv, has_iv, rc, has_rc,
                                      math.log1p(prior))

    data = Data(x=torch.from_numpy(x),
                edge_index=torch.from_numpy(np.stack([src, dst])) if len(keep)
                else torch.zeros((2, 0), dtype=torch.long),
                edge_attr=torch.from_numpy(ea))
    data.token = torch.from_numpy(tok)
    data.sample_id = graph['sample_id']
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
    return {'vocabulary': prep['vocabulary'].state(),
            'scaler': prep['scaler'].state(),
            'token_min_count': prep['token_min_count'],
            'node_feature_layout': list(NODE_KINDS) + ['scaled_value', 'has_value',
                                                       'time_signed_log', 'has_time',
                                                       'available_signed_log', 'has_available'],
            'edge_feature_layout': list(ALL_RELATIONS) + ['delta_signed_log', 'has_delta',
                                                          'interval_signed_log', 'has_interval',
                                                          'recency_signed_log', 'has_recency',
                                                          'log1p_prior_encounters'],
            'fit_scope': 'train fold only'}


def load_preprocessing(state):
    return {'vocabulary': Vocabulary(state['vocabulary']['tokens'],
                                     state['vocabulary']['min_count']),
            'scaler': Scaler(state['scaler']),
            'token_min_count': state['token_min_count']}
