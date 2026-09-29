"""Synthetic tests for CEI-GNN v3 unit U2: absence universe and derivation.

No data files are opened; every fixture is built in the test.
"""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from comparison.standardized.clinical_graph_v2.cei_v3_absence import (
    Universe, fit_universe)

VOCAB = {'measurement:lab:50912|mg/dL': 1, 'measurement:lab:51476|#/hpf': 2,
         'vital:heartrate': 3, 'vital:temperature': 4, 'complaint:chest pain': 5,
         'measurement:lab:rare|u': 6}
COUNTS = {'measurement:lab:50912|mg/dL': 100, 'vital:heartrate': 20,
          'measurement:lab:51476|#/hpf': 19, 'vital:temperature': 25,
          'measurement:lab:rare|u': 3}


def _fit(**kwargs):
    return fit_universe(dict(COUNTS), dict(VOCAB), **kwargs)


# ---------------------------------------------------------------- universe fit

def test_universe_keeps_identities_with_at_least_min_graphs_sorted():
    universe = _fit()
    assert universe.items == ('measurement:lab:50912|mg/dL', 'vital:heartrate',
                              'vital:temperature')
    assert len(universe) == 3


def test_universe_threshold_is_inclusive_and_configurable():
    assert 'vital:heartrate' in _fit(min_graphs=20).items
    assert 'vital:heartrate' not in _fit(min_graphs=21).items
    assert 'measurement:lab:51476|#/hpf' in _fit(min_graphs=19).items


def test_universe_is_subset_of_vocabulary_or_raises():
    with pytest.raises(ValueError, match='UNK'):
        fit_universe({'measurement:lab:unseen|u': 50, **COUNTS}, dict(VOCAB))
    with pytest.raises(ValueError, match='UNK'):
        fit_universe(dict(COUNTS), {**VOCAB, 'vital:heartrate': 0})


def test_universe_refuses_rule_looser_than_vocabulary_rule():
    # F10: the universe rule must be at least as strict as token_min_count.
    with pytest.raises(ValueError, match='token_min_count'):
        _fit(min_graphs=19, token_min_count=20)
    universe = _fit(min_graphs=20, token_min_count=20)
    assert universe.state()['token_min_count'] == 20


def test_universe_refuses_bad_counts_and_threshold():
    with pytest.raises(ValueError):
        _fit(min_graphs=0)
    with pytest.raises(ValueError):
        fit_universe({'vital:heartrate': -1}, dict(VOCAB))
    with pytest.raises(ValueError):
        fit_universe({'vital:heartrate': 20.5}, dict(VOCAB))
    with pytest.raises(ValueError, match='empty'):
        fit_universe({'vital:heartrate': 1}, dict(VOCAB))


def test_slot_of_token_maps_universe_tokens_and_marks_others_minus_one():
    universe = _fit()
    slots = universe.slot_of_token()
    assert slots.dtype == torch.long
    assert slots.tolist() == [-1, 0, -1, 1, 2, -1, -1]
    assert slots.numel() == len(VOCAB) + 1
    assert universe.slot_of_token(num_tokens=9).tolist() == [-1, 0, -1, 1, 2, -1, -1, -1, -1]
    with pytest.raises(ValueError):
        universe.slot_of_token(num_tokens=3)


def test_universe_state_replay_and_sha256_are_stable():
    universe = _fit(token_min_count=20)
    state = universe.state()
    assert state['version'] == 'cei_v3_absence_universe_v1'
    assert state['items'] == ['measurement:lab:50912|mg/dL', 'vital:heartrate',
                              'vital:temperature']
    assert state['token_index'] == [1, 3, 4]
    assert state['min_graphs'] == 20 and state['token_min_count'] == 20
    assert state['num_tokens'] == 7
    assert state['graph_counts'] == {'measurement:lab:50912|mg/dL': 100,
                                     'vital:heartrate': 20, 'vital:temperature': 25}
    expected = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':'),
                                         ensure_ascii=True).encode()).hexdigest()
    assert universe.sha256() == expected
    assert len(universe.sha256()) == 64
    replayed = Universe.load(json.loads(json.dumps(state)))
    assert replayed.state() == state
    assert replayed.sha256() == universe.sha256()
    assert torch.equal(replayed.slot_of_token(), universe.slot_of_token())
    assert _fit(token_min_count=20).sha256() == universe.sha256()
    assert _fit(min_graphs=25).sha256() != universe.sha256()


def test_universe_load_refuses_tampered_state():
    state = _fit().state()
    with pytest.raises(ValueError):
        Universe.load({**state, 'version': 'other'})
    with pytest.raises(ValueError):
        Universe.load({**state, 'token_index': [0, 3, 4]})
    with pytest.raises(ValueError):
        Universe.load({**state, 'items': list(reversed(state['items']))})
    with pytest.raises(ValueError):
        Universe.load({**state, 'token_index': [1, 3]})
    with pytest.raises(ValueError):
        Universe.load({**state, 'graph_counts': {**state['graph_counts'], 'vital:heartrate': 5}})
