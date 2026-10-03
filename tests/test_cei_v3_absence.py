"""Synthetic tests for CEI-GNN v3 unit U2: absence universe and derivation.

No data files are opened; every fixture is built in the test.
"""
from __future__ import annotations

import hashlib
import json

import pytest
import torch

from core import NODE_KINDS
from cei.studies.cei_v3_absence import (
    ABSENCE_LABEL, Universe, fit_universe, index_visit_absence)

MEASUREMENT, VITAL, COMPLAINT = (NODE_KINDS.index('measurement'), NODE_KINDS.index('vital'),
                                 NODE_KINDS.index('complaint'))
MEASUREMENT_KINDS = (MEASUREMENT, VITAL)

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


# ---------------------------------------------------------- absence derivation
# Universe slots: 0 = lab:50912 (token 1), 1 = heartrate (token 3), 2 = temperature (token 4).
SLOTS = torch.tensor([-1, 0, -1, 1, 2, -1, -1])


def _graph(nodes, memberships, num_visits):
    """One graph in local coordinates: nodes = [(kind, token)], memberships = [(visit, node)].

    Mirrors contracts.validate_visit_membership_record: node 0 is the global patient
    node without membership; every other node has at least one visit ordinal.
    """
    node_type = torch.tensor([kind for kind, _ in nodes], dtype=torch.long)
    token = torch.tensor([tok for _, tok in nodes], dtype=torch.long)
    membership = (torch.tensor(sorted(memberships), dtype=torch.long).t().contiguous()
                  if memberships else torch.zeros((2, 0), dtype=torch.long))
    return node_type, token, membership, num_visits


def _batch(graphs):
    """Collate as ClinicalGraphData.__inc__ does: visits offset by num_visits, nodes by N."""
    types, tokens, memberships, counts = [], [], [], []
    visit_offset = node_offset = 0
    for node_type, token, membership, num_visits in graphs:
        types.append(node_type)
        tokens.append(token)
        memberships.append(membership + torch.tensor([[visit_offset], [node_offset]]))
        counts.append(num_visits)
        visit_offset += num_visits
        node_offset += node_type.numel()
    return (torch.cat(memberships, dim=1), torch.tensor(counts, dtype=torch.long),
            torch.cat(types), torch.cat(tokens))


PATIENT = (NODE_KINDS.index('patient'), 0)
VISIT = (NODE_KINDS.index('visit'), 0)


def test_prior_visit_only_item_is_absent_and_index_visit_item_is_present():
    # Two visits (ordinals 0, 1); index visit = 1. Heartrate only at visit 0 -> absent.
    # lab:50912 at visit 1 -> present. Temperature nowhere -> absent.
    graph = _graph([PATIENT, VISIT, (VITAL, 3), (MEASUREMENT, 1)],
                   [(1, 1), (0, 2), (1, 3)], 2)
    absent = index_visit_absence(*_batch([graph]), SLOTS, MEASUREMENT_KINDS)
    assert absent.dtype == torch.bool and absent.shape == (1, 3)
    assert absent.tolist() == [[False, True, True]]


def test_invalid_value_index_node_counts_as_present_and_duplicates_count_once():
    # F9: presence is derived from membership only; the derivation receives no
    # has_value column, so a node with has_value=0 at the index visit is present.
    # Two heartrate nodes at the index visit count once (no error, still present).
    graph = _graph([PATIENT, VISIT, (VITAL, 3), (VITAL, 3), (MEASUREMENT, 1)],
                   [(0, 1), (0, 2), (0, 3), (0, 4)], 1)
    absent = index_visit_absence(*_batch([graph]), SLOTS, MEASUREMENT_KINDS)
    assert absent.tolist() == [[False, False, True]]


def test_node_in_both_prior_and_index_visit_is_ambiguous_and_raises():
    # V3 §4.3: membership that is ambiguous for an item node fails, it is never
    # inferred; EXT §9 U2 step 3: node with two memberships -> raises.
    graph = _graph([PATIENT, VISIT, (VITAL, 3)], [(1, 1), (0, 2), (1, 2)], 2)
    with pytest.raises(ValueError, match='membership'):
        index_visit_absence(*_batch([graph]), SLOTS, MEASUREMENT_KINDS)


def test_non_measurement_kinds_and_non_universe_tokens_are_ignored():
    # A complaint node carrying universe token 3 at the index visit must not count
    # (kind filter); a measurement with token 6 (not in universe) changes nothing.
    graph = _graph([PATIENT, VISIT, (COMPLAINT, 3), (MEASUREMENT, 6)],
                   [(0, 1), (0, 2), (0, 3)], 1)
    absent = index_visit_absence(*_batch([graph]), SLOTS, MEASUREMENT_KINDS)
    assert absent.tolist() == [[True, True, True]]
    # Passing only the vital kind makes measurement presence invisible.
    graph = _graph([PATIENT, VISIT, (MEASUREMENT, 1), (VITAL, 3)], [(0, 1), (0, 2), (0, 3)], 1)
    absent = index_visit_absence(*_batch([graph]), SLOTS, (VITAL,))
    assert absent.tolist() == [[True, False, True]]


def test_batch_offsets_use_cumsum_of_num_visits_minus_one():
    # Graph 0: 3 visits, heartrate at its index visit (ordinal 2) -> batch visit 2.
    # Graph 1: 1 visit, lab at ordinal 0 -> batch visit 3; heartrate absent.
    # Graph 2: 2 visits, temperature at its prior visit only (batch visit 4), the
    #          index visit is batch visit 5 -> everything absent.
    g0 = _graph([PATIENT, VISIT, (VITAL, 3)], [(2, 1), (2, 2)], 3)
    g1 = _graph([PATIENT, VISIT, (MEASUREMENT, 1)], [(0, 1), (0, 2)], 1)
    g2 = _graph([PATIENT, VISIT, (VITAL, 4)], [(1, 1), (0, 2)], 2)
    membership, num_visits, node_type, token = _batch([g0, g1, g2])
    assert membership[0].tolist() == [2, 2, 3, 3, 4, 5]
    absent = index_visit_absence(membership, num_visits, node_type, token, SLOTS,
                                 MEASUREMENT_KINDS)
    assert absent.shape == (3, 3)
    assert absent.tolist() == [[True, False, True],
                               [False, True, True],
                               [True, True, True]]


def test_empty_batch_and_empty_membership_are_all_absent():
    # No membership pairs at all is a contract violation only for visit-specific
    # nodes; a graph with just the global patient node and an index visit node with
    # membership yields every universe item absent.
    graph = _graph([PATIENT, VISIT], [(0, 1)], 1)
    absent = index_visit_absence(*_batch([graph]), SLOTS, MEASUREMENT_KINDS)
    assert absent.tolist() == [[True, True, True]]


def test_absence_label_wording_follows_f8():
    assert ABSENCE_LABEL.format(item='vital:heartrate') == (
        'vital:heartrate: no recorded result at this visit')
    assert 'not measured' not in ABSENCE_LABEL


# ---------------------------------------------------------- membership contract

def _single(graph):
    return _batch([graph])


def test_item_node_with_two_memberships_raises():
    # Measurement node 2 is assigned to visits 0 and 1 (2 visits): ambiguous.
    graph = _graph([PATIENT, VISIT, (MEASUREMENT, 1)], [(1, 1), (0, 2), (1, 2)], 2)
    with pytest.raises(ValueError, match='membership'):
        index_visit_absence(*_single(graph), SLOTS, MEASUREMENT_KINDS)
    # The same node with one membership is fine.
    graph = _graph([PATIENT, VISIT, (MEASUREMENT, 1)], [(1, 1), (1, 2)], 2)
    assert index_visit_absence(*_single(graph), SLOTS, MEASUREMENT_KINDS).tolist() == [
        [False, True, True]]


def test_item_node_without_membership_raises():
    # Vital node 3 has no membership pair at all: the derivation must not infer.
    graph = _graph([PATIENT, VISIT, (MEASUREMENT, 1), (VITAL, 3)], [(0, 1), (0, 2)], 1)
    with pytest.raises(ValueError, match='membership'):
        index_visit_absence(*_single(graph), SLOTS, MEASUREMENT_KINDS)


def test_item_node_outside_universe_still_needs_membership():
    # The contract holds for every measurement/vital node, not only universe items.
    graph = _graph([PATIENT, VISIT, (MEASUREMENT, 6)], [(0, 1)], 1)
    with pytest.raises(ValueError, match='membership'):
        index_visit_absence(*_single(graph), SLOTS, MEASUREMENT_KINDS)


def test_non_item_nodes_are_not_checked_for_membership():
    # Patient (global) and complaint nodes are outside the item contract: a
    # complaint without membership does not raise here.
    graph = _graph([PATIENT, VISIT, (COMPLAINT, 5), (VITAL, 3)], [(0, 1), (0, 3)], 1)
    assert index_visit_absence(*_single(graph), SLOTS, MEASUREMENT_KINDS).tolist() == [
        [True, False, True]]


def test_membership_contract_is_checked_across_the_batch():
    g0 = _graph([PATIENT, VISIT, (VITAL, 3)], [(0, 1), (0, 2)], 1)
    g1 = _graph([PATIENT, VISIT, (VITAL, 3)], [(1, 1), (0, 2), (1, 2)], 2)
    with pytest.raises(ValueError, match='membership'):
        index_visit_absence(*_batch([g0, g1]), SLOTS, MEASUREMENT_KINDS)
    g2 = _graph([PATIENT, VISIT, (VITAL, 4)], [(0, 1)], 1)
    with pytest.raises(ValueError, match='membership'):
        index_visit_absence(*_batch([g0, g2]), SLOTS, MEASUREMENT_KINDS)


def test_membership_outside_batch_or_crossing_graphs_raises():
    g0 = _graph([PATIENT, VISIT, (VITAL, 3)], [(0, 1), (0, 2)], 1)
    membership, num_visits, node_type, token = _batch([g0])
    bad_visit = membership.clone()
    bad_visit[0, 1] = 1  # visit 1 does not exist (num_visits = 1)
    with pytest.raises(ValueError):
        index_visit_absence(bad_visit, num_visits, node_type, token, SLOTS, MEASUREMENT_KINDS)
    bad_node = membership.clone()
    bad_node[1, 1] = 7
    with pytest.raises(ValueError):
        index_visit_absence(bad_node, num_visits, node_type, token, SLOTS, MEASUREMENT_KINDS)
    with pytest.raises(ValueError):
        index_visit_absence(membership, torch.tensor([0]), node_type, token, SLOTS,
                            MEASUREMENT_KINDS)
