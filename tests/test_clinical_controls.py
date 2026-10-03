"""Bounded regression tests for the clinical graph's model/control contract."""
from __future__ import annotations

import copy
from collections import Counter

import pytest
import torch

from core.model import (
    ClinicalGNN, EdgeConditionedLayer, GatedConceptHubLayer,
    HeteroGraphTransformerLayer,
)
from core.tensorize import PAYLOAD_WIDTH


def _layer(arm, use_edge_payload):
    torch.manual_seed(42)
    kwargs = dict(dropout=0.0, use_edge_payload=use_edge_payload)
    if arm == 'hgt':
        return HeteroGraphTransformerLayer(8, 4, PAYLOAD_WIDTH, heads=2, **kwargs).eval()
    if arm.startswith('gchm'):
        return GatedConceptHubLayer(
            8, 4 + PAYLOAD_WIDTH, torch.tensor([1., 2., 3.]),
            modulation='additive' if arm == 'gchm_additive' else 'multiplicative',
            **kwargs).eval()
    return EdgeConditionedLayer(8, 4 + PAYLOAD_WIDTH, **kwargs).eval()


@pytest.mark.parametrize('arm', ['edge_conditioned', 'gchm', 'gchm_additive', 'hgt'])
def test_no_edge_payload_preserves_relations_and_ignores_only_numeric_tail(arm):
    full, ablated = _layer(arm, True), _layer(arm, False)
    x = torch.randn(4, 8)
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 2], [2, 2, 1, 1, 3, 3]])
    relation = torch.tensor([0, 1, 2, 3, 1, 2])
    edge_attr = torch.cat([torch.nn.functional.one_hot(relation, 4).float(),
                           torch.randn(6, PAYLOAD_WIDTH)], dim=1)
    original = edge_attr.clone()
    blank = edge_attr.clone()
    blank[:, -PAYLOAD_WIDTH:] = 0

    def run(layer, attributes, kinds=relation):
        if arm == 'hgt':
            return layer(x, edge_index, kinds, attributes[:, -PAYLOAD_WIDTH:])
        return layer(x, edge_index, attributes)

    changed_relation = (relation + 1) % 4
    changed_attr = edge_attr.clone()
    changed_attr[:, :4] = torch.nn.functional.one_hot(changed_relation, 4)
    with torch.no_grad():
        actual = run(ablated, edge_attr)
        expected = run(full, blank)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, run(ablated, blank), rtol=0, atol=0)
        assert (actual - run(ablated, changed_attr, changed_relation)).abs().max() > 1e-6
        assert (actual - run(full, edge_attr)).abs().max() > 1e-6
    torch.testing.assert_close(edge_attr, original, rtol=0, atol=0)
    assert sum(p.numel() for p in full.parameters()) == sum(
        p.numel() for p in ablated.parameters())
    for name, param in full.state_dict().items():
        torch.testing.assert_close(param, ablated.state_dict()[name], rtol=0, atol=0)


def test_relation_diagnostic_exercises_full_message_not_just_first_linear(monkeypatch):
    from core import mechanism_check

    calls = []
    original = EdgeConditionedLayer.message

    from functools import wraps

    @wraps(original)
    def capture(self, x_j, x_i, edge_attr):
        result = original(self, x_j, x_i, edge_attr)
        calls.append(result.detach().clone())
        return result

    monkeypatch.setattr(EdgeConditionedLayer, 'message', capture)
    diagnostic = next(fn for name, fn in mechanism_check.CHECKS
                      if name.startswith('baseline_relation_'))
    report, ok = diagnostic()
    assert len(calls) == 2, 'diagnostic never exercised the full message MLP'
    difference = calls[0] - calls[1]
    measured = float((difference - difference[0]).abs().max())
    assert report['message_variation'] == measured
    assert report['preactivation_variation'] < 1e-6
    assert measured > 0.5  # constructed ReLU-crossing witness, not a lucky random seed
    assert ok
    assert diagnostic(seed=999) == (report, ok)


@pytest.mark.parametrize('conv', ['edge_conditioned', 'gchm', 'hgt'])
def test_nomp_active_parameter_count_excludes_bypassed_convs(conv):
    from torch_geometric.data import Data

    model = ClinicalGNN(
        num_tokens=10, node_dim=4, edge_dim=4 + PAYLOAD_WIDTH, num_classes=3,
        hidden=8, layers=2, dropout=0., token_dim=4, conv=conv, num_triples=4,
        payload_dim=PAYLOAD_WIDTH, heads=2, degree_histogram=torch.tensor([1., 3., 2.]),
        use_message_passing=False)
    total = sum(p.numel() for p in model.parameters())
    expected_active = sum(p.numel() for name, p in model.named_parameters()
                          if not name.startswith('layers.'))
    counter = getattr(model, 'active_parameter_count', None)
    assert callable(counter), 'NOMP needs an active count separate from allocated total'
    assert model.parameter_count() == total
    assert counter() == expected_active < total
    # Gradients on the actual NOMP forward establish which registered blocks run.
    graph = Data(x=torch.randn(3, 4), token=torch.tensor([1, 2, 3]))
    model(graph).square().sum().backward()
    assert all(p.grad is None for p in model.layers.parameters())
    assert sum(p.numel() for p in model.parameters() if p.grad is not None) == counter()
    # The same state remains loadable and the count follows the execution flag.
    before = copy.deepcopy(model.state_dict())
    model.use_message_passing = True
    assert counter() == total
    assert model.parameter_count() == total
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def _degrees(edges, kinds):
    result = Counter()
    for e in edges:
        triple = (e['relation'], kinds[e['source']], kinds[e['target']])
        result[triple + ('out', e['source'])] += 1
        result[triple + ('in', e['target'])] += 1
    return result


def _pairs(edges):
    return Counter((e['relation'], e['source'], e['target']) for e in edges)


def test_rewiring_preserves_per_node_typed_degrees_and_existing_multiplicity():
    from core.rewiring import rewire_edges

    nodes = ([{'id': f'm{i}', 'kind': 'measurement'} for i in range(5)]
             + [{'id': f'a{i}', 'kind': 'analyte'} for i in range(3)])
    kinds = {n['id']: n['kind'] for n in nodes}
    edges = [{'source': f'm{s}', 'target': f'a{t}', 'relation': 'instance_of',
              'informative': False} for s, t in [(0, 0), (0, 0), (1, 1), (2, 0), (3, 2)]]
    edges += [{'source': 'a0', 'target': 'a1', 'relation': 'instance_of',
               'informative': False},
              {'source': 'm0', 'target': 'm1', 'relation': 'trajectory_of',
               'informative': True, 'delta': 123.}]
    before = copy.deepcopy((edges, nodes))
    actual = rewire_edges(edges, nodes, kinds, {'instance_of'}, 'sample:17', 4)
    assert _degrees(actual, kinds) == _degrees(edges, kinds)
    assert len(actual) == len(edges)
    assert actual[-1] == edges[-1]
    assert (edges, nodes) == before
    assert all(e['source'] != e['target'] for e in actual)
    assert all(count <= max(1, _pairs(edges)[pair])
               for pair, count in _pairs(actual).items())
    assert _pairs(actual) != _pairs(edges)
    assert actual == rewire_edges(edges, nodes, kinds, {'instance_of'}, 'sample:17', 4)


def _measurement(node_id, value, time, token='lab:A', unit='mg/dL'):
    return {'id': node_id, 'kind': 'measurement', 'token': token, 'value': value,
            'unit': unit, 'time_hours': time}


def _lab_edges():
    return [{'source': s, 'target': t, 'relation': 'baseline_of', 'informative': True,
             'delta': 999., 'interval_hours': 999., 'rate_per_hour': 999.,
             'comparable_units': True, 'last_seen_hours': -999., 'prior_encounters': 99,
             'provenance': {'source_uri': 'old-pair-only'}, 'unknown_pair_field': 42}
            for s, t in [('a', 'b'), ('c', 'd')]]


@pytest.mark.parametrize('mismatch', [None, 'token', 'unit', 'missing_unit', 'nonfinite_value'])
def test_rewiring_recomputes_comparable_payload_and_clears_unverifiable_fields(mismatch):
    from core.rewiring import rewire_edges

    nodes = [_measurement('a', 1., -5.), _measurement('b', 9., -2.),
             _measurement('c', 7., -10.), _measurement('d', 13., -1.)]
    if mismatch == 'token':
        nodes[2]['token'] = nodes[3]['token'] = 'lab:B'
    elif mismatch == 'unit':
        nodes[2]['unit'] = nodes[3]['unit'] = 'mmol/L'
    elif mismatch == 'missing_unit':
        for node in nodes:
            node.pop('unit')
    elif mismatch == 'nonfinite_value':
        nodes[0]['value'] = nodes[2]['value'] = float('nan')
    lookup = {n['id']: n for n in nodes}
    kinds = {n['id']: n['kind'] for n in nodes}
    edges = _lab_edges()
    actual = rewire_edges(edges, nodes, kinds, {'baseline_of'}, 'sample', 7)
    assert _pairs(actual) != _pairs(edges)
    for edge in actual:
        source, target = lookup[edge['source']], lookup[edge['target']]
        interval = target['time_hours'] - source['time_hours']
        assert edge['interval_hours'] == interval
        if mismatch is None:
            assert edge['delta'] == target['value'] - source['value']
            assert edge['rate_per_hour'] == edge['delta'] / interval
        else:
            assert edge.get('delta') is None
            assert edge.get('rate_per_hour') is None
        assert edge.get('last_seen_hours') is None
        assert edge.get('prior_encounters') is None
        assert edge.get('provenance') is None
        assert edge.get('unknown_pair_field') is None
    assert edges == _lab_edges()


def test_rewiring_recurrence_uses_node_evidence_not_old_edge_claims():
    from core.rewiring import rewire_edges

    nodes = [{'id': 'd0', 'kind': 'diagnosis', 'prior_encounters': 2},
             {'id': 'd1', 'kind': 'diagnosis', 'prior_encounters': 4,
              'last_seen_hours': -50.},
             {'id': 'v0', 'kind': 'visit'}, {'id': 'v1', 'kind': 'visit'}]
    kinds = {n['id']: n['kind'] for n in nodes}
    edges = [{'source': f'd{i}', 'target': f'v{i}', 'relation': 'recurrence_of',
              'informative': True, 'prior_encounters': 999, 'last_seen_hours': -999.,
              'delta': 999.} for i in range(2)]
    actual = rewire_edges(edges, nodes, kinds, {'recurrence_of'}, 'sample', 1)
    assert [e['prior_encounters'] for e in actual] == [2, 4]
    assert [e.get('last_seen_hours') for e in actual] == [None, -50.]
    assert all(e.get('delta') is None for e in actual)


@pytest.mark.parametrize('shape', ['empty', 'single', 'star', 'clique', 'reciprocal'])
def test_rewiring_constrained_topologies_are_explicit_noops(shape):
    from core.rewiring import (
        REWIRING_POLICY, rewire_edges_with_summary,
    )

    nodes = [{'id': str(i), 'kind': 'complaint'} for i in range(4)]
    pairs = {'empty': [], 'single': [(0, 1)], 'star': [(0, 1), (0, 2), (0, 3)],
             'clique': [(s, t) for s in range(4) for t in range(4) if s != t],
             'reciprocal': [(0, 1), (1, 0)]}[shape]
    edges = [{'source': str(s), 'target': str(t), 'relation': 'co_complaint',
              'informative': True, 'untouched': 'old-edge'} for s, t in pairs]
    kinds = {n['id']: n['kind'] for n in nodes}
    result, report = rewire_edges_with_summary(edges, nodes, kinds, {'co_complaint'}, 'p', 3)
    assert result == edges
    assert report['policy'] == REWIRING_POLICY == 'degree_preserving_payload_blind_v1'
    assert report['selected_edges'] == len(edges)
    assert report['accepted_swaps'] == report['changed_edges'] == 0
    assert report['removed_pair_occurrences'] == 0
    assert report['no_op']
    assert report['no_op_groups'] == report['groups'] == int(bool(edges))


def test_rewiring_seeded_per_sample_without_touching_global_rng():
    import random
    from core.rewiring import rewire_edges

    nodes = ([{'id': f's{i}', 'kind': 'analyte'} for i in range(8)]
             + [{'id': f't{i}', 'kind': 'knowledge'} for i in range(8)])
    kinds = {n['id']: n['kind'] for n in nodes}
    edges = [{'source': f's{i}', 'target': f't{i}', 'relation': 'medical:r',
              'informative': True, 'provenance': {'old': True}} for i in range(8)]
    state = random.getstate()
    first = rewire_edges(edges, nodes, kinds, {'medical:r'}, 'patient-A', 5)
    assert random.getstate() == state
    assert first == rewire_edges(edges, list(reversed(nodes)), kinds, {'medical:r'}, 'patient-A', 5)
    variants = {tuple(sorted(_pairs(rewire_edges(edges, nodes, kinds, {'medical:r'}, 'patient-A', seed))))
                for seed in range(4)}
    samples = {tuple(sorted(_pairs(rewire_edges(edges, nodes, kinds, {'medical:r'}, sample, 5))))
               for sample in ['patient-A', 'patient-B', 'patient-C']}
    assert len(variants) > 1
    assert len(samples) > 1
    assert all('provenance' not in e for e in first)
    # Values, tokens and payload never participate in choosing partners.
    perturbed = [{**n, 'value': 999, 'time_hours': 777, 'token': 'changed'} for n in nodes]
    changed_edges = [{**e, 'delta': -100.} for e in edges]
    assert _pairs(first) == _pairs(rewire_edges(changed_edges, perturbed, kinds,
                                               {'medical:r'}, 'patient-A', 5))


@pytest.mark.parametrize('seed', range(12))
def test_rewiring_multigraph_invariants_and_summary(seed):
    import random
    from core.rewiring import (
        rewire_edges, rewire_edges_with_summary,
    )

    rng = random.Random(seed)
    nodes = [{'id': str(i), 'kind': ['diagnosis', 'complaint'][i % 2]} for i in range(10)]
    kinds = {n['id']: n['kind'] for n in nodes}
    # Include parallel occurrences and existing loops, not only simple graphs.
    edges = [{'source': str(rng.randrange(9)), 'target': str(rng.randrange(9)),
              'relation': rng.choice(['comorbid_with', 'co_complaint', 'untouched']),
              'informative': True, 'delta': 123.} for _ in range(70)]
    snapshot = copy.deepcopy((edges, nodes))
    selected = ['comorbid_with', 'co_complaint']
    result, report = rewire_edges_with_summary(edges, nodes, kinds, selected, 'sample', seed)
    assert _degrees(result, kinds) == _degrees(edges, kinds)
    assert len(result) == len(edges)
    assert (edges, nodes) == snapshot
    assert report['changed_edges'] == report['accepted_swaps'] * 2
    removed = sum((_pairs(edges) - _pairs(result)).values())
    assert report['removed_pair_occurrences'] == removed
    assert report['no_op'] == (removed == 0)
    for before, after in zip(edges, result):
        assert kinds[before['source']] == kinds[after['source']]
        assert kinds[before['target']] == kinds[after['target']]
        assert before['relation'] == after['relation']
        if before['relation'] == 'untouched':
            assert before == after
        if (before['source'], before['target']) != (after['source'], after['target']):
            assert after['source'] != after['target']
            assert after.get('delta') is None
    assert all(count <= max(1, _pairs(edges)[pair])
               for pair, count in _pairs(result).items())
    assert result == rewire_edges(edges, nodes, kinds, list(reversed(selected)), 'sample', seed)


@pytest.mark.parametrize('times', [(-10., -9., -2., -1.), (0., 0., 0., 0.), (None, -2., -5., None)])
def test_rewiring_elapsed_time_never_reuses_old_rate_when_time_is_invalid(times):
    from core.rewiring import rewire_edges

    nodes = [_measurement(node, value, time) for node, value, time in zip('abcd', [1., 9., 7., 13.], times)]
    lookup = {n['id']: n for n in nodes}
    kinds = {n['id']: n['kind'] for n in nodes}
    result = rewire_edges(_lab_edges(), nodes, kinds, {'baseline_of'}, 'sample', 0)
    for e in result:
        start, end = lookup[e['source']]['time_hours'], lookup[e['target']]['time_hours']
        expected = end - start if start is not None and end is not None else None
        assert e['interval_hours'] == expected
        if expected is None or expected <= 0:
            assert e['rate_per_hour'] is None
        else:
            assert e['rate_per_hour'] == e['delta'] / expected
