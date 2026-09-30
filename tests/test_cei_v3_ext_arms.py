"""Synthetic tests for the CEI-GNN v3 extension arms E2w, E2d and E6a (unit X14).

Extensions spec §2 (size arms), §5.2 / §5.5 item 3 (bidirectional arm), §6 (arm table) and
§9 X14 (binding guards). Fixtures are the U3/U3x synthetic ones plus small in-memory graph
documents; nothing here opens a data file, run directory, checkpoint or test-fold row.
"""
from __future__ import annotations

import json
from argparse import Namespace

import pytest
import torch

from comparison.standardized.clinical_graph_v2 import NODE_KINDS
from comparison.standardized.clinical_graph_v2 import tensorize as tz
from comparison.standardized.clinical_graph_v2.cei_v3_ext import arm_guards
from comparison.standardized.clinical_graph_v2.contracts import (
    VISIT_MEMBERSHIP_CONTRACT_VERSION, VISIT_MEMBERSHIP_FILENAME)
from comparison.standardized.clinical_graph_v2.methods import METHOD_REGISTRY
from comparison.standardized.clinical_graph_v2.methods.base import parameter_count, read_clinical_batch
from comparison.standardized.clinical_graph_v2.tensorize import (ALL_RELATIONS, PAYLOAD_WIDTH,
                                                                 REVERSE_RELATIONS)
from tests.test_cei_gnn_v3_core import (CLASSES, EDGE_DIM, HIDDEN, K, NODE_DIM, NUM_TOKENS,
                                        TOKEN_DIM, TRIPLES, _graph, _randomise_gates, _run,
                                        _share_weights, _visit_graph, _write_state)
from tests.test_cei_gnn_v3_hooks import (BLOCK_DELTA_AT_128, CONTROL_HIDDEN, E6A_CONTROL,
                                         E6A_SHAPES, E6A_WIDENED, _count, _network)

RELATIONS_FORWARD = len(ALL_RELATIONS)                       # 15
RELATIONS_BIDIRECTIONAL = RELATIONS_FORWARD + len(REVERSE_RELATIONS)   # 24
UNIVERSE_SIZE = 3   # the U3 fixture universe: lab1, lab2, hr
FIXTURE_DIMENSIONS = dict(num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                          num_classes=CLASSES, hidden=HIDDEN, token_dim=TOKEN_DIM,
                          num_triples=TRIPLES, num_relations=RELATIONS_FORWARD, pair_rank=16,
                          num_node_types=8)
# Recorded v2 full-run binding (spec §1) and the analytic arm table of §2.3, §5.2, §5.3.
SPEC_V2_TOTAL = 92_300
SPEC_DELTAS = {'E2w': 177_792, 'E2d': 16_768, 'E6a': 4_224, 'E6b': 2_228}
SPEC_TOTALS = {'E2w': 270_092, 'E2d': 109_068, 'E6a': 96_524, 'E6b': 94_528}


def _adapter(tmp_path, *, hidden=CONTROL_HIDDEN, layers=1, edge_direction=None, seed=1234,
             num_triples=TRIPLES, edge_dim=EDGE_DIM, **options):
    """`cei_gnn_v3` adapter on the U3 state file with overridable runner dimensions."""
    state_path = _write_state(tmp_path)
    args = Namespace(method_options={'arm': 'C', 'v3_state': str(state_path), 'k': K, **options},
                     seed=seed)
    if edge_direction is not None:
        args.edge_direction = edge_direction
    torch.manual_seed(seed)
    return METHOD_REGISTRY['cei_gnn_v3'](
        num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=edge_dim, num_classes=CLASSES,
        hidden=hidden, layers=layers, dropout=0.0, token_dim=TOKEN_DIM, num_triples=num_triples,
        args=args)


def _fixture_dimensions(**overrides):
    dims = dict(FIXTURE_DIMENSIONS)
    dims.update(overrides)
    return dims


# ------------------------------------------------------------------ step 1: size arms

def test_analytic_counts_reproduce_the_recorded_v2_total_and_the_spec_arm_table():
    dims = arm_guards.V2_REFERENCE_DIMENSIONS
    assert dims['hidden'] == 128 and dims['num_relations'] == 15 and dims['num_triples'] == 16
    assert arm_guards.V2_REFERENCE_PARAMETER_COUNT == SPEC_V2_TOTAL
    assert arm_guards.analytic_parameter_count(dims) == SPEC_V2_TOTAL
    assert arm_guards.ANALYTIC_V2_SCHEMA_DELTAS == SPEC_DELTAS
    for arm, delta in SPEC_DELTAS.items():
        assert arm_guards.analytic_arm_delta(arm, dims) == delta, arm
        assert SPEC_V2_TOTAL + delta == SPEC_TOTALS[arm], arm
    assert arm_guards.analytic_arm_delta('C', dims) == 0
    # The v3 blocks (PLE projection, absence tables) are added on top of the v2 schema and
    # do not change the deltas of E2d, E6a and E6b; E2w's delta grows with the PLE width.
    assert arm_guards.analytic_parameter_count(dims, K=8, universe_size=50) == (
        SPEC_V2_TOTAL + 9 * 128 + 2 * 50 * 10)
    for arm in ('E2d', 'E6a', 'E6b'):
        assert arm_guards.analytic_arm_delta(arm, dims, K=8, universe_size=50) == SPEC_DELTAS[arm]
    assert arm_guards.analytic_arm_delta('E2w', dims, K=8, universe_size=50) == SPEC_DELTAS['E2w'] + 9 * 128
    with pytest.raises(ValueError, match='arm'):
        arm_guards.analytic_arm_delta('O', dims)


def test_analytic_count_equals_parameter_count_of_the_v3_network_on_fixtures():
    for hidden in (HIDDEN, 16):
        for depth in (1, 2):
            network = _network(hidden=hidden, encoder_depth=depth)
            expected = arm_guards.analytic_parameter_count(
                _fixture_dimensions(hidden=hidden, pair_rank=4), K=K, universe_size=UNIVERSE_SIZE,
                encoder_depth=depth)
            assert _count(network) == expected, (hidden, depth)
    widened = _network(num_relations=RELATIONS_BIDIRECTIONAL, num_triples=2 * (TRIPLES - 1) + 1,
                       edge_dim=EDGE_DIM + len(REVERSE_RELATIONS))
    assert _count(widened) == arm_guards.analytic_parameter_count(
        _fixture_dimensions(pair_rank=4, num_relations=RELATIONS_BIDIRECTIONAL,
                            num_triples=2 * (TRIPLES - 1) + 1,
                            edge_dim=EDGE_DIM + len(REVERSE_RELATIONS)),
        K=K, universe_size=UNIVERSE_SIZE)
    for bad in ({'hidden': 0}, {'num_relations': -1}, {'pair_rank': 1.5}):
        with pytest.raises(ValueError):
            arm_guards.analytic_parameter_count(_fixture_dimensions(**bad), K=K, universe_size=1)
    with pytest.raises(ValueError, match='encoder_depth'):
        arm_guards.analytic_parameter_count(_fixture_dimensions(), K=K, universe_size=1, encoder_depth=0)


def test_e2w_and_e2d_analytic_deltas_equal_adapter_parameter_count_differences(tmp_path):
    control = _adapter(tmp_path)
    control_total = parameter_count(control)
    assert control.run_config()['parameter_count'] == control_total
    dims = _fixture_dimensions(hidden=CONTROL_HIDDEN)
    assert control_total == arm_guards.analytic_parameter_count(dims, K=K, universe_size=UNIVERSE_SIZE)
    wide = _adapter(tmp_path, hidden=256)
    assert parameter_count(wide) - control_total == arm_guards.analytic_arm_delta(
        'E2w', dims, K=K, universe_size=UNIVERSE_SIZE)
    deep = _adapter(tmp_path, layers=2)
    assert parameter_count(deep) - control_total == arm_guards.analytic_arm_delta(
        'E2d', dims, K=K, universe_size=UNIVERSE_SIZE)
    assert parameter_count(deep) - control_total == BLOCK_DELTA_AT_128
    # Inventory diff against C (§1, E17): E2w widens every tensor, E2d adds exactly one block.
    control_inventory = control.run_config()['parameter_inventory']
    wide_inventory, deep_inventory = wide.run_config()['parameter_inventory'], deep.run_config()['parameter_inventory']
    assert set(wide_inventory) == set(control_inventory)
    assert set(deep_inventory) - set(control_inventory) == {
        'encoder_blocks.0.linear.weight', 'encoder_blocks.0.linear.bias',
        'encoder_blocks.0.norm.weight', 'encoder_blocks.0.norm.bias'}
    assert all(deep_inventory[name] == control_inventory[name] for name in control_inventory)
    diff = arm_guards.inventory_diff(control_inventory, wide_inventory)
    assert diff['added'] == [] and diff['removed'] == []
    assert set(diff['widened']) == {name for name in control_inventory
                                    if wide_inventory[name]['shape'] != control_inventory[name]['shape']}
    assert 'token_embedding.weight' in diff['unchanged'] and 'bias' in diff['unchanged']
    assert 'node_head.weight' in diff['widened'] and 'ple_projection.weight' in diff['widened']
    diff = arm_guards.inventory_diff(control_inventory, deep_inventory)
    assert diff['widened'] == [] and set(diff['added']) == set(deep_inventory) - set(control_inventory)
    assert sorted(diff['unchanged']) == sorted(control_inventory)


def test_run_config_records_the_size_arm_definition_fields(tmp_path):
    control = _adapter(tmp_path).run_config()
    wide, deep = _adapter(tmp_path, hidden=256).run_config(), _adapter(tmp_path, layers=2).run_config()
    assert (control['hidden'], control['encoder_depth'], control['common_init_identical_to_c']) == (128, 1, True)
    assert (wide['hidden'], wide['encoder_depth'], wide['common_init_identical_to_c']) == (256, 1, False)
    assert (deep['hidden'], deep['encoder_depth'], deep['common_init_identical_to_c']) == (128, 2, True)
    for config in (control, wide, deep):
        assert config['arm'] == 'C' and config['edge_direction'] == 'forward'
        assert config['comorbid_block'] == 0 and config['pair_mode'] == 'additive'
        assert config['parameter_count'] == config['active_parameter_count'] + config['inactive_parameter_count']
        assert config['architecture']['encoder_depth'] == config['encoder_depth']
        assert config['architecture']['hidden'] == config['hidden']
    assert wide['architecture']['layers'] == 1 and deep['architecture']['layers'] == 2
    assert arm_guards.arm_of_run_config(control) == 'C'
    assert arm_guards.arm_of_run_config(wide) == 'E2w'
    assert arm_guards.arm_of_run_config(deep) == 'E2d'
    assert arm_guards.ARM_DEFINITIONS['E2w'] == {'hidden': 256, 'encoder_depth': 1,
                                                 'edge_direction': 'forward', 'comorbid_block': 0,
                                                 'common_init_identical_to_c': False}
    assert arm_guards.ARM_DEFINITIONS['E2d'] == {'hidden': 128, 'encoder_depth': 2,
                                                 'edge_direction': 'forward', 'comorbid_block': 0,
                                                 'common_init_identical_to_c': True}
    assert set(arm_guards.ARM_DEFINITIONS) == {'C', 'E2w', 'E2d', 'E6a', 'E6b'}
    # A configuration outside the arm table is never mapped to an arm.
    with pytest.raises(ValueError, match='arm'):
        arm_guards.arm_of_run_config(_adapter(tmp_path, hidden=256, layers=2).run_config())
    with pytest.raises(ValueError, match='arm'):
        arm_guards.arm_of_run_config(_adapter(tmp_path, hidden=64).run_config())
    with pytest.raises(ValueError, match='arm'):
        arm_guards.arm_of_run_config({**control, 'arm': 'A'})


def test_v3_adapter_accepts_layers_2_and_the_v2_adapter_still_refuses_it(tmp_path):
    assert _adapter(tmp_path, layers=2).network.encoder_depth == 2
    v2 = METHOD_REGISTRY['cei_gnn_v2']
    common = dict(num_tokens=NUM_TOKENS, node_dim=NODE_DIM, edge_dim=EDGE_DIM, num_classes=CLASSES,
                  hidden=CONTROL_HIDDEN, dropout=0.0, token_dim=TOKEN_DIM, num_triples=TRIPLES,
                  args=Namespace(method_options={'pair_mode': 'additive'}, seed=1234))
    with pytest.raises(ValueError, match='layers=1 only'):
        v2(layers=2, **common)
    assert v2(layers=1, **common).layers_count == 1
    # The v2 adapter has no depth option either: the runner cannot reach E2d through v2.
    with pytest.raises(ValueError, match='unknown method option'):
        v2(layers=1, **{**common, 'args': Namespace(
            method_options={'pair_mode': 'additive', 'encoder_depth': 2}, seed=1234)})
    assert arm_guards.assert_size_arm_support() == {'cei_gnn_v3': True, 'cei_gnn_v2': False}


# ------------------------------------------------------------ step 2: bidirectional arm

def _fixture_graph(sample_id='fixture-1'):
    """One contract-valid graph document: patient, index visit, complaint, vital, lab +
    analyte, two prior diagnoses linked by a symmetric `comorbid_with` pair (graph.py lines
    254–259). Reversible edges: reports_complaint, observed_vital, measured_in, instance_of."""
    nodes = [
        {'id': 'p', 'kind': 'patient', 'token': 'patient', 'age': 40, 'gender': 'F', 'race': 'A',
         'arrival_transport': 'WALK IN'},
        {'id': 'v', 'kind': 'visit', 'token': 'visit:index', 'acuity': 2},
        {'id': 'c', 'kind': 'complaint', 'token': 'cc:chest pain'},
        {'id': 'hr', 'kind': 'vital', 'token': 'vital:heartrate', 'unit': 'bpm', 'value': 70.0,
         'time_hours': -1.0, 'available_hours': -1.0},
        {'id': 'm', 'kind': 'measurement', 'token': 'lab:creatinine', 'unit': 'mg/dL', 'value': 1.4,
         'time_hours': -2.0, 'available_hours': -1.5},
        {'id': 'a', 'kind': 'analyte', 'token': 'lab:creatinine'},
        {'id': 'd1', 'kind': 'diagnosis', 'token': 'dx:I10'},
        {'id': 'd2', 'kind': 'diagnosis', 'token': 'dx:E11'},
    ]
    edges = [
        {'source': 'p', 'target': 'v', 'relation': 'has_visit', 'informative': False},
        {'source': 'v', 'target': 'p', 'relation': 'index_visit_of', 'informative': False},
        {'source': 'v', 'target': 'c', 'relation': 'reports_complaint', 'informative': False},
        {'source': 'v', 'target': 'hr', 'relation': 'observed_vital', 'informative': False},
        {'source': 'v', 'target': 'm', 'relation': 'measured_in', 'informative': False},
        {'source': 'm', 'target': 'a', 'relation': 'instance_of', 'informative': False},
        {'source': 'v', 'target': 'd1', 'relation': 'has_prior_diagnosis', 'informative': False},
        {'source': 'v', 'target': 'd2', 'relation': 'has_prior_diagnosis', 'informative': False},
        {'source': 'd1', 'target': 'd2', 'relation': 'comorbid_with', 'informative': True},
        {'source': 'd2', 'target': 'd1', 'relation': 'comorbid_with', 'informative': True},
    ]
    return {'sample_id': sample_id, 'nodes': nodes, 'edges': edges,
            'coverage': {'complaints': 1, 'index_measurements': 1, 'prior_visits': 1,
                         'informative_edges': 2}}


def _membership(graph):
    kinds = [node['kind'] for node in graph['nodes']]
    return {'contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION, 'sample_id': graph['sample_id'],
            'visit_ordinals': [0],
            'membership_pairs': [[0, i] for i, kind in enumerate(kinds) if kind != 'patient'],
            'global_node_mask': [kind == 'patient' for kind in kinds]}


def _fit(tmp_path, graphs):
    graphs_path, membership_path = tmp_path / 'graphs.jsonl', tmp_path / VISIT_MEMBERSHIP_FILENAME
    graphs_path.write_text(''.join(json.dumps(g) + '\n' for g in graphs))
    membership_path.write_text(''.join(json.dumps(_membership(g)) + '\n' for g in graphs))
    return tz.fit_preprocessing(graphs_path, {g['sample_id'] for g in graphs}, token_min_count=1,
                                membership_path=membership_path)


def _views(tmp_path):
    graph = _fixture_graph()
    prep = _fit(tmp_path, [graph])
    forward = tz.encode_graph(graph, prep, _membership(graph), edge_direction='forward')
    both = tz.encode_graph(graph, prep, _membership(graph), edge_direction='bidirectional')
    return graph, prep, forward, both


def test_bidirectional_tensorisation_appends_the_reverse_block_after_the_forward_block(tmp_path):
    graph, prep, forward, both = _views(tmp_path)
    reversible = [i for i, e in enumerate(graph['edges']) if e['relation'] in tz.REVERSIBLE_RELATIONS]
    F, R = forward.num_edges, len(reversible)
    assert (F, R) == (10, 4)
    assert both.num_edges == F + R
    # Forward ids are stable: the first F entries are the forward view verbatim.
    for field in ('edge_index', 'edge_relation', 'edge_triple', 'edge_payload'):
        assert torch.equal(getattr(both, field)[..., :F] if field == 'edge_index'
                           else getattr(both, field)[:F], getattr(forward, field)), field
    assert torch.equal(both.edge_attr[:F, :RELATIONS_FORWARD], forward.edge_attr[:, :RELATIONS_FORWARD])
    assert torch.equal(both.edge_attr[:F, RELATIONS_BIDIRECTIONAL:], forward.edge_attr[:, RELATIONS_FORWARD:])
    # Reverse block: endpoint-swapped copies of exactly the reversible forward edges, in order.
    assert torch.equal(both.edge_index[:, F:], forward.edge_index[:, reversible].flip(0))
    names = tz.relation_vocabulary('bidirectional')
    assert [names[r] for r in both.edge_relation[F:].tolist()] == [
        'rev:' + graph['edges'][i]['relation'] for i in reversible]
    assert both.edge_relation[F:].min() >= RELATIONS_FORWARD
    assert torch.equal(both.edge_triple[F:], forward.edge_triple[reversible] + len(prep['triples']))
    # comorbid_with is symmetric already and is NOT reversed (tensorize.py line 39).
    comorbid = ALL_RELATIONS.index('comorbid_with')
    assert int((forward.edge_relation == comorbid).sum()) == 2
    assert not any(names[r] == 'rev:comorbid_with' for r in both.edge_relation[F:].tolist())
    record = arm_guards.edge_view_record(both.edge_relation, 'bidirectional')
    assert record == {'edge_direction': 'bidirectional', 'forward_edges': F, 'reverse_edges': R,
                      'mask_length': F + R, 'reverse_of_forward': reversible}
    assert arm_guards.edge_view_record(forward.edge_relation, 'forward') == {
        'edge_direction': 'forward', 'forward_edges': F, 'reverse_edges': 0, 'mask_length': F,
        'reverse_of_forward': []}
    # A forward-view edge list carrying a reverse id, or a reverse block not strictly after the
    # forward block, is refused.
    with pytest.raises(ValueError, match='reverse'):
        arm_guards.edge_view_record(both.edge_relation, 'forward')
    shuffled = torch.cat((both.edge_relation[F:], both.edge_relation[:F]))
    with pytest.raises(ValueError, match='after the forward block'):
        arm_guards.edge_view_record(shuffled, 'bidirectional')
    with pytest.raises(ValueError, match='edge_direction'):
        arm_guards.edge_view_record(forward.edge_relation, 'sideways')


def _e6a_graph():
    """The U3 fixture graph re-expressed under the bidirectional view: F = 6 forward edges
    plus R = 5 reverse edges (every edge except has_visit is reversible: reports_complaint,
    measured_in, observed_vital, instance_of ×2)."""
    graph = _graph()
    F = graph.num_edges
    reversible = [i for i, r in enumerate(graph.edge_relation.tolist())
                  if ALL_RELATIONS[r] in tz.REVERSIBLE_RELATIONS]
    assert reversible == [1, 2, 3, 4, 5]
    rev_ids = [RELATIONS_FORWARD + REVERSE_RELATIONS.index('rev:' + ALL_RELATIONS[graph.edge_relation[i]])
               for i in reversible]
    graph.edge_index = torch.cat((graph.edge_index, graph.edge_index[:, reversible].flip(0)), dim=1)
    graph.edge_relation = torch.cat((graph.edge_relation, torch.tensor(rev_ids)))
    graph.edge_triple = torch.cat((graph.edge_triple, graph.edge_triple[reversible] + (TRIPLES - 1)))
    attr = torch.zeros(F + len(reversible), E6A_SHAPES['edge_dim'])
    attr[:F, :EDGE_DIM] = graph.edge_attr
    attr[F:, :EDGE_DIM] = graph.edge_attr[reversible]
    graph.edge_attr = attr
    return graph, F, len(reversible)


def test_edge_mask_of_length_f_plus_r_is_accepted_and_length_f_is_rejected():
    graph, F, R = _e6a_graph()
    torch.manual_seed(123)
    network = _network(seed=123, control_shapes=E6A_CONTROL, **E6A_SHAPES).eval()
    _randomise_gates(network)
    metadata = read_clinical_batch(graph, method='test', node_dim=NODE_DIM,
                                   edge_dim=E6A_SHAPES['edge_dim'], num_tokens=NUM_TOKENS,
                                   num_triples=E6A_SHAPES['num_triples'],
                                   num_relations=RELATIONS_BIDIRECTIONAL)
    features = network.continuous_inputs(metadata)

    def run():
        return network.forward_continuous(features, metadata.edge_index, metadata,
                                          graph.visit_membership_index, return_parts=True,
                                          visit_graph=_visit_graph(graph))

    record = arm_guards.edge_view_record(graph.edge_relation, 'bidirectional')
    assert record['mask_length'] == F + R
    ordinary = run()
    assert tuple(ordinary['edge_contributions'].shape) == (F + R, CLASSES)
    full = torch.ones(F + R)
    arm_guards.assert_edge_mask_length(full, record)
    network.set_edge_mask(full)
    torch.testing.assert_close(run()['logits'], ordinary['logits'])
    network.set_edge_mask(None)
    # A forward-length mask is refused by the guard and by the network itself.
    with pytest.raises(ValueError, match=f'{F} .*{F + R}|{F + R}'):
        arm_guards.assert_edge_mask_length(torch.ones(F), record)
    network.set_edge_mask(torch.ones(F))
    with pytest.raises(ValueError, match='edge mask length'):
        run()
    network.set_edge_mask(None)
    for bad in (torch.ones(F + R, 1), torch.ones(F + R + 1), torch.full((F + R,), 2.0)):
        with pytest.raises(ValueError):
            arm_guards.assert_edge_mask_length(bad, record)
    # Masking a forward edge to zero does not mask its reverse twin (spec §5.2).
    mask = torch.ones(F + R)
    mask[1] = 0.0                      # forward reports_complaint (reverse twin sits at F + 0)
    arm_guards.assert_edge_mask_length(mask, record)
    network.set_edge_mask(mask)
    masked = run()
    network.set_edge_mask(None)
    assert torch.count_nonzero(masked['edge_contributions'][1]) == 0
    assert torch.count_nonzero(masked['edge_contributions'][F]) > 0
    torch.testing.assert_close(masked['edge_contributions'][F], ordinary['edge_contributions'][F])


def test_reverse_edge_attribution_is_kept_separate_from_its_forward_edge():
    graph, F, R = _e6a_graph()
    torch.manual_seed(123)
    network = _network(seed=123, control_shapes=E6A_CONTROL, **E6A_SHAPES).eval()
    _randomise_gates(network)
    metadata = read_clinical_batch(graph, method='test', node_dim=NODE_DIM,
                                   edge_dim=E6A_SHAPES['edge_dim'], num_tokens=NUM_TOKENS,
                                   num_triples=E6A_SHAPES['num_triples'],
                                   num_relations=RELATIONS_BIDIRECTIONAL)
    parts = network.forward_continuous(network.continuous_inputs(metadata), metadata.edge_index,
                                       metadata, graph.visit_membership_index, return_parts=True,
                                       visit_graph=_visit_graph(graph))
    contributions = parts['edge_contributions']
    split = arm_guards.reverse_edge_attribution(contributions, graph.edge_relation, 'bidirectional')
    assert tuple(split['forward'].shape) == (F, CLASSES) and tuple(split['reverse'].shape) == (R, CLASSES)
    assert torch.equal(split['forward'], contributions[:F])
    assert torch.equal(split['reverse'], contributions[F:])
    assert split['reverse_relations'] == ['rev:reports_complaint', 'rev:measured_in',
                                          'rev:observed_vital', 'rev:instance_of', 'rev:instance_of']
    assert split['reverse_of_forward'] == [1, 2, 3, 4, 5]
    # A reverse edge has its own relation/triple embedding, so its vote differs from its twin.
    assert not torch.allclose(contributions[F], contributions[1])
    assert not torch.allclose(contributions[F + 1], contributions[2])
    # The forward + reverse sum is only available as a derived, labelled quantity.
    derived = split['derived_forward_plus_reverse']
    assert derived['label'] == 'derived: forward + reverse (not a model attribution)'
    assert tuple(derived['values'].shape) == (R, CLASSES)
    torch.testing.assert_close(derived['values'], contributions[1:F] + contributions[F:])
    # Under the forward view there is no reverse block and no derived sum.
    forward_graph = _graph()
    torch.manual_seed(123)
    control = _network(seed=123).eval()
    _share_weights(network, control)
    control_parts = _run(control, forward_graph)
    split = arm_guards.reverse_edge_attribution(control_parts['edge_contributions'],
                                                forward_graph.edge_relation, 'forward')
    assert torch.equal(split['forward'], control_parts['edge_contributions'])
    assert split['reverse'].numel() == 0 and split['reverse_relations'] == []
    assert split['derived_forward_plus_reverse'] is None
    with pytest.raises(ValueError, match='edge_contributions'):
        arm_guards.reverse_edge_attribution(contributions[:F], graph.edge_relation, 'bidirectional')


def test_e6a_adapter_widens_only_the_relation_tensors_and_records_the_arm(tmp_path):
    e6a = _adapter(tmp_path, edge_direction='bidirectional', num_triples=E6A_SHAPES['num_triples'],
                   edge_dim=RELATIONS_BIDIRECTIONAL + PAYLOAD_WIDTH)
    control = _adapter(tmp_path, edge_dim=RELATIONS_FORWARD + PAYLOAD_WIDTH)
    config, control_config = e6a.run_config(), control.run_config()
    assert config['edge_direction'] == 'bidirectional' and config['widened_tensors'] == list(E6A_WIDENED)
    assert config['common_init_identical_to_c'] is True
    assert arm_guards.arm_of_run_config(config) == 'E6a'
    assert arm_guards.arm_of_run_config(control_config) == 'C'
    dims = _fixture_dimensions(hidden=CONTROL_HIDDEN, edge_dim=RELATIONS_FORWARD + PAYLOAD_WIDTH)
    assert config['parameter_count'] - control_config['parameter_count'] == arm_guards.analytic_arm_delta(
        'E6a', dims, K=K, universe_size=UNIVERSE_SIZE)
    diff = arm_guards.inventory_diff(control_config['parameter_inventory'], config['parameter_inventory'])
    assert diff['added'] == [] and diff['removed'] == []
    assert diff['widened'] == sorted(E6A_WIDENED)
    assert len(diff['unchanged']) == len(control_config['parameter_inventory']) - len(E6A_WIDENED)
    # Preprocessing state is direction-free (§5.2): the same v3_state, K and hashes as C.
    for key in ('preprocessing_sha256', 'v3_state_sha256', 'knot_table_sha256', 'universe_sha256', 'k'):
        assert config[key] == control_config[key], key
