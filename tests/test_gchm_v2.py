"""Focused synthetic tests for the GCHM-PNA v2 work (edge view, dev split, protocol).

Everything runs on tiny in-memory or tmp_path fixtures that satisfy the current
visit-membership contract. No real patient data, no benchmark training, no test fold.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from core import aggregate, train
from core import tensorize as tz
from core.contracts import (
    VISIT_MEMBERSHIP_CONTRACT_VERSION, VISIT_MEMBERSHIP_FILENAME)
from gchm_pna.gchm_v2 import GCHMv2, HUB_KIND
from core.registry import build_method
from core.method_base import relation_count
from core.schema import sha256

# ------------------------------------------------------------------ fixtures


def clinical_graph(sample_id, value=70.0, *, extra_measurement=True, prior_visits=0):
    """One contract-valid graph: patient, index visit, complaint, vital, lab + analyte."""
    nodes = [
        {'id': 'p', 'kind': 'patient', 'token': 'patient', 'age': 40,
         'gender': 'F', 'race': 'A', 'arrival_transport': 'WALK IN'},
        {'id': 'v', 'kind': 'visit', 'token': 'visit:index', 'acuity': 2},
        {'id': 'c', 'kind': 'complaint', 'token': 'cc:chest pain'},
        {'id': 'hr', 'kind': 'vital', 'token': 'vital:heartrate', 'unit': 'bpm',
         'value': value, 'time_hours': -1.0, 'available_hours': -1.0},
    ]
    edges = [
        {'source': 'p', 'target': 'v', 'relation': 'has_visit', 'informative': False},
        {'source': 'v', 'target': 'p', 'relation': 'index_visit_of', 'informative': False},
        {'source': 'v', 'target': 'c', 'relation': 'reports_complaint', 'informative': False},
        {'source': 'v', 'target': 'hr', 'relation': 'observed_vital', 'informative': False},
    ]
    if extra_measurement:
        nodes += [{'id': 'm', 'kind': 'measurement', 'token': 'lab:creatinine', 'unit': 'mg/dL',
                   'value': value / 50, 'time_hours': -2.0, 'available_hours': -1.5},
                  {'id': 'a', 'kind': 'analyte', 'token': 'lab:creatinine'}]
        edges += [{'source': 'v', 'target': 'm', 'relation': 'measured_in', 'informative': False},
                  {'source': 'm', 'target': 'a', 'relation': 'instance_of', 'informative': False}]
    return {'sample_id': sample_id, 'nodes': nodes, 'edges': edges,
            'coverage': {'complaints': 1, 'index_measurements': int(extra_measurement),
                         'prior_visits': prior_visits, 'informative_edges': 0}}


def membership(graph):
    """Sidecar row: patient global, every other node in the single index visit."""
    kinds = [node['kind'] for node in graph['nodes']]
    return {'contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
            'sample_id': graph['sample_id'], 'visit_ordinals': [0],
            'membership_pairs': [[0, i] for i, kind in enumerate(kinds) if kind != 'patient'],
            'global_node_mask': [kind == 'patient' for kind in kinds]}


def fit(tmp_path, graphs, train_ids=None):
    graphs_path, membership_path = tmp_path / 'graphs.jsonl', tmp_path / VISIT_MEMBERSHIP_FILENAME
    graphs_path.write_text(''.join(json.dumps(g) + '\n' for g in graphs))
    membership_path.write_text(''.join(json.dumps(membership(g)) + '\n' for g in graphs))
    ids = set(train_ids) if train_ids is not None else {g['sample_id'] for g in graphs}
    return tz.fit_preprocessing(graphs_path, ids, token_min_count=1, membership_path=membership_path)


def artifact(tmp_path, count=24, test_rows=2):
    """Contract-valid artifact + bound target sidecar; one patient per sample id."""
    root = tmp_path / 'artifact'
    root.mkdir()
    graphs = [clinical_graph(f'sample-{i:02d}', 55.0 + 3 * i) for i in range(count)]
    splits = (['train'] * (count - 4 - test_rows) + ['validation'] * 4 + ['test'] * test_rows)
    graphs_path, membership_path = root / 'graphs.jsonl', root / VISIT_MEMBERSHIP_FILENAME
    graphs_path.write_text(''.join(json.dumps(g) + '\n' for g in graphs))
    membership_path.write_text(''.join(json.dumps(membership(g)) + '\n' for g in graphs))
    labels = [f'synthetic_label_{i}' for i in range(train.NUM_CLASSES)]
    canonical = tmp_path / 'canonical.json'
    canonical.write_text(json.dumps({'classes': labels}))
    manifest = {'status': 'completed', 'schema_version': 'clinical_graph_v2',
                'logic_contract_version': 'clinical_graph_logic_v2', 'temporal_clean': False,
                'graphs_sha256': sha256(graphs_path), 'inherited_cohort_sha256': 'synthetic',
                'visit_membership_contract_version': VISIT_MEMBERSHIP_CONTRACT_VERSION,
                'visit_membership_file': VISIT_MEMBERSHIP_FILENAME,
                'visit_membership_rows': count,
                'visit_membership_sha256': sha256(membership_path),
                'limitations': ['Synthetic fixture; not a research result.']}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    targets = tmp_path / 'targets.csv'
    targets.write_text('sample_id,target,split,subject_id\n' + ''.join(
        f'sample-{i:02d},{i % 2},{split},person-{i:02d}\n' for i, split in enumerate(splits)))
    (tmp_path / 'binding_manifest.json').write_text(json.dumps(
        {'status': 'completed', 'labels': labels, 'cohort_sha256': 'synthetic',
         'artifact_files': {'targets.csv': sha256(targets)}}))
    return root, targets, canonical


def run_args(root, targets, canonical, output, **overrides):
    # Like the protocol, dev runs draw a bounded TRAIN sample (12 of 18 rows here) so
    # unused TRAIN patients remain for the patient-disjoint dev split.
    values = dict(artifact=str(root), targets=str(targets), canonical=str(canonical),
                  output=str(output), method='clinical_gnn', conv='gchm_v2', seed=7,
                  top_k_labels=2, drop_relation=None, rewire_relation=None, edges='all',
                  train_limit=12, token_min_count=1, min_prior_visits=0, device='cpu',
                  batch_size=4, hidden=8, layers=1, dropout=0.0, token_dim=4, heads=2,
                  modulation='multiplicative', weights='none', lr=0.01, epochs=2,
                  no_edge_payload=False, no_message_passing=False,
                  edge_direction='bidirectional', selection_fold='dev', dev_limit=3,
                  final_eval='validation', sample_seed=None)
    values.update(overrides)
    return argparse.Namespace(**values)


# ------------------------------------------------------------- edge view


def test_forward_view_is_the_historical_default_and_bidirectional_appends_mirrors(tmp_path):
    graphs = [clinical_graph('a'), clinical_graph('b', 90.0)]
    prep = fit(tmp_path, graphs)
    record = membership(graphs[1])
    default = tz.encode_graph(graphs[1], prep, record)
    forward = tz.encode_graph(graphs[1], prep, record, edge_direction='forward')
    both = tz.encode_graph(graphs[1], prep, record, edge_direction='bidirectional')
    for field in ('x', 'edge_index', 'edge_attr', 'edge_relation', 'edge_triple', 'edge_payload'):
        assert torch.equal(getattr(default, field), getattr(forward, field)), field
    n, width = forward.edge_index.size(1), len(tz.ALL_RELATIONS)
    reversible = [i for i, e in enumerate(graphs[1]['edges'])
                  if e['relation'] in tz.REVERSIBLE_RELATIONS]
    assert both.edge_index.size(1) == n + len(reversible) == n + 4
    assert torch.equal(both.edge_index[:, :n], forward.edge_index)
    assert torch.equal(both.edge_index[:, n:], forward.edge_index[:, reversible].flip(0))
    assert torch.equal(both.edge_relation[:n], forward.edge_relation)
    assert both.edge_relation[n:].min() >= width
    names = tz.relation_vocabulary('bidirectional')
    assert [names[r] for r in both.edge_relation[n:].tolist()] == [
        'rev:' + graphs[1]['edges'][i]['relation'] for i in reversible]
    assert torch.equal(both.edge_payload[n:], forward.edge_payload[reversible])
    fitted = len(prep['triples'])
    assert torch.equal(both.edge_triple[n:], forward.edge_triple[reversible] + fitted)
    assert both.edge_attr.size(1) == len(names) + tz.PAYLOAD_WIDTH
    for field in ('x', 'token', 'node_type', 'visit_membership_index', 'global_node_mask'):
        assert torch.equal(getattr(both, field), getattr(forward, field)), field


def test_bidirectional_view_is_what_lets_evidence_reach_the_visit_hub(tmp_path):
    graph = clinical_graph('a')
    prep = fit(tmp_path, [graph])
    evidence = {tz.KIND_INDEX[k] for k in ('complaint', 'vital', 'measurement')}
    hub = tz.KIND_INDEX['visit']
    for view, expected in (('forward', set()), ('bidirectional', evidence)):
        data = tz.encode_graph(graph, prep, membership(graph), edge_direction=view)
        source, target = data.edge_index
        senders = set(data.node_type[source[data.node_type[target] == hub]].tolist())
        assert senders & evidence == expected, view


def test_unseen_triple_stays_unseen_in_reverse_and_views_validate(tmp_path):
    graph = clinical_graph('a', extra_measurement=False)
    prep = fit(tmp_path, [graph])
    novel = clinical_graph('b')  # measured_in/instance_of triples were never fitted
    data = tz.encode_graph(novel, prep, membership(novel), edge_direction='bidirectional')
    names = tz.relation_vocabulary('bidirectional')
    for relation, triple in zip(data.edge_relation.tolist(), data.edge_triple.tolist()):
        if names[relation] in ('measured_in', 'instance_of', 'rev:measured_in', 'rev:instance_of'):
            assert triple == tz.TRIPLE_UNK
    assert tz.triple_count(prep, 'bidirectional') == 2 * len(prep['triples']) + 1
    assert tz.edge_feature_layout(prep, 'forward') == tz.preprocessing_state(prep)[
        'edge_feature_layout']
    with pytest.raises(ValueError, match='edge_direction'):
        tz.relation_vocabulary('sideways')
    with pytest.raises(ValueError, match='edge_direction'):
        tz.encode_graph(graph, prep, membership(graph), edge_direction='sideways')


@pytest.mark.parametrize('method', ['protgnn', 'gsat', 'graphcare'])
def test_every_adapter_consumes_the_bidirectional_view(tmp_path, method):
    graphs = [clinical_graph(f's{i}', 60.0 + 10 * i) for i in range(3)]
    prep = fit(tmp_path, graphs)
    data = [tz.encode_graph(g, prep, membership(g), edge_direction='bidirectional')
            for g in graphs]
    for i, d in enumerate(data):
        d.y = torch.tensor([i % 2])
    args = argparse.Namespace(edge_direction='bidirectional', epochs=2, lr=1e-3,
                              weight_decay=0.0, batch_size=2, patience=2, min_delta=0.0)
    assert relation_count(args) == len(tz.relation_vocabulary('bidirectional'))
    torch.manual_seed(3)
    model = build_method(method, num_tokens=len(prep['vocabulary']), node_dim=data[0].x.size(1),
                         edge_dim=data[0].edge_attr.size(1), num_classes=2, hidden=8, layers=1,
                         dropout=0.0, token_dim=4,
                         num_triples=tz.triple_count(prep, 'bidirectional'), args=args)
    assert model.run_config()['architecture']['num_relations'] == relation_count(args)
    output = model(Batch.from_data_list(data), epoch=0)
    loss = torch.nn.functional.cross_entropy(output.logits, torch.tensor([0, 1, 0]))
    (loss + output.auxiliary_loss).backward()
    assert torch.isfinite(output.logits).all()
    forward_model = build_method(method, num_tokens=len(prep['vocabulary']),
                                 node_dim=data[0].x.size(1), edge_dim=data[0].edge_attr.size(1),
                                 num_classes=2, hidden=8, layers=1, dropout=0.0, token_dim=4,
                                 num_triples=tz.triple_count(prep, 'bidirectional'),
                                 args=argparse.Namespace(edge_direction='forward'))
    with pytest.raises(ValueError, match='relation'):
        forward_model(Batch.from_data_list(data), epoch=0)


# --------------------------------------------------------------- dev split


def test_dev_split_is_patient_disjoint_deterministic_and_seeded():
    targets = {f's{i:03d}': (i % 2, 'train', f'p{i // 2:03d}') for i in range(120)}
    targets.update({f'v{i}': (0, 'validation', f'val{i}') for i in range(5)})
    train_ids = train.sample_train_ids(targets, 40, 11)
    dev = train.select_dev_ids(targets, train_ids, 20, 11)
    assert len(dev) == 20 and not dev & train_ids
    assert all(targets[s][1] == 'train' for s in dev)
    sample_patients = {targets[s][2] for s in train_ids}
    assert not {targets[s][2] for s in dev} & sample_patients
    assert dev == train.select_dev_ids(targets, train_ids, 20, 11)
    assert dev != train.select_dev_ids(targets, train_ids, 20, 12)
    eligible = {s for s in targets if s.startswith('s') and int(s[1:]) % 3 == 0}
    assert train.select_dev_ids(targets, train_ids, 5, 11, eligible) <= eligible
    assert train.select_dev_ids(targets, train_ids, None, 11) == frozenset()
    with pytest.raises(ValueError, match='patient-disjoint'):
        train.select_dev_ids(targets, train_ids, 500, 11)


def test_cli_normalisation_guards_the_new_flags(tmp_path):
    parser = train.parser()
    base = ['--artifact', 'a', '--targets', 't', '--output', str(tmp_path / 'out')]
    legacy = train.normalize_method_args(parser.parse_args(base), parser)
    assert (legacy.edge_direction, legacy.selection_fold, legacy.final_eval) == (
        'forward', 'validation', 'validation')
    assert legacy.dev_limit is None and legacy.hidden == 96  # historical profile unchanged
    v2 = train.normalize_method_args(parser.parse_args(base + ['--conv', 'gchm_v2']), parser)
    assert (v2.hidden, v2.dropout, v2.weight_decay, v2.epochs) == (92, 0.3, 1e-4, 40)
    assert (v2.aggregation, v2.readout) == ('pna', 'hub')
    for bad in (['--aggregation', 'sum'], ['--no-hub-gate'], ['--readout', 'pool'],
                ['--final-eval', 'none'], ['--selection-fold', 'dev'], ['--dev-limit', '3'],
                ['--conv', 'gchm_v2', '--dev-limit', '0', '--selection-fold', 'dev']):
        with pytest.raises(SystemExit):
            train.normalize_method_args(parser.parse_args(base + bad), parser)
    assert not (tmp_path / 'out').exists()


# ---------------------------------------------------------- runner, end to end


def test_dev_split_fails_closed_when_no_unused_patients_remain(tmp_path):
    root, targets, canonical = artifact(tmp_path)
    with pytest.raises(ValueError, match='patient-disjoint'):
        train.run(run_args(root, targets, canonical, tmp_path / 'run', train_limit=None))


def test_v2_dev_run_reads_validation_once_and_binds_the_new_contract(tmp_path):
    root, targets, canonical = artifact(tmp_path)
    output = tmp_path / 'run'
    train.run(run_args(root, targets, canonical, output))
    binding = json.loads((output / 'binding.json').read_text())
    result = json.loads((output / 'result.json').read_text())
    assert binding['counts'] == {'train': 12, 'validation': 4, 'dev': 3}
    assert binding['edge_direction'] == 'bidirectional'
    assert binding['num_relations'] == len(tz.relation_vocabulary('bidirectional'))
    assert binding['edge_dim'] == len(binding['edge_feature_layout'])
    assert binding['selection_fold'] == 'dev' and binding['test_evaluated'] is False
    assert binding['conv'] == 'gchm_v2' and binding['hub_gate'] is True
    assert binding['parameter_count'] == binding['method_config']['architecture'][
        'parameter_count']
    assert result['validation_evaluations'] == 1
    assert all(row.get('selection_fold') == 'dev' for row in result['history'])
    with np.load(output / 'dev.npz') as dev, np.load(output / 'validation.npz') as val:
        assert set(dev['sample_ids']).isdisjoint(val['sample_ids'])
        assert set(dev['subjects']).isdisjoint(val['subjects'])
        assert len(dev['sample_ids']) == 3 and len(val['sample_ids']) == 4
        digest = hashlib.sha256(np.ascontiguousarray(val['proba']).tobytes())
    assert binding['selected_validation']['prediction_sha256'] == digest.hexdigest()
    assert binding['selected_dev']['epoch'] == result['selected_epoch']
    assert result['dev_metrics']['macro_f1'] == binding['selected_dev']['metric_value']


def test_tuning_run_never_reads_validation(tmp_path):
    root, targets, canonical = artifact(tmp_path)
    output = tmp_path / 'tune'
    train.run(run_args(root, targets, canonical, output, final_eval='none',
                       conv='gchm_v2', aggregation='sum'))
    result = json.loads((output / 'result.json').read_text())
    assert result['validation_evaluations'] == 0
    assert result['metrics'] is None and result['proba_sha256'] is None
    assert not (output / 'validation.npz').exists() and (output / 'dev.npz').exists()
    assert 'selected_validation' not in result['binding']
    with pytest.raises(ValueError, match='tuning runs'):
        aggregate.aggregate_rows({'tune_seed7': result})


@pytest.mark.parametrize('final_eval,expected_reads', [('none', 0), ('validation', 1)])
def test_dev_selection_touches_validation_only_in_the_final_read(tmp_path, monkeypatch,
                                                                final_eval, expected_reads):
    root, targets, canonical = artifact(tmp_path)
    validation_ids = {sid for sid, entry in train.load_targets(targets).items()
                      if entry[1] == 'validation'}
    assert len(validation_ids) == 4
    reads = []
    original = train.evaluate

    def spy(model, loader, device, *, epoch=0):
        ids = {data.sample_id for data in loader.dataset}
        if ids & validation_ids:
            reads.append(epoch)
        return original(model, loader, device, epoch=epoch)

    monkeypatch.setattr(train, 'evaluate', spy)
    train.run(run_args(root, targets, canonical, tmp_path / 'run', final_eval=final_eval,
                       epochs=3))
    assert len(reads) == expected_reads


def test_historical_path_is_unchanged_by_default(tmp_path):
    root, targets, canonical = artifact(tmp_path)
    output = tmp_path / 'legacy'
    train.run(run_args(root, targets, canonical, output, conv='gchm', selection_fold='validation',
                       dev_limit=None, edge_direction='forward', train_limit=None, hidden=8))
    binding = json.loads((output / 'binding.json').read_text())
    assert binding['counts'] == {'train': 18, 'validation': 4}
    assert binding['selection'] == 'best validation macro_f1; full fixed budget; no early stop'
    assert 'selected_dev' not in binding and not (output / 'dev.npz').exists()
    assert binding['num_meta_relations'] == len(binding['meta_relations'])


def test_gchm_v2_hub_readout_and_mechanism_switches_change_the_model(tmp_path):
    graphs = [clinical_graph(f's{i}', 60.0 + 5 * i) for i in range(2)]
    prep = fit(tmp_path, graphs)
    data = [tz.encode_graph(g, prep, membership(g), edge_direction='bidirectional')
            for g in graphs]
    batch = Batch.from_data_list(data)
    common = dict(num_tokens=len(prep['vocabulary']), node_dim=data[0].x.size(1),
                  edge_dim=data[0].edge_attr.size(1), num_classes=2,
                  num_relations=len(tz.relation_vocabulary('bidirectional')),
                  degree_histogram=tz.degree_histogram(data), hidden=8, layers=2, dropout=0.0)
    counts, outputs = {}, {}
    for name, options in (('full', {}), ('additive', {'modulation': 'additive'}),
                          ('sum', {'aggregation': 'sum'}), ('no_hub', {'hub_gate': False}),
                          ('pool', {'readout': 'pool'})):
        torch.manual_seed(5)
        model = GCHMv2(**common, **options).eval()
        counts[name] = model.parameter_count()
        with torch.no_grad():
            outputs[name] = model(batch)
    assert counts['additive'] == counts['full']
    assert counts['sum'] < counts['full'] and counts['no_hub'] < counts['full']
    assert counts['pool'] < counts['full']
    for name in ('additive', 'sum', 'no_hub', 'pool'):
        assert not torch.allclose(outputs[name], outputs['full']), name
    hub_rows = (batch.node_type == HUB_KIND).nonzero().flatten()
    assert hub_rows.numel() == 2  # one index-visit hub per graph


def test_hub_state_reaches_messages_in_the_full_model(tmp_path):
    """Same weights, hub-gate term zeroed: the output must change, else the gate is inert."""
    graphs = [clinical_graph(f's{i}', 60.0 + 5 * i) for i in range(2)]
    prep = fit(tmp_path, graphs)
    data = [tz.encode_graph(g, prep, membership(g), edge_direction='bidirectional')
            for g in graphs]
    batch = Batch.from_data_list(data)
    torch.manual_seed(9)
    model = GCHMv2(num_tokens=len(prep['vocabulary']), node_dim=data[0].x.size(1),
                   edge_dim=data[0].edge_attr.size(1), num_classes=2,
                   num_relations=len(tz.relation_vocabulary('bidirectional')),
                   degree_histogram=tz.degree_histogram(data), hidden=8, layers=2,
                   dropout=0.0).eval()
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        for layer in ablated.layers:
            layer.hub_gate.weight.zero_()
        full, without_hub = model(batch), ablated(batch)
    assert not torch.allclose(full, without_hub)
