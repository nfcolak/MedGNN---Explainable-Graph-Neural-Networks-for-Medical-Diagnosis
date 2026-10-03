"""Bounded synthetic regressions for the shared clinical input contract."""
import copy
import json

import numpy as np
import pytest
import torch

from comparison.standardized.clinical_graph_v2 import NODE_KINDS
from comparison.standardized.clinical_graph_v2 import tensorize as tz


def graph(sample_id, value=70.0, *, kind='vital', token='vital:heartrate',
          unit='bpm', **patient):
    return {
        'sample_id': sample_id,
        'nodes': [
            {'id': 'p', 'kind': 'patient', 'token': 'patient', **patient},
            {'id': 'v', 'kind': 'visit', 'token': 'visit:index', 'acuity': 3},
            {'id': 'n', 'kind': kind, 'token': token, 'unit': unit,
             'value': value, 'time_hours': -1.0, 'available_hours': -0.5},
        ],
        'edges': [
            {'source': 'p', 'target': 'v', 'relation': 'has_visit', 'informative': False},
            {'source': 'v', 'target': 'n', 'relation': 'observed_vital', 'informative': False},
        ],
        'coverage': {'complaints': 0, 'index_measurements': 0, 'prior_visits': 0,
                     'informative_edges': 0},
    }


def fit(tmp_path, graphs, train_ids=None, token_min_count=1):
    path = tmp_path / 'graphs.jsonl'
    path.write_text(''.join(json.dumps(g) + '\n' for g in graphs))
    ids = set(train_ids) if train_ids is not None else {g['sample_id'] for g in graphs}
    return tz.fit_preprocessing(path, ids, token_min_count=token_min_count)


def test_vital_magnitudes_are_fitted_and_not_zeroed(tmp_path):
    rows = [graph(f'train-{i}', 50.0 + i) for i in range(24)]
    prep = fit(tmp_path, rows)
    lo = tz.encode_graph(graph('validation-low', 50), prep)
    hi = tz.encode_graph(graph('validation-high', 80), prep)
    value_col = len(NODE_KINDS)
    assert lo.x[2, value_col] < hi.x[2, value_col]
    assert lo.x[2, value_col + 1] == hi.x[2, value_col + 1] == 1
    assert any(s['count'] == 24 for s in prep['scaler'].stats.values())


@pytest.mark.parametrize('kind,token', [('vital', 'vital:rare'),
                                        ('measurement', 'lab:rare'),
                                        ('knowledge', 'numeric:future-kind')])
def test_rare_and_unseen_finite_values_keep_magnitude(tmp_path, kind, token):
    prep = fit(tmp_path, [graph('train', 1, kind=kind, token=token)])
    for observed_token in (token, token + ':unseen'):
        low = tz.encode_graph(graph('v1', 2, kind=kind, token=observed_token), prep)
        high = tz.encode_graph(graph('v2', 200000, kind=kind, token=observed_token), prep)
        assert 0 < low.x[2, len(NODE_KINDS)] < high.x[2, len(NODE_KINDS)]
        assert high.x[2, len(NODE_KINDS) + 1] == 1


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), 'invalid'])
def test_nonfinite_or_invalid_values_are_missing_not_nan(tmp_path, bad):
    rows = [graph(f't{i}', i + 1) for i in range(24)] + [graph('bad', bad)]
    prep = fit(tmp_path, rows)
    data = tz.encode_graph(graph('validation', bad), prep)
    assert torch.isfinite(data.x).all()
    assert data.x[2, len(NODE_KINDS) + 1] == 0
    json.dumps(tz.preprocessing_state(prep), allow_nan=False)


def test_measurement_scaling_and_token_identity_are_unit_specific(tmp_path):
    rows = [graph(f'mg-{i}', i + 1, kind='measurement', token='lab:creatinine', unit='mg/dL')
            for i in range(24)]
    rows += [graph(f'umol-{i}', (i + 1) * 100, kind='measurement',
                   token='lab:creatinine', unit='umol/L') for i in range(24)]
    prep = fit(tmp_path, rows)
    left = tz.encode_graph(rows[0], prep)
    right = tz.encode_graph(rows[24], prep)
    assert left.token[2] != right.token[2]
    assert left.token[2] != 0 and right.token[2] != 0
    torch.testing.assert_close(left.x[2], right.x[2])
    unknown = tz.encode_graph(graph('v', 9, kind='measurement',
                                    token='lab:creatinine', unit='unknown-unit'), prep)
    assert unknown.token[2] == 0
    assert unknown.x[2, len(NODE_KINDS)] > 0  # fallback, not a cross-unit z-score


@pytest.mark.parametrize('field,old,new,node_index', [
    ('age', 40, 80, 0), ('gender', 'F', 'M', 0), ('race', 'A', 'B', 0),
    ('arrival_transport', 'WALK IN', 'AMBULANCE', 0), ('acuity', 2, 5, 1),
])
def test_clinical_context_reaches_node_features(tmp_path, field, old, new, node_index):
    first, second = graph('train-a'), graph('train-b')
    first['nodes'][node_index][field] = old
    second['nodes'][node_index][field] = new
    prep = fit(tmp_path, [first, second])
    a, b = tz.encode_graph(first, prep), tz.encode_graph(second, prep)
    assert not torch.equal(a.x[node_index], b.x[node_index])
    state = tz.preprocessing_state(prep)
    assert any(field in name for name in state['node_feature_layout'])


def test_context_state_is_train_only_stable_and_id_free(tmp_path):
    train = graph('train-id-secret', age=60, gender='F', race='A', arrival_transport='WALK IN')
    val = graph('validation-id-secret', age=999, gender='VALIDATION_ONLY', race='ZZ',
                arrival_transport='HELICOPTER', subject_id='subject-secret')
    prep = fit(tmp_path, [train, val], {'train-id-secret'})
    state = tz.preprocessing_state(prep)
    assert state['context_categories']['gender'] == ['F']
    assert 'VALIDATION_ONLY' not in json.dumps(state)
    assert 'subject-secret' not in json.dumps(state)
    assert 'train-id-secret' not in json.dumps(state)
    missing = copy.deepcopy(val)
    missing['nodes'][0].pop('gender')
    a, b = tz.encode_graph(val, prep), tz.encode_graph(missing, prep)
    assert not torch.equal(a.x[0], b.x[0])  # unknown != missing
    assert tz.preprocessing_state(prep) == state  # transform never refits
    shuffled = fit(tmp_path, [val, train], {'train-id-secret'})
    assert tz.preprocessing_state(shuffled) == state


def test_preprocessing_roundtrip_is_versioned_and_tensor_exact(tmp_path):
    g = graph('train', age=50, gender='F', race='A', arrival_transport='WALK IN')
    prep = fit(tmp_path, [g])
    state = json.loads(json.dumps(tz.preprocessing_state(prep), allow_nan=False))
    assert state.get('preprocessing_version'), 'the new layout must not replay legacy checkpoints'
    reloaded = tz.load_preprocessing(state)
    before, after = tz.encode_graph(g, prep), tz.encode_graph(g, reloaded)
    for name in ('x', 'token', 'edge_index', 'edge_attr', 'node_type', 'edge_triple'):
        assert torch.equal(getattr(before, name), getattr(after, name))
    assert tz.preprocessing_state(reloaded) == state
    for bad_version in (None, 'old'):
        legacy = dict(state)
        legacy['preprocessing_version'] = bad_version
        with pytest.raises(ValueError, match='(?i)(version|incompatible)'):
            tz.load_preprocessing(legacy)
    changed = copy.deepcopy(state)
    changed['node_feature_layout'].reverse()
    with pytest.raises(ValueError, match='(?i)(layout|incompatible)'):
        tz.load_preprocessing(changed)


def test_tabular_cannot_see_raw_values_or_coverage_hidden_from_gnn(tmp_path):
    from comparison.standardized.clinical_graph_v2 import tabular_control as tb
    rows = [graph(f't{i}', i + 1, age=50, gender='F') for i in range(24)]
    a, b = graph('va', 1000), graph('vb', 10000)
    b['coverage']['complaints'] = 999
    b['coverage']['index_measurements'] = 777
    b['coverage']['prior_visits'] = 333
    rows += [a, b]
    targets = {g['sample_id']: (0, 'train' if i < 24 else 'validation')
               for i, g in enumerate(rows)}
    prep = fit(tmp_path, rows, {f't{i}' for i in range(24)})
    encoded_a, encoded_b = tz.encode_graph(a, prep), tz.encode_graph(b, prep)
    assert torch.equal(encoded_a.x, encoded_b.x)
    X, _, _ = tb.build(tmp_path, targets, prep)
    np.testing.assert_array_equal(X[-2], X[-1])


def tabular_fixture(tmp_path):
    import csv
    from comparison.standardized.clinical_graph_v2.schema import sha256
    rows = [graph(f's{i}', 60 + i, age=40 + i, gender='F' if i % 2 else 'M')
            for i in range(7)]
    fit(tmp_path, rows)
    manifest = {'status': 'completed', 'logic_contract_version': 'clinical_graph_logic_v2',
                'schema_version': 'clinical_graph_v3', 'temporal_clean': False,
                'graphs_sha256': sha256(tmp_path / 'graphs.jsonl'),
                'inherited_cohort_sha256': sha256(tmp_path / 'graphs.jsonl')}
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    targets_path = tmp_path / 'targets.csv'
    with targets_path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'subject_id', 'target', 'split'])
        writer.writeheader()
        for i, g in enumerate(rows):
            writer.writerow({'sample_id': g['sample_id'], 'subject_id': f'patient{i // 2}',
                             'target': i % 2, 'split': 'train' if i < 4 else
                             'validation' if i < 6 else 'test'})
    from pathlib import Path
    labels = json.loads(Path('comparison/canonical_split.json').read_text())['classes']
    (tmp_path / 'binding_manifest.json').write_text(json.dumps({
        'status': 'completed', 'artifact_files': {'targets.csv': sha256(targets_path)},
        'cohort_sha256': manifest['inherited_cohort_sha256'], 'labels': labels}))
    return targets_path


def test_tabular_entrypoint_rejects_historical_manifest_before_fit(tmp_path, monkeypatch):
    import sys
    from comparison.standardized.clinical_graph_v2 import tabular_control as tb
    target = tabular_fixture(tmp_path)
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'completed'}))
    monkeypatch.setattr(sys, 'argv', ['tabular', '--artifact', str(tmp_path),
                                     '--targets', str(target), '--out', str(tmp_path / 'out'),
                                     '--rounds', '1'])
    with pytest.raises(ValueError, match='Historical clinical artifact'):
        tb.main()
    assert not (tmp_path / 'out').exists()


def test_tabular_one_round_fixture_saves_shared_state_and_visit_ids(tmp_path, monkeypatch):
    import sys
    from comparison.standardized.clinical_graph_v2 import tabular_control as tb
    from comparison.standardized.clinical_graph_v2 import train
    target = tabular_fixture(tmp_path)
    out = tmp_path / 'out'
    monkeypatch.setattr(sys, 'argv', ['tabular', '--artifact', str(tmp_path),
                                     '--targets', str(target), '--out', str(out),
                                     '--rounds', '1', '--top-k-labels', '2'])
    tb.main()
    saved = np.load(out / 'validation.npz')
    assert saved['sample_ids'].tolist() == ['s4', 's5']
    assert saved['subjects'].tolist() == ['patient2', 'patient2']
    result = json.loads((out / 'result.json').read_text())
    assert result['metrics'] == train.metrics(saved['y'], saved['proba'], 2)
    assert result['patient_equal'] == train.patient_equal_metrics(
        saved['y'], saved['proba'], saved['subjects'], 2)
    binding = json.loads((out / 'binding.json').read_text())
    assert binding['preprocessing_version'] == tz.PREPROCESSING_VERSION
    assert binding['evaluation_version'] == train.EVALUATION_VERSION
    assert binding['input_contract_version'] == tz.PREPROCESSING_VERSION
    assert binding['targets_path'] == str(target.resolve())
    from comparison.standardized.clinical_graph_v2.schema import sha256
    assert binding['target_binding_sha256'] == sha256(tmp_path / 'binding_manifest.json')
    assert binding['rewiring']['train']['changed_edges'] == 0
    assert binding['rewiring']['validation']['changed_graphs'] == 0
    state = json.loads((out / 'preprocessing.json').read_text())
    assert tz.preprocessing_state(tz.load_preprocessing(state)) == state
    assert (out / 'model.ubj').is_file()
    # An occupied run must never be silently overwritten.
    with pytest.raises(FileExistsError):
        tb.main()


def test_rewire_adapter_delegates_without_implementing_a_second_policy(monkeypatch):
    import sys
    import types
    calls = []
    module_name = 'comparison.standardized.clinical_graph_v2.core.rewiring'
    module = types.ModuleType(module_name)
    def rewire_edges(*args):
        calls.append(args)
        return ['delegated']
    setattr(module, 'rewire_edges', rewire_edges)
    monkeypatch.setitem(sys.modules, module_name, module)
    args = ([], [], {}, {'baseline_of'}, 'sample', 9)
    assert tz._rewire_edges(*args) == ['delegated']
    assert calls == [args]


@pytest.mark.parametrize('field', ['delta', 'interval_hours', 'last_seen_hours', 'prior_encounters'])
def test_nonfinite_edge_payload_is_missing_or_rejected(tmp_path, field):
    g = graph('train')
    prep = fit(tmp_path, [g])
    g['edges'][0][field] = float('nan')
    try:
        data = tz.encode_graph(g, prep)
    except ValueError as error:
        assert 'finite' in str(error)
    else:
        assert torch.isfinite(data.edge_attr).all()


def test_direct_legacy_prep_cannot_claim_new_encoding_version(tmp_path):
    prep = fit(tmp_path, [graph('train')])
    prep.pop('preprocessing_version')
    with pytest.raises(ValueError, match='(?i)(version|incompatible)'):
        tz.encode_graph(graph('validation'), prep)
    with pytest.raises(ValueError, match='(?i)(version|incompatible)'):
        tz.preprocessing_state(prep)


def test_tabular_rejects_unbound_targets_before_training(tmp_path, monkeypatch):
    import sys
    from comparison.standardized.clinical_graph_v2 import tabular_control as tb
    target = tabular_fixture(tmp_path)
    binding_path = tmp_path / 'binding_manifest.json'
    binding = json.loads(binding_path.read_text())
    binding.pop('cohort_sha256')
    binding_path.write_text(json.dumps(binding))
    monkeypatch.setattr(sys, 'argv', ['tabular', '--artifact', str(tmp_path),
                                     '--targets', str(target), '--out', str(tmp_path / 'out'),
                                     '--rounds', '1', '--top-k-labels', '2'])
    with pytest.raises(ValueError, match='cohort lineage'):
        tb.main()
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('change,expected', [('reorder', 0), ('pair', 1)])
def test_rewired_edge_count_uses_endpoint_multisets(tmp_path, monkeypatch, change, expected):
    g = graph('train')
    g['nodes'].append({**g['nodes'][2], 'id': 'n2'})
    prep = fit(tmp_path, [g])
    def controlled(edges, *unused):
        if change == 'reorder':
            return list(reversed(edges))
        return [{**e, 'target': 'n2'} if e['relation'] == 'observed_vital' else e
                for e in edges]
    monkeypatch.setattr(tz, '_rewire_edges', controlled)
    data = tz.encode_graph(g, prep, rewire_relations=['observed_vital'])
    assert data.rewired_edge_count == expected


@pytest.mark.parametrize('limit,min_prior', [(None, 0), (6, 1)])
def test_tabular_and_gnn_share_selected_training_fit(tmp_path, limit, min_prior):
    from comparison.standardized.clinical_graph_v2 import tabular_control as tb
    from comparison.standardized.clinical_graph_v2 import train
    rows = [graph(f's{i}', 20 + i, age=30 + i, gender='F' if i % 2 else 'M')
            for i in range(11)]
    for i, g in enumerate(rows):
        g['coverage']['prior_visits'] = i % 2
    targets = {g['sample_id']: (i % 2, 'train' if i < 8 else
                               'validation' if i < 10 else 'test', f'patient{i}')
               for i, g in enumerate(rows)}
    fit(tmp_path, rows)
    splits, gnn_prep = train.build_dataset(tmp_path, targets, 'all', limit, 1, 7,
                                          min_prior_visits=min_prior)
    selected = tb.subsample_train(targets, limit, 7)
    tab_prep = tb.fit_features(tmp_path, selected, min_count=1, min_prior_visits=min_prior)
    assert tz.preprocessing_state(tab_prep) == tz.preprocessing_state(gnn_prep)
    X, y, folds, metadata = tb.build(tmp_path, selected, tab_prep, min_prior,
                                     return_metadata=True)
    by_id = {sid: i for i, sid in enumerate(metadata['sample_ids'])}
    assert 's10' not in by_id  # TEST never reaches a tensor or matrix row
    for split in ('train', 'validation'):
        for data in splits[split]:
            i = by_id[data.sample_id]
            np.testing.assert_array_equal(X[i], tb.project_encoded(data, tab_prep))
            assert y[i] == int(data.y)
            assert folds[i] == (0 if split == 'train' else 1)
    assert X.shape[1] == len(tb.projection_layout(tab_prep))
    assert metadata['rewired_edge_counts'].tolist() == [0] * len(X)


@pytest.mark.parametrize('conv', ['edge_conditioned', 'hgt', 'gchm'])
def test_existing_gnn_consumes_new_context_width_without_architecture_changes(tmp_path, conv):
    from torch_geometric.data import Batch
    from comparison.standardized.clinical_graph_v2.model import ClinicalGNN
    rows = [graph('t1', age=50, gender='F', race='A'),
            graph('t2', 100, age=75, gender='M', race='B')]
    prep = fit(tmp_path, rows)
    data = [tz.encode_graph(g, prep) for g in rows]
    torch.manual_seed(7)
    model = ClinicalGNN(num_tokens=len(prep['vocabulary']), node_dim=data[0].x.shape[1],
                        edge_dim=data[0].edge_attr.shape[1], hidden=16, layers=1,
                        num_classes=2, dropout=0, conv=conv,
                        num_triples=len(prep['triples']) + 1,
                        payload_dim=tz.PAYLOAD_WIDTH,
                        degree_histogram=tz.degree_histogram(data) if conv == 'gchm' else None)
    model.eval()
    with torch.no_grad():
        output = model(Batch.from_data_list(data))
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
