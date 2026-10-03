"""Tiny synthetic CLI integration; no real patient data or benchmark training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from core import train
from xgboost_control import tabular_control
from core.schema import sha256
from core.tensorize import PREPROCESSING_VERSION


def tiny_artifact(tmp_path):
    artifact = tmp_path / 'artifact'
    artifact.mkdir()
    graphs, rows = [], []
    for i in range(7):
        split = 'train' if i < 4 else 'validation' if i < 6 else 'test'
        graph = {'sample_id': f'synthetic-{i}', 'subject_id': f'person-{i}', 'split': split,
                 'coverage': {'prior_visits': 1},
                 'nodes': [
                     {'id': 'p', 'kind': 'patient', 'token': 'patient', 'age': 30 + i,
                      'gender': 'F' if i % 2 else 'M', 'race': 'synthetic',
                      'arrival_transport': 'WALK IN'},
                     {'id': 'v', 'kind': 'visit', 'token': 'visit:index', 'acuity': 2},
                     {'id': 'm', 'kind': 'vital', 'token': 'vital:heartrate', 'unit': 'bpm',
                      'value': 60 + i * 5, 'time_hours': -1., 'available_hours': -1.}],
                 'edges': [
                     {'source': 'p', 'target': 'v', 'relation': 'has_visit', 'informative': False},
                     {'source': 'v', 'target': 'm', 'relation': 'observed_vital', 'informative': False}]}
        if split == 'test':
            graph['nodes'] = None  # train/test adapters must never tensorize this record
        graphs.append(graph)
        rows.append(f'synthetic-{i},{i % 2},{split},person-{i}\n')
    graph_path = artifact / 'graphs.jsonl'
    graph_path.write_text(''.join(json.dumps(g) + '\n' for g in graphs))
    labels = [f'synthetic_label_{i}' for i in range(30)]
    canonical = tmp_path / 'canonical.json'
    canonical.write_text(json.dumps({'classes': labels}))
    manifest = {'status': 'completed', 'schema_version': 'clinical_graph_v2',
                'logic_contract_version': 'clinical_graph_logic_v2', 'temporal_clean': False,
                'graphs_sha256': sha256(graph_path), 'inherited_cohort_sha256': 'synthetic-cohort',
                'limitations': ['Synthetic bounded integration fixture; not a research result.']}
    (artifact / 'manifest.json').write_text(json.dumps(manifest))
    targets = tmp_path / 'targets.csv'
    targets.write_text('sample_id,target,split,subject_id\n' + ''.join(rows))
    binding = {'status': 'completed', 'labels': labels, 'cohort_sha256': 'synthetic-cohort',
               'artifact_files': {'targets.csv': sha256(targets)}}
    (tmp_path / 'binding_manifest.json').write_text(json.dumps(binding))
    return artifact, targets, canonical


def args_for(artifact, targets, canonical, output, conv='edge_conditioned'):
    return argparse.Namespace(artifact=str(artifact), targets=str(targets), canonical=str(canonical),
                              output=str(output), seed=7, top_k_labels=2, drop_relation=None,
                              rewire_relation=None, edges='all', train_limit=None, token_min_count=1,
                              min_prior_visits=0, device='cpu', batch_size=2, conv=conv, hidden=8,
                              layers=1, dropout=0., token_dim=4, no_edge_payload=False,
                              no_message_passing=False, heads=2, modulation='multiplicative',
                              weights='none', lr=0.001, epochs=1)


@pytest.mark.parametrize('conv', ['edge_conditioned', 'hgt', 'gchm'])
def test_tiny_training_records_corrected_contract_and_visit_ids(tmp_path, conv):
    artifact, targets, canonical = tiny_artifact(tmp_path)
    output = tmp_path / 'run'
    train.run(args_for(artifact, targets, canonical, output, conv))
    binding = json.loads((output / 'binding.json').read_text())
    assert binding['input_contract_version'] == PREPROCESSING_VERSION
    assert binding['preprocessing_schema_version'] == PREPROCESSING_VERSION
    assert binding['active_parameter_count'] == binding['parameter_count']
    assert binding['evaluation_version'] == train.EVALUATION_VERSION
    assert binding['test_evaluated'] is False
    assert binding['counts'] == {'train': 4, 'validation': 2}
    assert binding['rewiring']['train']['changed_graphs'] == 0
    assert 'sample_ids' not in binding
    assert len(binding['split_sample_ids_sha256']['train']) == 64
    with np.load(output / 'validation.npz') as data:
        assert data['sample_ids'].tolist() == ['synthetic-4', 'synthetic-5']
        assert data['subjects'].tolist() == ['person-4', 'person-5']
        assert np.isfinite(data['proba']).all()
    result = json.loads((output / 'result.json').read_text())
    assert result['patient_equal']['visits'] == 2
    assert result['patient_equal']['weight_policy'] == 'inverse_evaluated_visits_per_patient'


def test_tabular_cli_matches_tiny_gnn_contract(tmp_path, monkeypatch):
    artifact, targets, canonical = tiny_artifact(tmp_path)
    run, output = tmp_path / 'run', tmp_path / 'tabular'
    train.run(args_for(artifact, targets, canonical, run))
    monkeypatch.setattr('sys.argv', ['tabular', '--artifact', str(artifact), '--targets', str(targets),
                                    '--canonical', str(canonical), '--out', str(output), '--rounds', '1',
                                    '--seed', '7', '--token-min-count', '1', '--top-k-labels', '2',
                                    '--weights', 'none', '--match-run', str(run)])
    tabular_control.main()
    binding = json.loads((output / 'binding.json').read_text())
    peer = json.loads((run / 'binding.json').read_text())
    for key in ('input_contract_version', 'preprocessing_schema_version', 'preprocessing_sha256',
                'evaluation_version', 'label_order', 'targets_path', 'target_binding_sha256',
                'artifact_graphs_sha256', 'counts', 'split_sample_ids_sha256'):
        assert binding[key] == peer[key]
    assert Path(binding['artifact']) == artifact
    assert binding['source_code'] == peer['source_code']
    assert json.loads((output / 'result.json').read_text())['status'] == 'completed'


def test_tabular_match_run_rejects_different_source_binding(tmp_path, monkeypatch):
    artifact, targets, canonical = tiny_artifact(tmp_path)
    run = tmp_path / 'run'
    train.run(args_for(artifact, targets, canonical, run))
    peer_path = run / 'binding.json'
    peer = json.loads(peer_path.read_text())
    peer['source_code']['train.py'] = '0' * 64
    peer_path.write_text(json.dumps(peer))
    monkeypatch.setattr('sys.argv', ['tabular', '--artifact', str(artifact), '--targets', str(targets),
                                    '--canonical', str(canonical), '--out', str(tmp_path / 'tabular'),
                                    '--rounds', '1', '--seed', '7', '--token-min-count', '1',
                                    '--top-k-labels', '2', '--weights', 'none', '--match-run', str(run)])
    with pytest.raises(SystemExit, match='source_code differs'):
        tabular_control.main()
