"""Graph strata must partition visits, not silently lose or duplicate them."""
import numpy as np
import pytest
import json

from comparison.standardized.clinical_graph_v2 import stratify


def test_repeated_subjects_without_sample_ids_are_ambiguous():
    shapes = {'a': {'subject': 'patient'}, 'b': {'subject': 'patient'}}
    with pytest.raises(ValueError, match='sample_ids'):
        stratify.verify_alignment(['a', 'b'], shapes, ['patient', 'patient'])


def test_strata_cover_every_nonnegative_count_exactly_once():
    assert hasattr(stratify, 'count_bins'), 'strata lack a checked partition'
    values = np.arange(31)
    bins = stratify.count_bins('measurements', values, [0, 5, 9, 15])
    np.testing.assert_array_equal(np.sum([mask for _, mask in bins], axis=0),
                                  np.ones(len(values)))
    assert bins[1][0] == 'measurements=1-4'


def test_visit_ids_detect_swapped_visits_of_same_patient():
    shapes = {'a': {'subject': 'patient', 'target': 0},
              'b': {'subject': 'patient', 'target': 1}}
    with pytest.raises(ValueError, match='sample_ids'):
        stratify.verify_alignment(['a', 'b'], shapes, ['patient', 'patient'],
                                   ['b', 'a'], [1, 0])
    assert stratify.verify_alignment(['a', 'b'], shapes, ['patient', 'patient'],
                                     ['a', 'b'], [0, 1])


def test_shape_cache_is_bound_to_graph_bytes_and_target_mapping(tmp_path):
    artifact = tmp_path / 'graphs'
    artifact.mkdir()
    graph = {'sample_id': 'visit', 'nodes': [{}], 'edges': [],
             'coverage': dict.fromkeys(('prior_visits', 'complaints', 'index_measurements',
                                       'baseline_links', 'informative_edges'), 0)}
    path = artifact / 'graphs.jsonl'
    path.write_text(json.dumps(graph) + '\n')
    targets = {'visit': (0, 'validation', 'patient')}
    first = stratify.extract_shapes(artifact, targets, tmp_path / 'cache')
    graph['nodes'].append({})
    path.write_text(json.dumps(graph) + '\n')
    second = stratify.extract_shapes(artifact, targets, tmp_path / 'cache')
    assert first['shapes']['visit']['nodes'] == 1
    assert second['shapes']['visit']['nodes'] == 2
    assert second['shapes']['visit']['target'] == 0
    third = stratify.extract_shapes(artifact, {'visit': (2, 'validation', 'patient')},
                                     tmp_path / 'cache')
    assert third['shapes']['visit']['target'] == 2


def test_cli_replays_history_filter_and_checks_control_visit_ids(tmp_path, monkeypatch, capsys):
    from comparison.standardized.clinical_graph_v2.schema import sha256
    artifact, run, control = (tmp_path / n for n in ('artifact', 'run', 'control'))
    for path in (artifact, run, control):
        path.mkdir()
    graphs = []
    for i in range(3):
        graphs.append({'sample_id': f'v{i}', 'nodes': [{}] * (i + 1), 'edges': [],
                       'coverage': {'prior_visits': i, 'complaints': i,
                                    'index_measurements': i + 1, 'baseline_links': 0,
                                    'informative_edges': 0}})
    gp = artifact / 'graphs.jsonl'
    gp.write_text(''.join(json.dumps(g) + '\n' for g in graphs))
    tp = tmp_path / 'targets.csv'
    tp.write_text('sample_id,target,split,subject_id\n' + ''.join(
        f'v{i},{i},validation,p\n' for i in range(3)))
    binding = {'artifact': str(artifact), 'targets_path': str(tp),
               'targets_sha256': sha256(tp), 'artifact_graphs_sha256': sha256(gp),
               'min_prior_visits': 1, 'num_classes': 3,
               'preprocessing_sha256': 'a' * 64}
    for path in (run, control):
        (path / 'binding.json').write_text(json.dumps(binding))
        np.savez(path / 'validation.npz', y=[1, 2], proba=np.eye(3)[[1, 2]],
                 subjects=['p', 'p'], sample_ids=['v1', 'v2'])
    monkeypatch.setattr('sys.argv', ['stratify', '--run', str(run), '--control', str(control)])
    stratify.main()
    assert 'alignment verified: 2 validation rows' in capsys.readouterr().out
    mismatched_binding = dict(binding, preprocessing_sha256='b' * 64)
    (control / 'binding.json').write_text(json.dumps(mismatched_binding))
    with pytest.raises(ValueError, match='preprocessing_sha256'):
        stratify.main()
    for invalid_binding in ({key: value for key, value in binding.items()
                             if key != 'preprocessing_sha256'},
                            dict(binding, preprocessing_sha256='')):
        (control / 'binding.json').write_text(json.dumps(invalid_binding))
        with pytest.raises(ValueError, match='preprocessing_sha256'):
            stratify.main()
    (control / 'binding.json').write_text(json.dumps(binding))
    np.savez(control / 'validation.npz', y=[1, 2], proba=np.eye(3)[[1, 2]],
             subjects=['p', 'p'], sample_ids=['v2', 'v1'])
    with pytest.raises(ValueError, match='sample_ids'):
        stratify.main()
