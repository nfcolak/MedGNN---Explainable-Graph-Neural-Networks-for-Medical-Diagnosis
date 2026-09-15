"""Train-only preprocessing and isolated runner contracts."""
import importlib.util
import json
import numpy as np
import pytest
import torch


def test_capacity_controls_are_measured_without_rng_side_effects():
    import pna_analysis.model as module
    assert hasattr(module, 'capacity_report'), 'measured PNA controls missing'
    state = torch.get_rng_state().clone()
    report = module.capacity_report(193, 30, width=64, layers=2, rank=16)
    assert torch.equal(state, torch.get_rng_state())
    assert report['plain']['parameters'] < report['interaction']['parameters']
    assert report['plain_wide']['width'] > 64
    assert report['relative_gap'] <= report['tolerance'] == .02
    for name, item in ((k, report[k]) for k in ('plain', 'interaction', 'plain_wide')):
        model = module.PNAPredictor(193, 30, torch.tensor([1, 8, 2]),
                                   width=item['width'], layers=2, rank=16, interactions=name == 'interaction')
        assert item['parameters'] == sum(p.numel() for p in model.parameters())
        assert sum(p.numel() for n, p in model.named_parameters() if '.pair.' in n) == (8192 if name == 'interaction' else 0)


def input_fixture(tmp_path):
    from shared.lib.benchmark_contract import file_sha256
    root = tmp_path / 'common'; root.mkdir()
    source = tmp_path / 'source.csv'; source.write_text('synthetic fixture,not clinical data\n')
    classes = ['class' + str(i) for i in range(6)]
    folds = np.array([0] * 6 + [1, 2])
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'classes': classes, 'fold': {str(i): int(f) for i, f in enumerate(folds)}}))
    np.savez(root / 'inputs_no_identifiers.npz', presence=np.array([[1, 0, 1, 0]] * 8, dtype=np.int8),
             y=np.array(list(range(6)) + [0, 1]), folds=folds)
    contract = {'input_sha256': file_sha256(root / 'inputs_no_identifiers.npz'),
                'source_sha256': file_sha256(source), 'split_sha256': file_sha256(split),
                'names': ['med:a', 'med:b', 'cc:c', 'cc:d'], 'classes': classes,
                'counts': [6, 1, 1], 'concept_channels': 4, 'effective_feature_dim': 5,
                'fit_rows': 6, 'fit_scope': 'canonical train rows of already-filtered source snapshot only',
                'fit_identity_sha256': 'fixture-train-binding', 'patient_specific_hub_channels': 0,
                'temporal_clean': False, 'raw_to_model_train_only': False, 'audited_graphs': 8}
    (root / 'input_contract_and_audit.json').write_text(json.dumps(contract))
    manifest = root / 'trials/gsat/current/run_manifest.json'; manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'status': 'completed', 'topology': 'star', 'n_train': 6, 'n_validation': 1,
                                    'binding': {k: contract[k] for k in ('input_sha256', 'split_sha256')}}))
    return root, source, split


def test_common_loader_checks_existing_bindings_and_class_order(tmp_path):
    import pna_analysis.data as module
    assert hasattr(module, 'load_common_input'), 'bound common-input loader missing'
    root, source, split = input_fixture(tmp_path)
    bundle = module.load_common_input(root, source, split)
    assert bundle['presence'].shape == (8, 4)
    assert bundle['contract']['classes'] == ['class' + str(i) for i in range(6)]
    assert torch.equal(bundle['degree_histogram'], torch.tensor([0, 12, 6, 0, 0]))
    manifest = root / 'trials/gsat/current/run_manifest.json'
    payload = json.loads(manifest.read_text()); payload['binding']['input_sha256'] = 'wrong'
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='binding'):
        module.load_common_input(root, source, split)
    payload['binding']['input_sha256'] = bundle['contract']['input_sha256']
    manifest.write_text(json.dumps(payload))
    audit_path = root / 'input_contract_and_audit.json'
    audit = json.loads(audit_path.read_text()); audit['classes'].reverse(); audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='class order'):
        module.load_common_input(root, source, split)
    audit['classes'].reverse(); audit_path.write_text(json.dumps(audit))
    source.write_text('changed source')
    with pytest.raises(ValueError, match='source_sha256'):
        module.load_common_input(root, source, split)


def test_cli_dry_run_validates_input_without_creating_output(tmp_path, capsys):
    assert importlib.util.find_spec('pna_analysis.run') is not None, 'opt-in runner missing'
    from pna_analysis.run import main
    root, source, split = input_fixture(tmp_path)
    target = tmp_path / 'untouched'
    argv = ['--input-root', str(root), '--source', str(source), '--split', str(split),
            '--output-dir', str(target), '--train-limit', '6', '--val-limit', '1', '--dry-run']
    assert main(argv) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['action'] == 'dry_run'
    assert plan['input_counts'] == [6, 1, 1]
    assert not target.exists()
    assert plan['config']['selection'] == 'fixed-step last checkpoint; no validation selection'
    with pytest.raises(SystemExit):
        main(['--steps', '101'])
    with pytest.raises(SystemExit):
        main(['--dry-run', '--execute'])


def test_opt_in_training_checkpoint_reload_and_no_overwrite(tmp_path, monkeypatch, capsys):
    import pna_analysis.run as runner
    root, source, split = input_fixture(tmp_path)
    monkeypatch.setattr(runner, 'EXPERIMENT_ROOT', tmp_path / 'experiments')
    target = runner.EXPERIMENT_ROOT / 'smoke'
    argv = ['--input-root', str(root), '--source', str(source), '--split', str(split),
            '--output-dir', str(target), '--train-limit', '6', '--val-limit', '0',
            '--width', '8', '--layers', '1', '--rank', '3', '--batch-size', '6', '--steps', '2', '--execute']
    assert runner.main(argv) == 0
    manifest = json.loads((target / 'run_manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['optimizer_steps'] == 2
    assert manifest['test_evaluated'] is False
    assert manifest['replay']['logits_max_abs_diff'] == 0
    assert manifest['replay']['metrics_equal'] is True
    assert manifest['preservation_verified'] is True
    assert all(v > 0 for v in manifest['interaction_parameter_deltas'].values())
    history = json.loads((target / 'history.json').read_text())
    assert all(all(v > 0 for v in row['pair_gradient_l1'].values()) for row in history)
    before = (target / 'checkpoint.pt').read_bytes()
    with pytest.raises(FileExistsError):
        runner.main(argv)
    assert (target / 'checkpoint.pt').read_bytes() == before
    capsys.readouterr()


def test_replay_cli_is_read_only(tmp_path, monkeypatch, capsys):
    import pna_analysis.run as runner
    from shared.lib.benchmark_contract import file_sha256
    root, source, split = input_fixture(tmp_path)
    monkeypatch.setattr(runner, 'EXPERIMENT_ROOT', tmp_path / 'experiments')
    target = runner.EXPERIMENT_ROOT / 'smoke'
    runner.main(['--input-root', str(root), '--source', str(source), '--split', str(split),
                 '--output-dir', str(target), '--train-limit', '6', '--val-limit', '0',
                 '--width', '8', '--layers', '1', '--rank', '3', '--steps', '1', '--execute'])
    capsys.readouterr()
    before = {p.name: file_sha256(p) for p in target.iterdir()}
    assert runner.main(['--replay', str(target)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['arrays_equal'] and report['metrics_equal']
    assert before == {p.name: file_sha256(p) for p in target.iterdir()}
    # Replayed output corruption must be rejected, not silently relabeled success.
    (target / 'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='replay'):
        runner.main(['--replay', str(target)])


@pytest.mark.filterwarnings('ignore:y_pred contains classes not in y_true:UserWarning')
def test_sqrt_weights_fit_full_train_not_optimization_subset(tmp_path, monkeypatch, capsys):
    import pna_analysis.run as runner
    from shared.lib.benchmark_contract import file_sha256
    root, source, split = input_fixture(tmp_path)
    npz = root / 'inputs_no_identifiers.npz'
    with np.load(npz) as a:
        y = np.concatenate([np.zeros(8, dtype=np.int64), a['y']])
        folds = np.concatenate([np.zeros(8, dtype=np.int64), a['folds']])
        presence = np.concatenate([a['presence'][:1].repeat(8, axis=0), a['presence']])
    np.savez(npz, y=y, folds=folds, presence=presence)
    split_data = json.loads(split.read_text())
    split_data['fold'] = {str(i): int(f) for i, f in enumerate(folds)}
    split.write_text(json.dumps(split_data))
    audit_path = root / 'input_contract_and_audit.json'; audit = json.loads(audit_path.read_text())
    audit.update(input_sha256=file_sha256(npz), split_sha256=file_sha256(split), counts=[14, 1, 1], fit_rows=14, audited_graphs=16)
    audit_path.write_text(json.dumps(audit))
    manifest_path = root / 'trials/gsat/current/run_manifest.json'; binding = json.loads(manifest_path.read_text())
    binding['n_train'] = 14; binding['binding'] = {k: audit[k] for k in ('input_sha256', 'split_sha256')}
    manifest_path.write_text(json.dumps(binding))
    monkeypatch.setattr(runner, 'EXPERIMENT_ROOT', tmp_path / 'experiments')
    target = runner.EXPERIMENT_ROOT / 'sqrt'
    runner.main(['--input-root', str(root), '--source', str(source), '--split', str(split),
                 '--output-dir', str(target), '--train-limit', '2', '--val-limit', '0', '--loss', 'sqrt_inverse',
                 '--width', '8', '--layers', '1', '--steps', '1', '--execute'])
    manifest = json.loads((target / 'run_manifest.json').read_text())
    assert manifest['class_counts_train'] == [9, 1, 1, 1, 1, 1]
    assert np.allclose(manifest['class_weights'], np.sqrt(14 / (6 * np.array([9, 1, 1, 1, 1, 1]))))
    assert manifest['n_train_smoke'] == 2
    capsys.readouterr()


def test_failure_manifest_retains_completed_step_count(tmp_path, monkeypatch, capsys):
    import pna_analysis.run as runner
    root, source, split = input_fixture(tmp_path)
    monkeypatch.setattr(runner, 'EXPERIMENT_ROOT', tmp_path / 'experiments')
    target = runner.EXPERIMENT_ROOT / 'interrupted'
    original = torch.optim.Adam.step
    calls = []
    def interrupted(optimizer, *args, **kwargs):
        if calls:
            raise RuntimeError('injected interruption after one real optimizer step')
        calls.append(True)
        return original(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, 'step', interrupted)
    with pytest.raises(RuntimeError, match='injected interruption'):
        runner.main(['--input-root', str(root), '--source', str(source), '--split', str(split),
                     '--output-dir', str(target), '--train-limit', '6', '--val-limit', '0',
                     '--width', '8', '--layers', '1', '--steps', '2', '--execute'])
    manifest = json.loads((target / 'run_manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['optimizer_steps'] == 1
    assert len(json.loads((target / 'history.json').read_text())) == 1
    assert not (target / 'checkpoint.pt').exists()
    capsys.readouterr()


def test_degree_histogram_uses_only_training_stars():
    import pna_analysis.data as module
    assert hasattr(module, 'train_degree_histogram'), 'train-only degree fit missing'
    presence = np.array([[0, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0]])
    folds = np.array([0, 0, 1, 2])
    actual = module.train_degree_histogram(presence, folds)
    assert torch.equal(actual, torch.tensor([1, 2, 1, 0]))
    presence[folds != 0] = 0
    assert torch.equal(actual, module.train_degree_histogram(presence, folds))
    assert not actual.requires_grad
    from pna_analysis.model import PNAPredictor
    from pna_analysis.data import concept_graph
    # Training made only of hub-only patients must not make scalers NaN.
    deg = module.train_degree_histogram(np.zeros((2, 3)), np.array([0, 0]))
    model = PNAPredictor(4, 6, deg, width=8, layers=1)
    out = model(concept_graph([], 0, ['med:a', 'cc:b', 'cc:c']))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
