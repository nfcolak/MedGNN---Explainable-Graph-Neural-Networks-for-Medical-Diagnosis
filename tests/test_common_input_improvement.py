"""Regression contract for isolated common-input diagnostics."""
import importlib.util
import numpy as np
import pytest


def module():
    name = 'comparison.standardized.common_input_improvement'
    assert importlib.util.find_spec(name) is not None, 'common-input adapter is missing'
    return __import__(name, fromlist=['*'])


def test_exact_features_hub_edges_and_empty_patient():
    m = module()
    for codes in ([0, 2], []):
        p = m.pyg_graph(codes, 1, 3)
        g = m.graphcare_graph(codes, 1, 3)
        assert np.array_equal(p.x.numpy(), np.eye(4, dtype=np.float32)[g['node_ids'].numpy()])
        assert np.array_equal(p.edge_index.numpy(), g['edge_index'].numpy())
        assert np.array_equal(p.x[-1].numpy(), [0, 0, 0, 1])
        assert g['ehr_nodes'][-1] == 0
        assert g['visit_node'][-1] == 1
        assert p.x.shape == (len(codes) + 1, 4)
        assert p.edge_index.shape == (2, 2 * len(codes))


def test_train_only_fit_retains_all_rows_and_excludes_unsafe_fields():
    import pandas as pd
    m = module()
    assert hasattr(m, 'fit_snapshot'), 'train-only common snapshot builder missing'
    df = pd.DataFrame({'med_train': [1, 0, 0], 'med_heldout': [0, 1, 1],
                       'chiefcomplaint_1': ['train', 'heldout', ''],
                       'lab_future': [99, 1, 2], 'disease_1': ['a', 'b', 'c']})
    names, presence = m.fit_snapshot(df, [0])
    assert names == ['med:train', 'cc:train']
    assert presence.tolist() == [[True, True], [False, False], [False, False]]
    changed = df.copy(); changed.loc[1:, 'chiefcomplaint_1'] = 'another'
    changed.loc[1:, 'med_heldout'] = 0
    assert m.fit_snapshot(changed, [0])[0] == names
    with pytest.raises(ValueError):
        m.fit_snapshot(df, [])


def test_checkpoint_restores_optimizer_rng_and_progress(tmp_path):
    import random
    import torch
    m = module()
    assert hasattr(m, 'save_checkpoint'), 'resumable checkpoint missing'
    torch.manual_seed(1234); np.random.seed(1234); random.seed(1234)
    net = torch.nn.Linear(3, 2); opt = torch.optim.Adam(net.parameters())
    net(torch.ones(2, 3)).sum().backward(); opt.step()
    path = tmp_path / 'last.pt'
    m.save_checkpoint(path, net, opt, {'next_epoch': 1, 'history': [{'epoch': 0}], 'updates': 1})
    expected = (random.random(), np.random.random(), torch.rand(3))
    net2 = torch.nn.Linear(3, 2); opt2 = torch.optim.Adam(net2.parameters())
    progress = m.restore_checkpoint(path, net2, opt2)
    assert progress['next_epoch'] == 1 and progress['updates'] == 1
    assert random.random() == expected[0] and np.random.random() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    for a, b in zip(net.parameters(), net2.parameters()): assert torch.equal(a, b)
    for model, optimizer in ((net, opt), (net2, opt2)):
        optimizer.zero_grad(); model(torch.ones(2, 3)).sum().backward(); optimizer.step()
    for a, b in zip(net.parameters(), net2.parameters()): assert torch.equal(a, b)
    assert not path.with_suffix('.pt.tmp').exists()


@pytest.mark.parametrize('method', ['protgnn', 'gsat', 'graphcare'])
@pytest.mark.parametrize('empty_only', [False, True])
def test_real_model_forward_backward_including_empty(method, empty_only):
    import torch
    from torch_geometric.data import Batch
    m = module()
    assert hasattr(m, 'build_model'), 'common-input real model wiring missing'
    if method == 'graphcare':
        pytest.importorskip('pyhealth', reason='GraphCare real model tested in .venv-graphcare')
    model, _, _, device = m.build_model(method, 3, device='cpu')
    if method == 'graphcare':
        from graphcare_analysis.adapter import _collate
        graphs = [m.graphcare_graph([], 0, 3)]
        if not empty_only: graphs.append(m.graphcare_graph([0, 2], 1, 3))
        b = _collate(graphs)
    else:
        graphs = [m.pyg_graph([], 0, 3)]
        if not empty_only: graphs.append(m.pyg_graph([0, 2], 1, 3))
        b = Batch.from_data_list(graphs)
    model.train()
    logits, y, aux = m.forward_loss(model, method, b, 0, True, device)
    loss = torch.nn.functional.cross_entropy(logits, y) + aux
    assert logits.shape == (len(graphs), 30) and torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_runner_dry_run_occupied_and_resume_binding(tmp_path):
    m = module()
    assert hasattr(m, 'check_attempt'), 'attempt guard missing'
    target = tmp_path / 'cell'
    binding = {'method': 'gsat', 'variant': 'current', 'seed': 1234, 'epochs': 30, 'input_sha256': 'abc'}
    assert m.check_attempt(target, binding, execute=False, resume=False) == 'dry_run'
    assert not target.exists()
    target.mkdir()
    with pytest.raises(FileExistsError): m.check_attempt(target, binding, execute=True, resume=False)
    with pytest.raises(ValueError): m.check_attempt(target, binding, execute=True, resume=True)
    import json
    (target/'state.json').write_text(json.dumps({'status':'interrupted','binding':binding}))
    with pytest.raises(ValueError): m.check_attempt(target, dict(binding, input_sha256='changed'), execute=True, resume=True)
    assert m.check_attempt(target, binding, execute=True, resume=True) == 'resume'
    (target/'state.json').write_text(json.dumps({'status':'completed','binding':binding}))
    assert m.check_attempt(target, binding, execute=True, resume=True) == 'completed'


def test_protgnn_warmup_and_projection_schedule():
    m = module()
    assert hasattr(m, 'training_phase'), 'native lifecycle missing'
    assert m.training_phase(0) == (True, False)
    assert m.training_phase(9) == (True, False)
    assert m.training_phase(10) == (False, False)
    assert m.training_phase(20) == (False, True)
    assert m.training_phase(29) == (False, False)
