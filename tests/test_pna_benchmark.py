"""Full-budget runner: deterministic batch-boundary optimizer resume."""
import importlib.util
import numpy as np
import pytest
import torch
from pna_analysis.data import train_degree_histogram


def runner():
    assert importlib.util.find_spec('comparison.standardized.pna_benchmark'), 'full benchmark runner missing'
    from comparison.standardized import pna_benchmark
    return pna_benchmark


def bundle():
    presence = np.array([[1,1],[1,0],[0,1],[0,0]] * 6, dtype=np.uint8)
    folds = np.array([0]*12+[1]*6+[2]*6)
    return dict(presence=presence, y=np.arange(24)%6, folds=folds,
                contract={'names':['med:a','cc:b'], 'classes':list('abcdef')},
                degree_histogram=train_degree_histogram(presence,folds),preservation={})


def test_batch_boundary_resume_matches_uninterrupted(tmp_path):
    r = runner(); b = bundle()
    c = {**r.CONFIG, 'epochs':2, 'batch_size':5, 'width':8, 'threads':1}
    full = r.train_cell(tmp_path/'full', b, 'interaction', c)
    with pytest.raises(InterruptedError):
        r.train_cell(tmp_path/'resume', b, 'interaction', c, interrupt_after=2)
    state = torch.load(tmp_path/'resume'/'last.pt', weights_only=False)
    assert state['steps']==2 and state['offset']==10
    resumed = r.train_cell(tmp_path/'resume', b, 'interaction', c, resume=True)
    assert resumed['status']==full['status']=='completed'
    assert resumed['optimizer_steps']==6
    assert [h['n_train'] for h in resumed['history']]==[12,12]
    a = torch.load(tmp_path/'full'/'last.pt', weights_only=False)
    z = torch.load(tmp_path/'resume'/'last.pt', weights_only=False)
    for key in a['model']:
        assert torch.equal(a['model'][key], z['model'][key]), key
    assert resumed['final_replay']['arrays_equal'] is True
    assert resumed['selected_metrics']==full['selected_metrics']
    assert resumed['replay']['logits_max_abs_diff']==0
    with pytest.raises(FileExistsError):
        r.train_cell(tmp_path/'full',b,'plain',c)
    with pytest.raises(ValueError, match='binding'):
        r.train_cell(tmp_path/'resume',b,'interaction',{**c,'lr':.02},resume=True)
