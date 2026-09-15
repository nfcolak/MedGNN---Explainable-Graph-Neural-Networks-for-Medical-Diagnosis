import json
import numpy as np
import pandas as pd
import torch
import pytest
from pathlib import Path


def test_train_fit_and_missingness_are_native_not_enrichment():
    from comparison.standardized.native_reference_v1.data import transform_native
    df = pd.DataFrame({'age': [1., 3., 999., np.nan]})
    z, stats = transform_native(df, ['age'], np.array([0, 1]))
    assert stats['age']['mean'] == 2.
    assert stats['age']['std'] == np.sqrt(2.)
    assert z[3, 0] == 0
    with pytest.raises(ValueError, match='scaler'):
        transform_native(df, ['age'], np.arange(4), expected=stats)
    with pytest.raises(ValueError, match='Missing'):
        transform_native(df, ['absent'], np.array([0, 1]))
    with pytest.raises(ValueError, match='target'):
        transform_native(df.assign(disease_1=1), ['disease_1'], np.array([0, 1]))


@pytest.mark.parametrize('method', ['protgnn','gsat','pna','pna_interaction','graphcare'])
def test_actual_models_consume_native_numeric_without_batch_leak(method):
    from comparison.standardized.native_reference_v1.models import build_model, predict, GraphXAIWrapper
    from comparison.standardized.native_reference_v1.data import Reference
    from torch_geometric.data import Batch
    if method == 'graphcare' and int(torch.__version__.split('.')[0]) >= 2:
        pytest.skip('GraphCare is exercised in its actual legacy interpreter')
    elif method != 'graphcare' and int(torch.__version__.split('.')[0]) < 2:
        pytest.skip('Main models are exercised in their actual modern interpreter')
    torch.set_num_threads(2); torch.manual_seed(1234)
    ref = Reference()
    model, _, _ = build_model(method, ref, 'cpu')
    model.eval()
    g, other = ref[0], ref[1]
    g.x.requires_grad_()
    out = predict(model, method, Batch.from_data_list([g]), 0, False)[0]
    grad = torch.autograd.grad(out.square().sum(), g.x)[0]
    assert torch.isfinite(grad).all() and grad[0,210].abs() > 0
    changed = g.clone(); changed.x = changed.x.detach(); changed.x[0,210] += 2
    out2 = predict(model,method,Batch.from_data_list([changed]),0,False)[0]
    assert not torch.allclose(out, out2, atol=1e-7, rtol=1e-7)
    together = predict(model,method,Batch.from_data_list([g,other]),0,False)[0]
    torch.testing.assert_close(out, together[:1], atol=3e-5, rtol=3e-5)
    wrapper=GraphXAIWrapper(model,method,g)
    x=g.x.detach().clone().requires_grad_()
    wrapped=wrapper(x,g.edge_index)
    torch.testing.assert_close(out,wrapped)
    assert torch.autograd.grad(wrapped.square().sum(),x)[0][0,210].abs()>0
    empty = ref[int(np.flatnonzero(np.diff(ref.arrays['node_ptr']) == 1)[0])]
    result=predict(model,method,Batch.from_data_list([empty]),0,False)[0]
    assert torch.isfinite(result).all()


def test_full_default_dry_run_is_no_write(tmp_path):
    from comparison.standardized.native_reference_v1.run import parser, run
    args=parser().parse_args(['--method','protgnn','--output',str(tmp_path/'absent'),'--dry-run'])
    report=run(args)
    assert report['counts']==[59607,7448,7456]
    assert report['selected_counts']==[59607,7448]
    assert not (tmp_path/'absent').exists()


def test_schedule_covers_warmup_projection_and_curriculum():
    from comparison.standardized.native_reference_v1.run import phase
    assert phase(0)==(True,False)
    assert phase(10)==(False,False)
    assert phase(20)==(False,True)
    assert phase(21)==(False,False)
    assert phase(45)==(False,True)


@pytest.mark.parametrize('field',['feature_names','hub_fields','labels','patient_num_stats','split_sha256','source_bindings','native_tensor_sha256'])
def test_mutated_contract_fails_closed_even_if_resigned(tmp_path,field):
    from comparison.standardized.native_reference_v1.data import Reference, DEFAULT, digest
    c=json.loads((DEFAULT/'contract.json').read_text())
    c[field]={'bad':'changed'}
    c.pop('contract_sha256');c['contract_sha256']=digest(c)
    (tmp_path/'contract.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='fingerprint'):
        Reference(tmp_path)


def test_numeric_width_and_missing_metadata_fail_closed():
    from comparison.standardized.native_reference_v1.models import canonical_input
    from comparison.standardized.native_reference_v1.data import Reference
    from torch_geometric.data import Batch
    b=Batch.from_data_list([Reference()[0]])
    b.x=b.x[:,:330]
    with pytest.raises(ValueError,match='331'):
        canonical_input(b)


def test_node_permutation_keeps_numeric_metadata_aligned():
    from comparison.standardized.native_reference_v1.models import build_model,predict
    from comparison.standardized.native_reference_v1.data import Reference
    from torch_geometric.data import Batch
    ref=Reference();g=ref[0]
    methods=['graphcare'] if int(torch.__version__.split('.')[0])<2 else ['protgnn','gsat','pna','pna_interaction']
    for method in methods:
        torch.manual_seed(3);model,_,_=build_model(method,ref);model.eval()
        perm=torch.arange(g.num_nodes-1,-1,-1);inverse=perm.argsort()
        changed=g.clone();changed.x=g.x[perm];changed.node_ids=g.node_ids[perm];changed.node_type=g.node_type[perm];changed.edge_index=inverse[g.edge_index]
        a=predict(model,method,Batch.from_data_list([g]),0,False)[0]
        b=predict(model,method,Batch.from_data_list([changed]),0,False)[0]
        torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)


def test_native_projection_terminal_root_and_gsat_curriculum(tmp_path):
    if int(torch.__version__.split('.')[0])<2:pytest.skip('Modern model mechanisms')
    from comparison.standardized.native_reference_v1.run import project
    from comparison.standardized.native_reference_v1.models import build_model,predict
    from comparison.standardized.native_reference_v1.data import Reference
    from torch_geometric.data import Batch
    ref=Reference();empty=ref[int(np.flatnonzero(np.diff(ref.arrays['node_ptr'])==1)[0])]
    model,_,_=build_model('protgnn',ref)
    records=project(model,[empty],20,tmp_path)
    assert sum(r['terminal_root_scored'] for r in records)>0
    assert any(r['replaced'] for r in records)
    assert torch.isfinite(model.model.prototype_vectors).all()
    gsat,_,_=build_model('gsat',ref)
    assert gsat.get_r(0)==pytest.approx(.9)
    assert gsat.get_r(10)==pytest.approx(.8)
    assert gsat.get_r(20)==pytest.approx(.7)
    out,aux=predict(gsat,'gsat',Batch.from_data_list([ref[0],ref[1]]),20,True)
    assert torch.isfinite(aux) and aux>0
    (out.square().mean()+aux).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in gsat.extractor.parameters())


def test_production_schedule_projection_branch(tmp_path,monkeypatch):
    if int(torch.__version__.split('.')[0])<2:pytest.skip('Modern model mechanisms')
    import comparison.standardized.native_reference_v1.run as runner
    calls=[]
    monkeypatch.setattr(runner,'phase',lambda epoch:(False,True))
    monkeypatch.setattr(runner,'project',lambda model,train,epoch,output:calls.append((len(train),epoch)))
    a=runner.parser().parse_args(['--method','protgnn','--output',str(tmp_path/'run'),'--limit','2','--epochs','1','--execute'])
    runner.run(a)
    assert calls==[(2,0)]
    state=json.loads((tmp_path/'run/run_manifest.json').read_text())
    assert state['status']=='completed' and state['replay']['exact_logits']


def test_legacy_summary_refuses_native_result_mixture():
    from comparison.standardized.summarize import load_compatible_runs
    from comparison.standardized.native_reference_v1.data import REPO
    with pytest.raises(ValueError):
        load_compatible_runs(REPO/'comparison/standardized/native_runs/integration_v1')


def test_graphcare_zero_numeric_recovers_original_and_no_global_mutation():
    if int(torch.__version__.split('.')[0])>=2:pytest.skip('Actual legacy GraphCare interpreter')
    from comparison.standardized.native_reference_v1.models import build_model,graphcare_arguments
    from comparison.standardized.native_reference_v1.data import Reference
    from graphcare_analysis.run import build_graphcare_model
    from torch_geometric.data import Batch
    ref=Reference();numeric,_,_=build_model('graphcare',ref);numeric.eval()
    old=build_graphcare_model({'num_nodes':193,'num_rels':3},30,'cpu');old.eval()
    old.load_state_dict({k:v for k,v in numeric.state_dict().items() if not k.startswith('numeric_encoder.')})
    b=Batch.from_data_list([ref[0],ref[1]])
    args=graphcare_arguments(b);args['numeric']=torch.zeros_like(args['numeric'])
    before=numeric.node_emb.weight.detach().clone()
    y=numeric(**args);legacy_args={k:v for k,v in args.items() if k!='numeric'}
    torch.testing.assert_close(y,old(**legacy_args),atol=0,rtol=0)
    numeric(**graphcare_arguments(b))
    assert torch.equal(before,numeric.node_emb.weight)


def test_source_mismatch_at_real_runner_is_rejected(tmp_path,monkeypatch):
    import comparison.standardized.native_reference_v1.data as data
    from comparison.standardized.native_reference_v1.run import parser,run
    original=data.sha
    monkeypatch.setattr(data,'sha',lambda p:'changed' if Path(p).name=='merged_ed.csv' else original(p))
    a=parser().parse_args(['--method','protgnn','--output',str(tmp_path/'no-output'),'--dry-run'])
    with pytest.raises(ValueError,match='Source/split/cache'):
        run(a)
    assert not a.output.exists()
