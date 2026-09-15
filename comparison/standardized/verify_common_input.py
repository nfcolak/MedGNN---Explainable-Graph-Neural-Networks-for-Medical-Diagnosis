"""Independent all-epoch validation and preservation verifier; no test inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from comparison.standardized.common_input_improvement import ROOT, load_checkpoint
from comparison.standardized.performance_review_20260913.evidence import save


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def independent_metrics(y, pred, prob):
    cm=np.zeros((30,30),dtype=np.int64); np.add.at(cm,(y,pred),1)
    tp=cm.diagonal(); support=cm.sum(1); predicted=cm.sum(0)
    recall=np.divide(tp,support,out=np.zeros(30,dtype=float),where=support!=0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(30,dtype=float),where=support+predicted!=0)
    accuracy=float(tp.sum()/cm.sum())
    metrics={'accuracy':accuracy,'balanced_acc':float(recall.mean()),'macro_f1':float(f1.mean()),'micro_f1':accuracy}
    rank=np.argsort(prob,axis=1,kind='mergesort')[:,::-1]
    for k in (3,5): metrics[f'top{k}_acc']=float((rank[:,:k]==y[:,None]).any(1).mean())
    return {k:round(v,6) for k,v in metrics.items()}


def verify():
    original=json.loads((ROOT/'preservation_before.json').read_text())
    after={p:{'size':Path(p).stat().st_size,'sha256':sha(Path(p))} for p in original}
    assert original==after, 'Protected artifact changed'
    save(ROOT/'preservation_after.json',after)
    a=np.load(ROOT/'inputs_no_identifiers.npz'); val_y=a['y'][a['folds']==1]
    plan=json.loads((ROOT/'plan.json').read_text()); results=[]
    for method,variant in plan['cells']:
        p=ROOT/'trials'/method/variant
        state=json.loads((p/'state.json').read_text()); assert state['status']=='completed'
        history=json.loads((p/'history.json').read_text()); assert len(history)==30
        assert [r['epoch'] for r in history]==list(range(30))
        assert state['test_evaluated'] is False and state['n_validation']==7448
        for i,row in enumerate(history):
            assert row['updates']==(i+1)*466
            a=np.load(p/f'validation_epoch_{i:02d}.npz')
            y,pred,prob=a['y'],a['pred'],a['probability']
            assert np.array_equal(y,val_y) and prob.shape==(7448,30)
            assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=1e-5)
            assert np.array_equal(pred,prob.argmax(1))
            m=independent_metrics(y,pred,prob)
            for k,v in m.items(): assert abs(v-row[k])<=1e-6,(method,variant,i,k)
        last=load_checkpoint(p/'last.pt'); best=load_checkpoint(p/'best.pt')
        assert last['progress']['next_epoch']==30 and last['progress']['updates']==13980
        assert len(last['optimizer']['state'])>0
        winner=max(history,key=lambda r:r['macro_f1'])
        assert best['epoch']==winner['epoch']==state['best_epoch']
        assert last['progress']['best_epoch']==winner['epoch']
        row={'method':method,'variant':variant,'n_validation':7448,'best_validation':winner,
             'last_validation':history[-1],'parameter_count':state['parameter_count'],
             'last_checkpoint_next_epoch':30,'optimizer_updates':13980,
             'best_checkpoint_sha256':sha(p/'best.pt'),'last_checkpoint_sha256':sha(p/'last.pt')}
        if method=='protgnn':
            projection=json.loads((p/'projection_epoch_20.json').read_text())
            assert projection['train_only'] and len(projection['prototypes'])==90
            assert all(r['train_candidates']==10 for r in projection['prototypes'])
            row['projection_replaced']=sum(r['replaced'] for r in projection['prototypes'])
        if method=='gsat':
            assert [history[i]['curriculum_r'] for i in (0,10,20)]==[.9,.8,.7]
        results.append(row)
    save(ROOT/'results_verified.json',results)
    summary={'status':'verified','protected_files_unchanged':len(original),
             'completed_cells':len(results),'epochs_each':30,'epoch_predictions_independently_recomputed':150,
             'n_train':59607,'n_validation':7448,'test_inferences':0,'temporal_clean':False}
    save(ROOT/'verification.json',summary)
    print(json.dumps({'verification':summary,'results':results},indent=2))


def replay(method):
    import torch
    from comparison.standardized.common_input_improvement import pyg_graph,graphcare_graph,build_model,forward_loss
    torch.set_num_threads(4)
    a=np.load(ROOT/'inputs_no_identifiers.npz'); val_indices=np.flatnonzero(a['folds']==1)
    make=graphcare_graph if method=='graphcare' else pyg_graph
    val=[make(np.flatnonzero(a['presence'][i]),int(a['y'][i]),192) for i in val_indices]
    if method=='graphcare':
        from torch.utils.data import DataLoader
        from graphcare_analysis.adapter import _collate
        loader=DataLoader(val,batch_size=128,collate_fn=_collate,shuffle=False)
    else:
        from torch_geometric.loader import DataLoader
        loader=DataLoader(val,batch_size=128,shuffle=False)
    results=[]
    for variant in (['current'] if method=='gsat' else ['current','sqrt_inverse']):
        p=ROOT/'trials'/method/variant
        state=json.loads((p/'state.json').read_text()); assert state['status']=='completed'
        model,_,_,device=build_model(method,192,device=state['device'])
        for which in ('best','last'):
            ck=load_checkpoint(p/(which+'.pt'))
            epoch=ck['epoch'] if which=='best' else ck['progress']['next_epoch']-1
            model.load_state_dict(ck['state_dict'] if which=='best' else ck['model'],strict=True)
            model.eval(); probs=[]; labels=[]
            with torch.no_grad():
                for batch in loader:
                    logits,y,_=forward_loss(model,method,batch,epoch,False,device)
                    probs.append(logits.softmax(1).cpu().numpy()); labels.append(y.cpu().numpy())
            prob=np.concatenate(probs); yy=np.concatenate(labels)
            saved=np.load(p/f'validation_epoch_{epoch:02d}.npz')
            difference=float(np.abs(prob-saved['probability']).max())
            assert np.array_equal(yy,saved['y'])
            assert np.allclose(prob,saved['probability'],atol=2e-5,rtol=2e-4),difference
            assert np.array_equal(prob.argmax(1),saved['pred'])
            results.append({'method':method,'variant':variant,'checkpoint':which,'epoch':epoch,
                            'max_probability_difference':difference,'n_validation':len(yy),
                            'metrics':independent_metrics(yy,prob.argmax(1),prob)})
    save(ROOT/f'checkpoint_replay_{method}.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--replay',choices=['protgnn','gsat','graphcare'])
    args=p.parse_args()
    replay(args.replay) if args.replay else verify()
