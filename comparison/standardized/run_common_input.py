"""Bounded 30-epoch validation-only evidence runner, fixed isolated namespace.

Epoch-atomic optimizer/RNG resume; immutable epoch predictions/checkpoints. Any
interrupted partial epoch is replayed from its last committed epoch. No test
loader is constructed. Requires --execute; --stop-after is a checkpoint smoke,
not a changed epoch budget. Run main-module identity is not serialized.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from comparison.standardized.common_input_improvement import (
    ROOT, pyg_graph, graphcare_graph, build_model, forward_loss, check_attempt,
    save_checkpoint, restore_checkpoint, training_phase, load_checkpoint)
from comparison.standardized.performance_review import class_weights
from comparison.standardized.performance_review_20260913.evidence import save
from shared.lib.config_base import set_seed
from shared.lib.metrics import multiclass_metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(model, train_graphs, target, epoch):
    from protgnn_analysis.my_mcts import mcts
    from protgnn_analysis.config import train_args, model_args
    assert train_args.nearest_graphs == 10
    model.eval(); candidates=list(range(len(train_graphs))); random.shuffle(candidates)
    result=[]; start=time.time()
    for p in range(30 * model_args.num_prototypes_per_class):
        cls=p // model_args.num_prototypes_per_class
        count=0; best=0.; projected=None
        for i in candidates:
            d=train_graphs[i]
            if int(d.y.item()) != cls: continue
            count+=1
            with torch.no_grad():
                _, similarity, emb=mcts(d,model,model.model.prototype_vectors[p])
            if similarity > best: best=similarity; projected=emb
            if count >= train_args.nearest_graphs: break
        if projected is not None:
            model.model.prototype_vectors.data[p] = projected
        result.append({'prototype':p,'class':cls,'train_candidates':count,'best_similarity':best,'replaced':projected is not None})
        print(json.dumps({'projection_epoch':epoch,'prototype_done':p+1,'seconds':time.time()-start}),flush=True)
    save(target/f'projection_epoch_{epoch:02d}.json',{'epoch':epoch,'method':'original MCTS','train_only':True,'prototypes':result,'seconds':time.time()-start})


def run(args):
    root=ROOT
    target=root/'trials'/args.method/args.variant
    if args.method=='gsat' and args.variant!='current': raise ValueError('GSAT control only')
    contract=json.loads((root/'input_contract_and_audit.json').read_text())
    assert contract['audited_graphs']==74511 and contract['effective_feature_dim']==193
    assert contract['temporal_clean'] is False
    assert sha(root/'inputs_no_identifiers.npz') == contract['input_sha256']
    binding={'method':args.method,'variant':args.variant,'seed':1234,'epochs':30,
             'batch_size':128,'input_sha256':contract['input_sha256'],'split_sha256':contract['split_sha256'],
             'harness_sha256':sha(Path(__file__)),
             'adapter_sha256':sha(Path('comparison/standardized/common_input_improvement.py'))}
    action=check_attempt(target,binding,args.execute,args.resume)
    if action=='dry_run':
        print(json.dumps({'action':action,'target':str(target),'binding':binding})); return
    target.mkdir(parents=True,exist_ok=True)
    lock=(target/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    if action=='completed':
        state=json.loads((target/'state.json').read_text())
        ck=load_checkpoint(target/'last.pt')
        assert ck['progress']['next_epoch']==30
        assert len(json.loads((target/'history.json').read_text()))==30
        print(json.dumps({'status':'already_completed','method':args.method})); return
    state={'status':'running','binding':binding,'pid':os.getpid(),'command':sys.argv,
           'n_train':59607,'n_validation':7448,'test_evaluated':False,
           'scope':contract['name'],'selection':'validation macro_f1','checkpoint_path':'best.pt',
           'metrics_path':'history.json','resume_action':action}
    def event(name, **extra):
        with (target/'events.jsonl').open('a') as f:
            f.write(json.dumps({'event':name,'unix_time':time.time(),**extra})+'\n')
    event(action,pid=os.getpid()); save(target/'state.json',state)
    set_seed(1234); torch.set_num_threads(4)
    a=np.load(root/'inputs_no_identifiers.npz')
    pres,y,folds=a['presence'],a['y'],a['folds']
    assert pres.shape==(74511,192) and np.bincount(folds).tolist()==[59607,7448,7456]
    model,lr,wd,device=build_model(args.method,192)
    # Only train/validation are instantiated. Inputs carry no patient identifiers.
    make=graphcare_graph if args.method=='graphcare' else pyg_graph
    train=[make(np.flatnonzero(pres[i]),int(y[i]),192) for i in np.flatnonzero(folds==0)]
    val=[make(np.flatnonzero(pres[i]),int(y[i]),192) for i in np.flatnonzero(folds==1)]
    if args.method=='graphcare':
        from torch.utils.data import DataLoader
        from graphcare_analysis.adapter import _collate
        tr=DataLoader(train,batch_size=128,shuffle=True,collate_fn=_collate)
        va=DataLoader(val,batch_size=128,shuffle=False,collate_fn=_collate)
    else:
        from torch_geometric.loader import DataLoader
        tr=DataLoader(train,batch_size=128,shuffle=True)
        va=DataLoader(val,batch_size=128,shuffle=False)
    policy=('inverse' if args.method=='protgnn' else 'none') if args.variant=='current' else 'sqrt_inverse'
    weight=torch.tensor(class_weights(y[folds==0],30,policy),dtype=torch.float32,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    state.update({'device':str(device),'lr':lr,'weight_decay':wd,'class_weight_policy':policy,
                  'parameter_count':sum(p.numel() for p in model.parameters()),'updates_per_epoch':len(tr)})
    progress={'next_epoch':0,'history':[],'best':-1.,'best_epoch':None,'updates':0}
    if action=='resume':
        progress=restore_checkpoint(target/'last.pt',model,opt)
        if progress.get('best_checkpoint') is not None:
            torch.save(progress['best_checkpoint'], target/'best.pt')
        save(target/'history.json',progress['history'])
        event('restored',next_epoch=progress['next_epoch'],updates=progress['updates'])
    else:
        save_checkpoint(target/'last.pt',model,opt,progress)
    save(target/'state.json',state)
    t=time.time()
    try:
        for epoch in range(progress['next_epoch'],30):
            if args.method=='protgnn':
                warm,projection=training_phase(epoch)
                if projection: project(model,train,target,epoch)
                for p in model.parameters(): p.requires_grad=True
                for p in model.model.last_layer.parameters(): p.requires_grad=not warm
            model.train(); loss_sum=0.; seen=0
            for batch in tr:
                logits,labels,aux=forward_loss(model,args.method,batch,epoch,True,device)
                loss=torch.nn.functional.cross_entropy(logits,labels,weight=weight)+aux
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss; no fallback or bypass')
                opt.zero_grad(); loss.backward()
                if args.method=='protgnn': torch.nn.utils.clip_grad_value_(model.parameters(),2.)
                opt.step(); progress['updates']+=1
                loss_sum+=float(loss.detach())*len(labels); seen+=len(labels)
            assert seen==59607
            model.eval(); ys=[]; probabilities=[]
            with torch.no_grad():
                for batch in va:
                    logits,labels,_=forward_loss(model,args.method,batch,epoch,False,device)
                    ys.append(labels.cpu().numpy()); probabilities.append(logits.softmax(1).cpu().numpy())
            yy=np.concatenate(ys); pp=np.concatenate(probabilities); pred=pp.argmax(1)
            assert len(yy)==7448 and np.array_equal(yy,y[folds==1])
            metrics=multiclass_metrics(yy,pred,pp)
            row={'epoch':epoch,'epoch_one_based':epoch+1,'train_loss':loss_sum/seen,
                 'updates':progress['updates'],'seconds_this_invocation':time.time()-t,**metrics}
            if args.method=='gsat':
                row['curriculum_r']=float(model.get_r(epoch)) if hasattr(model,'get_r') else None
            progress['history'].append(row)
            np.savez_compressed(target/f'validation_epoch_{epoch:02d}.npz',y=yy,pred=pred,probability=pp)
            if metrics['macro_f1']>progress['best']:
                progress['best']=metrics['macro_f1']; progress['best_epoch']=epoch
                # Best checkpoint is reconstructable from its epoch-specific copy.
                ck={'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,'binding':binding,'validation':metrics}
                progress['best_checkpoint']=ck
                tmp=target/'best.pt.tmp'; torch.save(ck,tmp); tmp.replace(target/'best.pt')
                save(target/'class_report.json',classification_report(yy,pred,labels=list(range(30)),output_dict=True,zero_division=0))
                save(target/'confusion.json',confusion_matrix(yy,pred,labels=list(range(30))).tolist())
            progress['next_epoch']=epoch+1
            save_checkpoint(target/'last.pt',model,opt,progress)
            save(target/'history.json',progress['history'])
            state.update({'next_epoch':epoch+1,'updates':progress['updates'],'best_epoch':progress['best_epoch'],
                          'best_validation':progress['history'][progress['best_epoch']]})
            save(target/'state.json',state)
            event('epoch_committed',epoch=epoch,updates=progress['updates'])
            print(json.dumps({'method':args.method,'variant':args.variant,**row}),flush=True)
            if args.stop_after and epoch+1 >= args.stop_after and epoch+1 < 30:
                state['status']='interrupted'; save(target/'state.json',state)
                event('bounded_stop',next_epoch=epoch+1); return
        assert progress['updates']==30*len(tr)
        state['status']='completed'; save(target/'state.json',state)
        event('completed',updates=progress['updates'])
    except BaseException as exc:
        state['status']='failed'; state['error']=type(exc).__name__+': '+str(exc)
        save(target/'state.json',state); event('failed',error=state['error']); raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--method',required=True,choices=['protgnn','graphcare','gsat'])
    p.add_argument('--variant',required=True,choices=['current','sqrt_inverse'])
    p.add_argument('--execute',action='store_true'); p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int,choices=range(1,30))
    run(p.parse_args())
