"""Production full-cohort runner. Bounded proof uses this exact path with --limit.

No automatic test evaluation; selection/replay use validation. Epoch-atomic resume
replays an interrupted partial epoch from its committed optimizer/RNG checkpoint.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import numpy as np
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from .data import Reference, DEFAULT, REPO, digest, sha, save
from .models import METHODS, build_model, predict
from comparison.standardized.common_input_improvement import save_checkpoint, restore_checkpoint, load_checkpoint
from comparison.standardized.performance_review import class_weights
from shared.lib.metrics import multiclass_metrics


def phase(epoch):
    from protgnn_analysis.config import train_args
    return epoch < train_args.warm_epochs, epoch >= train_args.proj_epochs and (epoch-train_args.proj_epochs)%train_args.proj_interval==0


def project(model, dataset, epoch, output):
    """Native MCTS with explicit whole-graph scoring for terminal-size roots only.

    Native MCTS leaves a <=min_atoms root unscored at P=0. Here its only available
    coalition is scored exactly, without dropping hub-only patients or changing
    legacy code. Larger graphs execute the unchanged native MCTS search.
    """
    from protgnn_analysis.my_mcts import mcts, gnn_prot_score
    from protgnn_analysis.config import train_args, mcts_args
    from torch_geometric.data import Batch
    model.eval(); order=list(range(len(dataset)));random.shuffle(order); report=[]
    per_class=model.model.prototype_class_identity.shape[0]//30
    for p in range(model.model.prototype_vectors.shape[0]):
        best=-float('inf'); replacement=None; count=0; terminal=0
        for i in order:
            g=dataset[i]
            if int(g.y)!=p//per_class: continue
            count+=1
            with torch.no_grad():
                if g.num_nodes<=mcts_args.min_atoms:
                    terminal+=1
                    bat=Batch.from_data_list([g])
                    score=gnn_prot_score(list(range(g.num_nodes)),bat,model,model.model.prototype_vectors[p])
                    emb=model(bat)[3]
                else:
                    _,score,emb=mcts(g,model,model.model.prototype_vectors[p])
                if score>best: best=score;replacement=emb.reshape(-1).detach().clone()
            if count>=train_args.nearest_graphs: break
        if replacement is not None:
            with torch.no_grad(): model.model.prototype_vectors[p].copy_(replacement)
        report.append({'prototype':p,'candidates':count,'terminal_root_scored':terminal,'replaced':replacement is not None})
    save(output/f'projection_{epoch:03d}.json',{'epoch':epoch,'policy':'native MCTS + explicitly scored terminal root v1','prototypes':report})
    return report


def source_bindings():
    paths=[]
    for directory in ['comparison/standardized/native_reference_v1','protgnn_analysis','gsat_analysis','graphcare_analysis','pna_analysis','shared/lib','external/GraphCare/graphcare_']:
        paths.extend(p for p in (REPO/directory).rglob('*.py') if 'outputs' not in p.parts)
    paths.extend(REPO/p for p in ['comparison/standardized/train_identical.py','comparison/standardized/common_input_improvement.py','comparison/standardized/performance_review.py'])
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(set(paths))}


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--artifact',type=Path,default=DEFAULT)
    p.add_argument('--expected-contract',default=None)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--seed',type=int,default=1234)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--loss',choices=['native','ce','sqrt_inverse'],default='ce')
    p.add_argument('--limit',type=int,default=None,help='Explicit bounded wiring scope; first N train and val ordinals, never benchmark')
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--replay',action='store_true')
    p.add_argument('--stop-after',type=int,default=None)
    return p


def evaluate(model, method, loader, epoch):
    model.eval(); logits=[]; ys=[]; ordinals=[]
    with torch.no_grad():
        for batch in loader:
            result,_=predict(model,method,batch,epoch,False)
            if not torch.isfinite(result).all(): raise ValueError('Nonfinite validation logits')
            logits.append(result.cpu());ys.append(batch.y);ordinals.append(batch.graph_ordinal)
    ll=torch.cat(logits); y=torch.cat(ys).numpy();p=ll.softmax(1).numpy()
    return ll.numpy(), y, torch.cat(ordinals).numpy(), multiclass_metrics(y,p.argmax(1),p)


def replay(model, method, loader, output, binding):
    ck=load_checkpoint(output/'best.pt')
    if ck['binding']!=binding: raise ValueError('Checkpoint binding mismatch')
    model.load_state_dict(ck['model'])
    logits,y,ordinals,metrics=evaluate(model,method,loader,ck['epoch'])
    prior=np.load(output/f"validation_{ck['epoch']:03d}.npz")
    if not np.array_equal(y,prior['y']) or not np.array_equal(ordinals,prior['ordinals']): raise ValueError('Replay cohort mismatch')
    np.testing.assert_allclose(logits,prior['logits'],rtol=0,atol=0)
    if metrics!=ck['metrics']: raise ValueError('Replayed metrics mismatch')
    proof={'exact_logits':True,'selected_epoch':ck['epoch'],'validation_count':len(y),'checkpoint_sha256':sha(output/'best.pt'),'test_evaluated':False}
    save(output/'replay.json',proof); return proof


def run(args):
    if args.epochs<1 or args.batch_size<1 or (args.limit is not None and args.limit<1): raise ValueError('Positive epochs/batch/limit required')
    if args.execute and args.dry_run: raise ValueError('Choose execute or dry-run')
    ref=Reference(args.artifact,expected=args.expected_contract)
    torch.set_num_threads(2);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    tr_idx=ref.fold(0,args.limit);va_idx=ref.fold(1,args.limit)
    model,lr,wd=build_model(args.method,ref,'cpu')
    policy=('inverse' if args.method=='protgnn' else 'none') if args.loss=='native' else ('none' if args.loss=='ce' else 'sqrt_inverse')
    code=source_bindings()
    binding={'contract_sha256':ref.fingerprint,'source_code':code,'method':args.method,'epochs':args.epochs,
        'seed':args.seed,'batch_size':args.batch_size,'loss':args.loss,'weight_policy':policy,'limit':args.limit,
        'train_ordinals_sha256':digest(tr_idx.tolist()),'validation_ordinals_sha256':digest(va_idx.tolist()),
        'learning_rate':lr,'weight_decay':wd,'torch':torch.__version__,
        'projection_policy':'native MCTS + scored terminal root v1','selection':'validation macro_f1; full fixed epoch budget; no early stop'}
    report={'method':args.method,'counts':ref.contract['counts'],'selected_counts':[len(tr_idx),len(va_idx)],
        'parameters':sum(p.numel() for p in model.parameters()),'contract_sha256':ref.fingerprint,
        'scope':'bounded_wiring_not_benchmark' if args.limit else 'full_cohort','binding':binding}
    if not args.execute:
        print(json.dumps({k:v for k,v in report.items() if k!='binding'}));return report
    output=args.output.resolve()
    if output.exists() and not (args.resume or args.replay): raise FileExistsError('Occupied output: explicit --resume or --replay required')
    if not output.exists() and (args.resume or args.replay): raise ValueError('Cannot resume missing output')
    train=Subset(ref,tr_idx.tolist());val=Subset(ref,va_idx.tolist())
    va=DataLoader(val,batch_size=args.batch_size,shuffle=False)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    weights=torch.tensor(class_weights(ref.arrays['y'][ref.arrays['folds']==0],30,policy),dtype=torch.float32)
    progress={'next_epoch':0,'updates':0,'best':-1.,'history':[],'binding':binding,'best_checkpoint':None}
    if output.exists():
        state=json.loads((output/'run_manifest.json').read_text())
        if state['binding']!=binding: raise ValueError('Run input/source/split/feature/scaler/label/topology/code binding differs')
        if args.replay:
            print(json.dumps(replay(model,args.method,va,output,binding)));return report
        progress=restore_checkpoint(output/'last.pt',model,opt)
        if progress['binding']!=binding: raise ValueError('Resume checkpoint binding mismatch')
        if progress['best_checkpoint'] is not None: torch.save(progress['best_checkpoint'],output/'best.pt')
        if state['status']=='completed':
            replay(model,args.method,va,output,binding);return report
    else:
        output.mkdir(parents=True)
        for path in code:
            dst=output/'source_snapshot'/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/path,dst)
        np.savez_compressed(output/'cohort.npz',train_ordinals=tr_idx,validation_ordinals=va_idx)
        save_checkpoint(output/'last.pt',model,opt,progress)
    state={**report,'status':'running','checkpoint_path':'best.pt','metrics_path':'history.json','test_evaluated':False,'pid':os.getpid(),'command':sys.argv}
    save(output/'run_manifest.json',state)
    import fcntl
    lock=(output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        for epoch in range(progress['next_epoch'],args.epochs):
            warm,projection=phase(epoch)
            if args.method=='protgnn':
                if projection: project(model,train,epoch,output)
                for p in model.parameters():p.requires_grad=True
                for p in model.model.last_layer.parameters():p.requires_grad=not warm
            # Independent deterministic sampler: architecture RNG consumption cannot change cohort/order.
            generator=torch.Generator().manual_seed(args.seed+epoch)
            tr=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=generator)
            model.train();seen=0;total=0.;aux_total=0.;sequence=[]
            for batch in tr:
                result,aux=predict(model,args.method,batch,epoch,True)
                loss=torch.nn.functional.cross_entropy(result,batch.y,weight=weights)+aux
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss; no bypass')
                opt.zero_grad();loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('Nonfinite gradient')
                if args.method=='protgnn':torch.nn.utils.clip_grad_value_(model.parameters(),2.)
                opt.step();seen+=batch.y.numel();progress['updates']+=1
                total+=float(loss.detach())*batch.y.numel();aux_total+=float(aux)*batch.y.numel()
                sequence.extend(batch.graph_ordinal.tolist())
            if seen!=len(tr_idx) or sorted(sequence)!=sorted(tr_idx.tolist()):raise ValueError('Training cohort mismatch')
            ll,y,ordinals,metrics=evaluate(model,args.method,va,epoch)
            if not np.array_equal(ordinals,va_idx): raise ValueError('Validation cohort mismatch')
            row={'epoch':epoch,'updates':progress['updates'],'train_count':seen,'validation_count':len(y),
                'train_loss':total/seen,'aux_loss':aux_total/seen,'order_sha256':digest(sequence),'validation':metrics,
                'prototype_warmup':warm if args.method=='protgnn' else None,'projection_executed':projection if args.method=='protgnn' else False,
                'gsat_r':float(model.get_r(epoch)) if args.method=='gsat' else None}
            np.savez_compressed(output/f'validation_{epoch:03d}.npz',logits=ll,y=y,ordinals=ordinals)
            if metrics['macro_f1']>progress['best']:
                progress['best']=metrics['macro_f1']
                progress['best_checkpoint']={'model':copy.deepcopy(model.state_dict()),'epoch':epoch,'binding':binding,'metrics':metrics}
                torch.save(progress['best_checkpoint'],output/'best.pt')
            progress['next_epoch']=epoch+1;progress['history'].append(row)
            save_checkpoint(output/'last.pt',model,opt,progress);save(output/'history.json',progress['history'])
            state.update(next_epoch=epoch+1,updates=progress['updates']);save(output/'run_manifest.json',state)
            print(json.dumps({'method':args.method,'epoch':epoch,'updates':progress['updates'],'train_count':seen,'validation_count':len(y)}),flush=True)
            if args.stop_after and epoch+1>=args.stop_after and epoch+1<args.epochs:
                state['status']='interrupted';save(output/'run_manifest.json',state);return report
        proof=replay(model,args.method,va,output,binding)
        # Re-read exact original input/source bindings before claiming completion.
        Reference(args.artifact,expected=ref.fingerprint)
        if source_bindings()!=code: raise ValueError('Source code changed during run')
        state.update(status='completed',replay=proof);save(output/'run_manifest.json',state)
    except BaseException as exc:
        state.update(status='failed',error=type(exc).__name__+': '+str(exc));save(output/'run_manifest.json',state);raise
    finally:
        lock.close()
    return report


if __name__=='__main__':run(parser().parse_args())
