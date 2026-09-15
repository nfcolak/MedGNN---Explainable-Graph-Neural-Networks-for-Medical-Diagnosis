"""Opt-in paired FULL-data PNA benchmark. No test graph construction/evaluation.

CPU, deterministic updates; atomic checkpoint each batch. Resume restores the
permutation, next offset, optimizer and all RNGs. Existing smoke sources untouched.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Batch
from sklearn.metrics import precision_recall_fscore_support

from pna_analysis.data import load_common_input
from pna_analysis.train import ROOT, build_predictor, code_hashes, graph_subset, predict
from shared.lib.benchmark_contract import file_sha256

CONFIG = dict(seed=1234, epochs=30, batch_size=128, width=64, layers=2, rank=16,
              lr=.001, weight_decay=1e-5, threads=4, cross_pairs=True,
              no_messages=False, loss='ce', device='cpu', drop_last=False,
              selection='first maximum shared macro_f1 (six decimal convention)')
STOP = False


def atomic_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    os.replace(temp, path)


def atomic_torch(path, value):
    temp = Path(str(path)+'.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def binding(bundle, config, architecture):
    codes = code_hashes()
    codes[str(Path(__file__).resolve().relative_to(ROOT))] = file_sha256(__file__)
    codes['shared/lib/benchmark_contract.py'] = file_sha256(ROOT/'shared/lib/benchmark_contract.py')
    return dict(config=config, architecture=architecture, code=codes,
                input=bundle['contract'], protected=bundle['preservation'],
                degree=bundle['degree_histogram'].tolist(),
                arrays={k:hashlib.sha256(np.ascontiguousarray(bundle[k]).tobytes()).hexdigest()
                        for k in ('presence','y','folds')})


def train_cell(target, bundle, architecture, config=None, resume=False,
               interrupt_after=None, graphs=None):
    """One bounded cell; interrupt_after is a synthetic-test hook, not CLI scope."""
    config = dict(CONFIG if config is None else config)
    config['interactions'] = architecture == 'interaction'
    target = Path(target)
    if target.exists() and not resume:
        raise FileExistsError(target)
    target.mkdir(parents=True, exist_ok=True)
    with (target/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _locked_cell(target,bundle,architecture,config,resume,interrupt_after,graphs)


def _locked_cell(target,bundle,architecture,config,resume,interrupt_after,graphs):
    bound = binding(bundle,config,architecture)
    torch.set_num_threads(config['threads']); torch.use_deterministic_algorithms(True)
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    model = build_predictor(config,bundle)
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(config['seed'])
    rows = [np.flatnonzero(bundle['folds']==f) for f in (0,1)]
    if graphs is None:
        graphs = [graph_subset(bundle,r) for r in rows]
    train,val = graphs
    if [len(train),len(val)] != [len(r) for r in rows]:
        raise ValueError('graph fold size mismatch')
    nclasses = len(bundle['contract']['classes'])
    if set(bundle['y'][rows[1]]) != set(range(nclasses)):
        raise ValueError('Validation must contain every canonical class for shared macro metrics')
    state = dict(epoch=0,offset=0,order=None,steps=0,history=[],best_score=-1.,best_epoch=None,
                 epoch_loss=0.,epoch_train_seconds=0.,seconds=0.)
    manifest = dict(schema='medgnn.pna-full-paired-v1',status='running',architecture=architecture,
                    binding=bound,n_train=len(train),n_validation=len(val),test_evaluated=False,
                    topology='star',parameter_count=sum(p.numel() for p in model.parameters()),
                    checkpoint_path='best.pt',last_checkpoint_path='last.pt',metrics_path='metrics.json',
                    runtime=dict(python=sys.version,torch=str(torch.__version__),pyg=str(torch_geometric.__version__),numpy=np.__version__),
                    pid=os.getpid(),class_counts_train=np.bincount(bundle['y'][rows[0]],minlength=nclasses).tolist())
    if resume:
        saved = torch.load(target/'last.pt',map_location='cpu',weights_only=False)
        if saved['binding'] != bound:
            raise ValueError('Resume binding mismatch')
        prior = json.loads((target/'run_manifest.json').read_text())
        if prior['status']=='completed':
            return prior
        if prior['status']=='failed':
            raise ValueError('Failed cell requires investigation; automatic retry forbidden')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        state = saved['state']
        random.setstate(saved['python_rng']); np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng']); generator.set_state(saved['order_rng'])
        with (target/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(event='interrupted_to_running',previous_status=prior['status'],steps=state['steps']))+'\n')
    else:
        atomic_json(target/'config.json',config)
        atomic_json(target/'preservation_before.json',bundle['preservation'])
        (target/'git_status_before.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
        snapshot = target/'source_snapshot'; snapshot.mkdir()
        for rel in bound['code']:
            dest = snapshot/rel; dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((ROOT/rel).read_bytes())
    start = time.monotonic(); previous_seconds=state['seconds']

    def persist(status='running', **extra):
        state['seconds']=previous_seconds+time.monotonic()-start
        payload=dict(binding=bound,state=state,model=model.state_dict(),optimizer=optimizer.state_dict(),
                     python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                     order_rng=generator.get_state(),steps=state['steps'],offset=state['offset'],
                     train_indices=torch.from_numpy(rows[0]),validation_indices=torch.from_numpy(rows[1]))
        atomic_torch(target/'last.pt',payload)
        manifest.update(status=status,optimizer_steps=state['steps'],epoch=state['epoch'],
                        next_offset=state['offset'],seconds=state['seconds'],history=state['history'],**extra)
        atomic_json(target/'run_manifest.json',manifest)
        return payload

    persist()
    try:
        while state['epoch'] < config['epochs']:
            if state['order'] is None:
                state['order']=torch.randperm(len(train),generator=generator)
            model.train()
            while state['offset'] < len(train):
                tick=time.monotonic()
                indices=state['order'][state['offset']:state['offset']+config['batch_size']].tolist()
                batch=Batch.from_data_list([train[i] for i in indices])
                optimizer.zero_grad(set_to_none=True)
                logits=model(batch,cross_pairs=True,no_messages=False)
                loss=torch.nn.functional.cross_entropy(logits,batch.y)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('Nonfinite gradient')
                pair_grad=sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if '.pair.' in n and p.grad is not None)
                optimizer.step()
                elapsed=time.monotonic()-tick
                state['steps']+=1; state['offset']+=len(indices)
                state['epoch_loss']+=float(loss.detach())*len(indices)
                state['epoch_train_seconds']+=elapsed
                persist()
                event=dict(event='optimizer_step',architecture=architecture,epoch=state['epoch']+1,
                           step=state['steps'],seen=state['offset'],n_train=len(train),loss=float(loss.detach()),
                           batch_seconds=elapsed,pair_gradient_l1=pair_grad)
                if state['steps']<=10 or state['steps']%25==0:
                    print(json.dumps(event),flush=True)
                    with (target/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
                if STOP or (interrupt_after is not None and state['steps']==interrupt_after):
                    raise InterruptedError('Checkpointed stop at optimizer boundary')
            tick=time.monotonic()
            arrays,metrics=predict(model,val,config)
            epoch=state['epoch']+1
            record=dict(epoch=epoch,n_train=state['offset'],optimizer_steps=state['steps'],
                        train_loss=state['epoch_loss']/len(train),train_seconds=state['epoch_train_seconds'],
                        validation_seconds=time.monotonic()-tick,metrics=metrics)
            if metrics['macro_f1'] > state['best_score']:
                state['best_score']=metrics['macro_f1']; state['best_epoch']=epoch
                atomic_torch(target/'best.pt',dict(model=model.state_dict(),binding=bound,epoch=epoch,metrics=metrics))
                np.savez_compressed(target/'best_validation.npz',**arrays)
            np.savez_compressed(target/'final_validation.npz',**arrays)
            state['history'].append(record); state['epoch']=epoch
            state.update(offset=0,order=None,epoch_loss=0.,epoch_train_seconds=0.)
            persist(); atomic_json(target/'history.json',state['history'])
            print(json.dumps(dict(event='epoch_completed',architecture=architecture,**record)),flush=True)
            if STOP: raise InterruptedError('Checkpointed epoch stop')
        final_checkpoint=torch.load(target/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(final_checkpoint['model'])
        final_arrays,final_metrics=predict(model,val,config)
        with np.load(target/'final_validation.npz') as saved:
            if not all(np.array_equal(saved[k],final_arrays[k]) for k in final_arrays):
                raise RuntimeError('Final checkpoint replay differs')
        if final_metrics != state['history'][-1]['metrics']:
            raise RuntimeError('Final metrics replay differs')
        selected=torch.load(target/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(selected['model'])
        arrays,metrics=predict(model,val,config)
        with np.load(target/'best_validation.npz') as saved:
            difference=float(np.max(np.abs(saved['logits']-arrays['logits'])))
            if not all(np.array_equal(saved[k],arrays[k]) for k in arrays):
                raise RuntimeError('Selected checkpoint replay differs')
        if metrics != selected['metrics']: raise RuntimeError('Replayed metrics differ')
        after={p:file_sha256(p) for p in bundle['preservation']}
        if after!=bundle['preservation'] or binding(bundle,config,architecture)!=bound:
            raise RuntimeError('Input/code preservation mismatch')
        atomic_json(target/'preservation_after.json',after)
        precision,recall,f1,support=precision_recall_fscore_support(arrays['y'],arrays['pred'],labels=np.arange(nclasses),zero_division=0)
        perclass=[dict(index=i,label=bundle['contract']['classes'][i],precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i]),support=int(support[i]),train_support=manifest['class_counts_train'][i]) for i in range(nclasses)]
        rare=np.argsort(manifest['class_counts_train'],kind='stable')[:max(1,nclasses//4)]
        detail=dict(selected=metrics,final=state['history'][-1]['metrics'],per_class=perclass,
                    rare_policy='lowest train-support floor(C/4) classes; stable class-index ties',rare_indices=rare.tolist(),
                    rare_macro_f1=float(f1[rare].mean()),rare_balanced_acc=float(recall[rare].mean()),
                    class_convention='all canonical classes present in full validation; shared six-decimal metrics')
        atomic_json(target/'metrics.json',detail)
        model.load_state_dict(final_checkpoint['model'])
        persist('completed',selected_metrics=metrics,final_metrics=detail['final'],best_epoch=state['best_epoch'],
                final_replay=dict(arrays_equal=True,metrics_equal=True,n_evaluated=len(val),tolerance=0.),
                replay=dict(logits_max_abs_diff=difference,arrays_equal=True,metrics_equal=True,n_evaluated=len(val),tolerance=0.),
                checkpoint_sha256=file_sha256(target/'best.pt'),checkpoint_bytes=(target/'best.pt').stat().st_size,
                preservation_verified=True)
        return manifest
    except BaseException as exc:
        persist('interrupted' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'failed',error=str(exc))
        raise


def summarize(root):
    manifests=[json.loads((root/a/'run_manifest.json').read_text()) for a in ('interaction','plain')]
    if any(m['status']!='completed' for m in manifests): raise ValueError('Both cells must complete')
    rows=[dict(architecture=m['architecture'],**m['selected_metrics'],best_epoch=m['best_epoch'],
               seconds=m['seconds'],parameters=m['parameter_count']) for m in manifests]
    delta={k:rows[0][k]-rows[1][k] for k in manifests[0]['selected_metrics']}
    atomic_json(root/'results.json',dict(rows=rows,interaction_minus_plain=delta,test_evaluated=False))
    with (root/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    text='# PNA tam veri performans karşılaştırması\n\n59.607 train / 7.448 validation; test değerlendirilmedi. Seed1234, 30 epoch, batch128, Adam lr0.001 wd1e-5, CE; aynı star/common input.\n\n| Model | Macro-F1 | Balanced accuracy | Accuracy | Best epoch | Parametre |\n|---|---:|---:|---:|---:|---:|\n'
    for r in rows: text+=f"| {r['architecture']} | {r['macro_f1']:.6f} | {r['balanced_acc']:.6f} | {r['accuracy']:.6f} | {r['best_epoch']} | {r['parameters']} |\n"
    text+='\nInteraction − plain: `'+json.dumps(delta)+'`.\n\nSeçim: ilk maksimum shared macro-F1 (6 ondalık); 30 sınıfın tamamı validation içinde. Per-class, train-en-az 7 sınıf rare metrikleri, final epoch ve süreler hücre metrics/manifest dosyalarında. Seçilmiş checkpoint full validation üzerinde birebir yeniden üretildi.\n\nTek seed ve farklı parametre sayıları: yöntem üstünlüğü ispatı değil. Tarihsel GraphCare 0.481306 farklı sqrt-loss referansıdır, eş-loss kontrol değildir. Kaynak snapshot temporal_clean=false/raw_to_model_train_only=false; klinik erken-tahmin iddiası yok.\n\nArtifact: `'+str(root.relative_to(ROOT))+'`.\n'
    (root/'pna-performance-results.md').write_text(text)
    (ROOT/'docs/pna-performance-results.md').write_text(text)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    root=args.output_dir.resolve()
    allowed=ROOT/'comparison/standardized/pna_experiments'
    if allowed.resolve() not in root.parents: raise ValueError('Output must be isolated under pna_experiments')
    if not args.execute:
        print(json.dumps(dict(config=CONFIG,architectures=['interaction','plain'],counts=[59607,7448],test_evaluated=False))); return
    if root.exists() and not args.resume: raise FileExistsError(root)
    root.mkdir(parents=True,exist_ok=True)
    def stop(signum,frame):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    bundle=load_common_input(ROOT/'comparison/standardized/common_input_20260913',ROOT/'data/merged_ed.csv',ROOT/'comparison/canonical_split.json')
    if bundle['contract']['counts'][:2]!=[59607,7448] or len(bundle['contract']['classes'])!=30:
        raise ValueError('Approved full-data contract mismatch')
    atomic_json(root/'launch.json',dict(config=CONFIG,architectures=['interaction','plain'],pid=os.getpid(),argv=sys.argv))
    print(json.dumps(dict(event='building_full_train_validation_graphs',counts=[59607,7448])),flush=True)
    graphs=[graph_subset(bundle,np.flatnonzero(bundle['folds']==f)) for f in (0,1)]
    print(json.dumps(dict(event='graph_cache_ready',counts=[len(g) for g in graphs],tensor_bytes=sum(t.numel()*t.element_size() for fold in graphs for g in fold for _,t in g if torch.is_tensor(t)))),flush=True)
    for architecture in ('interaction','plain'):
        target=root/architecture
        train_cell(target,bundle,architecture,resume=args.resume and target.exists(),graphs=graphs)
        if STOP: raise InterruptedError('Stopped before next cell')
    summarize(root)
    print(json.dumps(dict(event='paired_benchmark_completed',root=str(root))),flush=True)


if __name__=='__main__':
    main()
