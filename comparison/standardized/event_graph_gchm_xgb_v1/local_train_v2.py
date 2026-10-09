"""Local, one-seed EventGCHM training with bounded memory and exact update resume.

No XGBoost, test evaluation, remote execution, or implicit package installation.
CPU is deliberately selected for bounded local resources and portable PNA scatter.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import random
import signal
import sqlite3
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from sklearn.metrics import f1_score

from .labels import DEFAULT_GRAPH_ROOT, REPO, sha256
from .features import load_binding
from event_graph_analysis.tensorize import EventGraphTensorizer
from event_graph_analysis.model import EventGCHM

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.chmod(0o600)
    os.replace(tmp, path)


def atomic_torch(path, obj):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.chmod(0o600)
    os.replace(tmp, path)


def emit(root, event, **fields):
    row = dict(event=event, unix_time=time.time(), pid=os.getpid(), **fields)
    line = json.dumps(row, allow_nan=False)
    with (root / 'progress.jsonl').open('a') as f:
        f.write(line + '\n')
    print(line, flush=True)


def source_binding(root, binding_root, gm, bm, args):
    sources = [Path(__file__), Path(__file__).with_name('local_labels_v2.py'), Path(__file__).with_name('labels.py'),
               Path(__file__).with_name('features.py'), REPO / 'event_graph_analysis/model.py', REPO / 'event_graph_analysis/tensorize.py']
    return {'sources': {str(p.relative_to(REPO)): sha256(p) for p in sources},
            'graph_manifest_sha256': sha256(DEFAULT_GRAPH_ROOT / 'manifest.json'),
            'graphs_sha256': gm['graphs_sha256'], 'cohort_sha256': gm['cohort_sha256'],
            'binding_manifest_sha256': sha256(binding_root / 'binding_manifest.json'),
            'targets_sha256': bm['artifact_files']['targets.csv'], 'labels': bm['labels'],
            'seed': 1234, 'epochs': args.epochs, 'microbatch': 1, 'accumulation': args.accumulation,
            'device': 'cpu', 'threads': args.threads, 'weight_policy': 'sqrt_inverse_train_only',
            'hidden_dim': 64, 'relation_dim': 16, 'layers': 3, 'dropout': 0.1,
            'lr': 0.001, 'weight_decay': 0.00001, 'torch_version': str(torch.__version__)}


def prepare_index(root, binding, gm):
    """Durable per-line index; resume scans only its unfinished suffix.

    SHA of the full immutable file is checked at completion and every process
    start. Per-record hashes protect lazy reads; no worker reindexes JSONL.
    """
    path = DEFAULT_GRAPH_ROOT / 'graphs.jsonl'
    db = sqlite3.connect(root / 'graph_index.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS graphs (sample TEXT PRIMARY KEY, offset INTEGER UNIQUE, end_offset INTEGER, digest TEXT, split TEXT, target INTEGER, nodes INTEGER, edges INTEGER)')
    db.commit()
    done = db.execute('SELECT COUNT(*), COALESCE(MAX(end_offset),0) FROM graphs').fetchone()
    count, end = done
    emit(root, 'index_start', indexed=count, byte_offset=end)
    with path.open('rb') as f:
        f.seek(end)
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            g = json.loads(line)
            row = binding.get(g['sample_id'])
            if row is None or row['split'] != g['split'] or g['schema_version'] != 'event_graph_v1' or 'target' in g:
                raise ValueError('Graph/binding mismatch or graph contains target')
            # Cohort sample identity and subject/stay are already exactly bound by labels.
            db.execute('INSERT INTO graphs VALUES (?,?,?,?,?,?,?,?)', (g['sample_id'], offset, f.tell(), hashlib.sha256(line).hexdigest(), g['split'], row['target'], len(g['nodes']), len(g['edges'])))
            count += 1
            if count % 5000 == 0:
                db.commit()
                emit(root, 'index_progress', indexed=count)
            if STOP:
                db.commit()
                raise InterruptedError('Preparation stopped at durable index boundary')
    db.commit()
    if count != gm['counts']['graphs']:
        raise ValueError('Incomplete graph count')
    emit(root, 'graph_hash_start', graphs=count)
    if sha256(path) != gm['graphs_sha256']:
        raise ValueError('Graph fingerprint mismatch')
    records = {}
    for split in ('train', 'validation'):
        rows = db.execute('SELECT offset,digest,target,nodes,edges FROM graphs WHERE split=? AND target>=0 ORDER BY offset', (split,)).fetchall()
        records[split] = rows
    coverage = dict(db.execute('SELECT split,COUNT(*) FROM graphs GROUP BY split').fetchall())
    if any(coverage.get(k) != v['graphs'] for k,v in gm['coverage'].items()):
        raise ValueError('Fold coverage mismatch')
    db.close()
    emit(root, 'index_completed', counts={k: len(v) for k,v in records.items()})
    return records


def read_graph(stream, record):
    offset, digest = record[:2]
    stream.seek(offset)
    line = stream.readline()
    if hashlib.sha256(line).hexdigest() != digest:
        raise ValueError('Graph changed after indexing')
    return json.loads(line)


def prepare_adapter(root, records, base_binding):
    path = root / 'preprocessing.pt'
    if path.exists():
        obj = torch.load(path, map_location='cpu', weights_only=False)
        if obj['binding'] != base_binding or obj['fit_graph_count'] != len(records):
            raise ValueError('Preprocessing source mismatch')
        return EventGraphTensorizer.from_dict(obj['adapter']), torch.tensor(obj['degree_histogram'], dtype=torch.float32)
    histogram = np.zeros(1, dtype=np.int64)
    start = time.monotonic()
    def graphs():
        nonlocal histogram
        with (DEFAULT_GRAPH_ROOT / 'graphs.jsonl').open('rb') as stream:
            for i, record in enumerate(records):
                graph = read_graph(stream, record)
                index = {n['id']: j for j,n in enumerate(graph['nodes'])}
                degrees = np.zeros(len(index), dtype=np.int64)
                # Reverse edges are enabled: each original edge contributes to both destinations.
                for edge in graph['edges']:
                    degrees[index[edge['target']]] += 1
                    degrees[index[edge['source']]] += 1
                counts = np.bincount(degrees)
                if len(histogram) < len(counts):
                    histogram = np.pad(histogram, (0,len(counts)-len(histogram)))
                histogram[:len(counts)] += counts
                yield graph
                if (i + 1) % 2000 == 0:
                    emit(root, 'scaler_progress', fitted=i+1, total=len(records), elapsed_seconds=time.monotonic()-start)
                if STOP:
                    raise InterruptedError('Scaler interrupted; completed index preserved; scaler restarts train-only fit')
    emit(root, 'scaler_start', train_graphs=len(records))
    adapter = EventGraphTensorizer(add_reverse_edges=True).fit(graphs())
    obj = {'adapter': adapter.to_dict(), 'degree_histogram': histogram.tolist(), 'binding': base_binding,
           'fit_split': 'train', 'fit_graph_count': len(records), 'histogram_policy': 'bincount(in_degree) over nodes, including reverse edges'}
    atomic_torch(path, obj)
    atomic_json(root / 'adapter.json', {k: v for k,v in obj.items() if k != 'binding'})
    emit(root, 'scaler_completed', tokens=adapter.num_tokens, relations=adapter.num_relations)
    return adapter, torch.tensor(histogram, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binding-root', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Explicit --execute required')
    if args.epochs < 1 or args.threads < 1 or args.accumulation < 1:
        parser.error('Positive sizes required')
    root = args.output.resolve()
    if root.exists() and not args.resume:
        raise FileExistsError('Occupied output; use --resume for same approved run')
    root.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    atomic_json(root / 'process.json', {'pid': os.getpid(), 'argv': sys.argv, 'python': sys.executable, 'hostname': os.uname().nodename, 'device': 'cpu', 'threads': args.threads})
    binding, bm = load_binding(args.binding_root)
    gm = json.loads((DEFAULT_GRAPH_ROOT / 'manifest.json').read_text())
    if bm['graph_sha256'] != gm['graphs_sha256'] or bm['labels'] != gm['reference_class_order_only']:
        raise ValueError('Input artifact mismatch')
    if sha256(DEFAULT_GRAPH_ROOT / 'cohort.csv') != gm['cohort_sha256']:
        raise ValueError('Cohort changed')
    folds = {}
    for row in binding.values():
        previous = folds.setdefault(row['subject_id'], row['split'])
        if previous != row['split']:
            raise ValueError('Subject fold overlap')
    base = source_binding(root, args.binding_root, gm, bm, args)
    if (root / 'binding.json').exists():
        if json.loads((root / 'binding.json').read_text()) != base:
            raise ValueError('Run source/input/config binding changed')
    else:
        atomic_json(root / 'binding.json', base)
        snap = root / 'source_snapshot'
        for rel in base['sources']:
            dest = snap / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((REPO / rel).read_bytes())
    state = {'status': 'preparing', 'method': 'EventGCHM', 'seed': 1234, 'epochs': args.epochs,
             'device': 'cpu', 'threads': args.threads, 'test_evaluated': False, 'temporal_clean': False,
             'target_binding_manifest': str(args.binding_root.resolve() / 'binding_manifest.json'),
             'graph_root': str(DEFAULT_GRAPH_ROOT), 'checkpoint_path': str(root / 'last.pt'),
             'selection': 'validation_macro_f1_all_30_classes', 'counts': bm['counts']['by_split']}
    atomic_json(root / 'run_manifest.json', state)
    try:
        records = prepare_index(root, binding, gm)
        del binding, folds
        for split, rs in records.items():
            if len(rs) != bm['counts']['by_split'][split]['samples']:
                raise ValueError('Labelled split count mismatch')
        adapter, hist = prepare_adapter(root, records['train'], base)
        full_binding = {**base, 'preprocessing_sha256': sha256(root / 'preprocessing.pt')}
        model = EventGCHM(adapter.num_tokens, adapter.num_relations, len(bm['labels']), degree_histogram=hist,
                          hidden_dim=64, relation_dim=16, num_layers=3, dropout=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        targets = np.asarray([r[2] for r in records['train']], dtype=np.int64)
        counts = np.bincount(targets, minlength=len(bm['labels']))
        if (counts == 0).any():
            raise ValueError('Missing training class')
        w = 1 / np.sqrt(counts.astype(float))
        w /= w.mean()
        weights = torch.tensor(w, dtype=torch.float32)
        atomic_json(root / 'train_class_weights.json', {'counts': counts.tolist(), 'weights': weights.tolist(), 'fit_split': 'train'})
        epoch, cursor, step, permutation = 0, 0, 0, None
        best_score, best_epoch, history = -1.0, -1, []
        epoch_loss, epoch_updates = 0.0, 0
        if args.resume and (root / 'last.pt').exists():
            ck = torch.load(root / 'last.pt', map_location='cpu', weights_only=False)
            if ck['binding'] != full_binding:
                raise ValueError('Checkpoint binding mismatch')
            model.load_state_dict(ck['model'])
            optimizer.load_state_dict(ck['optimizer'])
            epoch, cursor, step, permutation = ck['epoch'], ck['cursor'], ck['optimizer_steps'], ck['permutation']
            best_score, best_epoch, history = ck['best_score'], ck['best_epoch'], ck['history']
            epoch_loss, epoch_updates = ck['epoch_loss'], ck['epoch_updates']
            random.setstate(ck['rng_python'])
            np.random.set_state(ck['rng_numpy'])
            torch.set_rng_state(ck['rng_torch'])
            emit(root, 'resumed', optimizer_steps=step, epoch=epoch, cursor=cursor)
        last_save = time.monotonic()
        started = time.monotonic()
        def save(status='running'):
            nonlocal last_save
            ck = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'model_config': model.config,
                  'binding': full_binding, 'epoch': epoch, 'cursor': cursor, 'optimizer_steps': step,
                  'permutation': permutation, 'best_score': best_score, 'best_epoch': best_epoch, 'history': history,
                  'epoch_loss': epoch_loss, 'epoch_updates': epoch_updates, 'class_weights': weights,
                  'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(), 'rng_torch': torch.get_rng_state(),
                  'gradient_boundary': True, 'sampler_policy': 'one seeded randperm per epoch, next position cursor',
                  'adapter': adapter.to_dict()}
            atomic_torch(root / 'last.pt', ck)
            state.update(status=status, epoch=epoch, cursor=cursor, optimizer_steps=step,
                         best_epoch=best_epoch, last_checkpoint_unix=time.time())
            atomic_json(root / 'run_manifest.json', state)
            last_save = time.monotonic()
            emit(root, 'checkpoint', optimizer_steps=step, epoch=epoch, cursor=cursor, status=status)
        emit(root, 'training_start', parameters=sum(p.numel() for p in model.parameters()), device='cpu', threads=args.threads,
             mps_available=bool(torch.backends.mps.is_available()), microbatch=1, accumulation=args.accumulation)
        with (DEFAULT_GRAPH_ROOT / 'graphs.jsonl').open('rb') as stream:
            while epoch < args.epochs:
                if permutation is None:
                    permutation = torch.randperm(len(records['train'])).tolist()
                    cursor, epoch_loss, epoch_updates = 0, 0.0, 0
                model.train()
                while cursor < len(permutation):
                    selected = permutation[cursor:cursor+args.accumulation]
                    denominator = weights[torch.tensor(targets[selected])].sum()
                    optimizer.zero_grad(set_to_none=True)
                    loss_value = 0.0
                    nodes, edges = 0, 0
                    for index in selected:
                        record = records['train'][index]
                        graph = read_graph(stream, record)
                        graph['target'] = int(record[2])
                        batch = Batch.from_data_list([adapter.transform(graph)])
                        del graph
                        logits = model(batch)
                        loss = F.cross_entropy(logits, batch.y.view(-1), weight=weights, reduction='sum') / denominator
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite loss')
                        loss.backward()
                        loss_value += float(loss.detach())
                        nodes += batch.num_nodes
                        edges += batch.num_edges
                        del batch, logits, loss
                    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise ValueError('Nonfinite gradients')
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
                    optimizer.step()
                    cursor += len(selected)
                    step += 1
                    epoch_updates += 1
                    epoch_loss += loss_value
                    if step <= 10 or step % 10 == 0:
                        emit(root, 'optimizer_update', epoch=epoch, optimizer_steps=step, cursor=cursor,
                             loss=loss_value, gradient_norm=grad_norm, graphs=len(selected), nodes=nodes, edges=edges,
                             elapsed_training_seconds=time.monotonic()-started)
                    if step == 1 or step % 25 == 0 or time.monotonic()-last_save >= 60 or STOP:
                        save('interrupted' if STOP else 'running')
                    if STOP:
                        return
                save()
                emit(root, 'validation_start', epoch=epoch, samples=len(records['validation']))
                model.eval()
                probabilities, truth = [], []
                with torch.no_grad():
                    for record in records['validation']:
                        graph = read_graph(stream, record)
                        batch = Batch.from_data_list([adapter.transform(graph)])
                        probabilities.append(model(batch).softmax(-1).numpy())
                        truth.append(record[2])
                        if STOP:
                            save('interrupted')
                            return
                proba = np.concatenate(probabilities)
                y = np.asarray(truth)
                if not np.isfinite(proba).all():
                    raise ValueError('Nonfinite validation prediction')
                score = float(f1_score(y, proba.argmax(1), labels=np.arange(len(bm['labels'])), average='macro', zero_division=0))
                row = {'epoch': epoch, 'optimizer_steps': step, 'train_loss': epoch_loss/epoch_updates, 'validation_macro_f1': score}
                history.append(row)
                if score > best_score:
                    best_score, best_epoch = score, epoch
                    atomic_torch(root / 'best.pt', {'model': model.state_dict(), 'model_config': model.config, 'binding': full_binding,
                                                   'epoch': epoch, 'validation_macro_f1': score, 'adapter': adapter.to_dict()})
                    np.savez_compressed(root / 'validation.npz', proba=proba, y=y)
                atomic_json(root / 'history.json', history)
                emit(root, 'epoch_completed', **row)
                epoch += 1
                permutation = None
                cursor = 0
                save()
        save('completed')
    except InterruptedError as exc:
        state.update(status='interrupted', reason=str(exc))
        atomic_json(root / 'run_manifest.json', state)
        emit(root, 'interrupted', reason=str(exc))
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        atomic_json(root / 'run_manifest.json', state)
        emit(root, 'failed', error=repr(exc))
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
